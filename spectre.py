import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Union
import argparse
from scipy.special import erfc
from sklearn.utils.extmath import randomized_svd
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_random_state
import torchvision
import torchvision.transforms as transforms


class BeingRobust(EmpiricalCovariance):

    def __init__(self,
                 eps: float = 0.1,
                 tau: float = 0.1,
                 cher: float = 2.5,
                 use_randomized_svd: bool = True,
                 debug: bool = False,
                 assume_centered: bool = False,
                 random_state: Union[int, np.random.RandomState] = None,
                 keep_filtered: bool = False):
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.cher = cher
        self.use_randomized_svd = use_randomized_svd
        self.debug = debug
        self.random_state = random_state
        self.assume_centered = assume_centered
        self.keep_filtered = keep_filtered

    def fit(self, X, y=None) -> 'BeingRobust':
        """Fits the data to obtain the robust estimate.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: Ignored
            Not used, present for API consistence purpose.
        Returns
        -------
        self : BeingRobust
        """
        X = self._validate_data(X, ensure_min_samples=1, estimator='BeingRobust')
        random_state = check_random_state(self.random_state)

        self.location_, X = filter_gaussian_mean(X,
                                                 eps=self.eps,
                                                 tau=self.tau,
                                                 cher=self.cher,
                                                 use_randomized_svd=self.use_randomized_svd,
                                                 debug=self.debug,
                                                 assume_centered=self.assume_centered,
                                                 random_state=random_state)
        if self.keep_filtered:
            self.filtered_ = X

        return self


def filter_gaussian_mean(X: np.ndarray,
                         eps: float = 0.1,
                         tau: float = 0.1,
                         cher: float = 2.5,
                         use_randomized_svd: bool = True,
                         debug: bool = False,
                         assume_centered: bool = False,
                         random_state: int = None) -> Tuple[float, np.ndarray]:
    """Being Robust (in High Dimensions) Can Be Practical: robust estimator of location (and potentially covariance).
    This estimator is to be applied on Gaussian-distributed data. For other distributions some changes might be
    required. Please check out the original paper and/or Matlab code.
    Parameters
    ----------
    eps : float, optional
        Fraction of perturbed data points, by default 0.1
    tau : float, optional
        Significance level, by default 0.1
    cher : float, optional
        Factor filter criterion, by default 2.5
    use_randomized_svd : bool, optional
        If True use `sklearn.utils.extmath.randomized_svd`, else use full SVD, by default True
    debug : bool, optional
        If True print debug information, by default False
    assume_centered : bool
        If True, the data is not centered beforehand, by default False
    random_state : Union[int, np.random.RandomState],
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls. By default none
    Returns
    -------
    Tuple[float, np.ndarray]
        The robust location estimate, the filtered version of `X`
    """
    n_samples, n_features = X.shape

    emp_mean = X.mean(axis=0)

    if assume_centered:
        centered_X = X
    else:
        centered_X = (X - emp_mean) / np.sqrt(n_samples)

    if use_randomized_svd:
        U, S, Vh = randomized_svd(centered_X.T, n_components=1, random_state=random_state)
    else:
        U, S, Vh = np.linalg.svd(centered_X.T, full_matrices=False)

    lambda_ = S[0]**2
    v = U[:, 0]

    if debug:
        print(f'\nRecursing on X of shape {X.shape}')
        print(f'lambda_ < 1 + 3 * eps * np.log(1 / eps) -> {lambda_} < {1 + 3 * eps * np.log(1 / eps)}')
    if lambda_ < 1 + 3 * eps * np.log(1 / eps):
        return emp_mean, X

    delta = 2 * eps
    if debug:
        print(f'delta={delta}')

    projected_X = X @ v
    med = np.median(projected_X)
    projected_X = np.abs(projected_X - med)
    sorted_projected_X_idx = np.argsort(projected_X)
    sorted_projected_X = projected_X[sorted_projected_X_idx]

    for i in range(n_samples):
        T = sorted_projected_X[i] - delta
        filter_crit_lhs = n_samples - i
        filter_crit_rhs = cher * n_samples * \
            erfc(T / np.sqrt(2)) / 2 + eps / (n_samples * np.log(n_samples * eps / tau))
        if filter_crit_lhs > filter_crit_rhs:
            break

    if debug:
        print(f'filter data at index {i}')

    if i == 0 or i == n_samples - 1:
        return emp_mean, X

    return filter_gaussian_mean(
        X[sorted_projected_X_idx[:i + 1]],
        eps=eps,
        tau=tau,
        cher=cher,
        use_randomized_svd=use_randomized_svd,
        debug=debug,
        assume_centered=assume_centered,
        random_state=random_state
    )

def get_model(name):
    if name == "convnet":
        return ConvNet()
    else:
        return AlexNet()

class AlexNet(nn.Module):
    def __init__(self, channel=3, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x1 = self.classifier(x)
        return x1


class ConvNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='none', net_pooling='avgpooling', im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Sequential(
            nn.Linear(num_feat, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out1 = self.classifier(out)
        return out1

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


def get_features(x_train, y_train, model, num_classes):

    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        # _, x_feats = model(x_train)
        x_feats = _get_middle_output(x_train, model, -2)
        for i in range(len(x_train)):
            feats.append(x_feats[i].cpu().numpy())
            class_indices[y_train[i]].append(i)

    return feats, class_indices

def get_features_2(data_loader, model, num_classes):

    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        count = 0
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            # _, x_feats = model(ins_data)
            x_feats = _get_middle_output(ins_data, model, -2)
            this_batch_size = len(ins_target)
            
            for bid in range(this_batch_size):
                b_target = ins_target[bid].item()
                if b_target >= num_classes:
                    continue
                feats.append(x_feats[bid].cpu().numpy())
                class_indices[b_target].append(count)
                count += 1
    return feats, class_indices


def QUEscore(temp_feats, n_dim):

    n_samples = temp_feats.shape[1]
    alpha = 4.0
    Sigma = torch.matmul(temp_feats, temp_feats.T) / n_samples
    I = torch.eye(n_dim).cuda()
    Q = torch.exp((alpha * (Sigma - I)) / (torch.linalg.norm(Sigma, ord=2) - 1))
    trace_Q = torch.trace(Q)

    taus = []
    for i in range(n_samples):
        h_i = temp_feats[:, i:i + 1]
        tau_i = torch.matmul(h_i.T, torch.matmul(Q, h_i)) / trace_Q
        tau_i = tau_i.item()
        taus.append(tau_i)
    taus = np.array(taus)

    return taus

def SPECTRE(U, temp_feats, n_dim, budget, oracle_clean_feats=None):

    projector = U[:, :n_dim].T # top left singular vectors
    temp_feats = torch.matmul(projector, temp_feats)

    if oracle_clean_feats is None:
        estimator = BeingRobust(random_state=0, keep_filtered=True).fit((temp_feats.T).cpu().numpy())
        clean_mean = torch.FloatTensor(estimator.location_).cuda()
        filtered_feats = (torch.FloatTensor(estimator.filtered_).cuda() - clean_mean).T
        clean_covariance = torch.cov(filtered_feats)
    else:
        clean_feats = torch.matmul(projector, oracle_clean_feats)
        clean_covariance = torch.cov(clean_feats)
        clean_mean = clean_feats.mean(dim = 1)


    temp_feats = (temp_feats.T - clean_mean).T

    # whiten the data
    L, V = torch.linalg.eig(clean_covariance)
    L, V = L.real, V.real
    L = (torch.diag(L)**(1/2)+0.001).inverse()
    normalizer = torch.matmul(V, torch.matmul( L, V.T ) )
    temp_feats = torch.matmul(normalizer, temp_feats)

    # compute QUEscore
    taus = QUEscore(temp_feats, n_dim)

    sorted_indices = np.argsort(taus)
    n_samples = len(sorted_indices)

    budget = min(budget, n_samples//2) # default assumption : at least a half of samples in each class is clean

    suspicious = sorted_indices[-budget:]
    left = sorted_indices[:n_samples-budget]

    return suspicious, left

def _get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output
        return hook

def _get_middle_output(x, model, layer):
        temp = []

        for name, _ in model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if -layer > len(temp):
            raise IndexError('layer is out of range')

        name = temp[layer].split('.')
        var = eval('model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(_get_activation(str(layer), out))
        
        _ = model(x)

        return out[str(layer)]


def cleanser(num_classes, args, oracle_clean_set):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    model = get_model(args.m)
    model.load_state_dict(torch.load(args.mn + ".pth"))
    model = model.to(device)

    trigger_x_train = torch.load(args.mn + "_img.pth")
    trigger_y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1)

    benign_x_train = torch.load(args.bn + ".pth")
    benign_y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1)

    x_train = torch.cat([trigger_x_train[:10], benign_x_train[:10]])
    y_train = torch.cat([trigger_y_train[:10], benign_y_train[:10]])
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    model = model.to(device)

    # inspection_split_loader = torch.utils.data.DataLoader(
    #     inspection_set,
    #     batch_size=128, shuffle=False, **kwargs)

    feats, class_indices = get_features(x_train, y_train, model, num_classes)

    clean_set_loader = torch.utils.data.DataLoader(
        oracle_clean_set,
        batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    clean_feats, clean_class_indices = get_features_2(clean_set_loader, model, num_classes)

    suspicious_indices = []
    # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
    budget = int(0.01 * len(oracle_clean_set) * 1.5)
    #print(budget)
    # allow removing additional 50% (following the original paper)

    max_dim = 2 # 64
    class_taus = []
    class_S = []
    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            # feats for class i in poisoned set
            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats).cuda()

            temp_clean_feats = None
            if oracle_clean_set is not None:
                temp_clean_feats = np.array([clean_feats[temp_idx] for temp_idx in clean_class_indices[i]])
                temp_clean_feats = torch.FloatTensor(temp_clean_feats).cuda()
                temp_clean_feats = temp_clean_feats - temp_feats.mean(dim=0)
                temp_clean_feats = temp_clean_feats.T

            temp_feats = temp_feats - temp_feats.mean(dim=0) # centered data
            temp_feats = temp_feats.T # feats arranged in column

            U, _, _ = torch.svd(temp_feats)
            U = U[:, :max_dim]

            # full projection
            projected_feats = torch.matmul(U.T, temp_feats)

            max_tau = -999999
            best_n_dim = -1
            best_to_be_removed = None

            for n_dim in range(2, max_dim+1): # enumarate all possible "reudced dimensions" and select the best

                S_removed, S_left = SPECTRE(U, temp_feats, n_dim, budget, temp_clean_feats)

                left_feats = projected_feats[:, S_left]
                covariance = torch.cov(left_feats)

                L, V = torch.linalg.eig(covariance)
                L, V = L.real, V.real
                L = (torch.diag(L) ** (1 / 2) + 0.001).inverse()
                normalizer = torch.matmul(V, torch.matmul(L, V.T))

                whitened_feats = torch.matmul(normalizer, projected_feats)

                tau = QUEscore(whitened_feats, max_dim).mean()

                if tau > max_tau:
                    max_tau = tau
                    best_n_dim = n_dim
                    best_to_be_removed = S_removed


            print('class=%d, dim=%d, tau=%f' % (i, best_n_dim, max_tau))

            class_taus.append(max_tau)

            suspicious_indices = []
            for temp_index in best_to_be_removed:
                suspicious_indices.append(class_indices[i][temp_index])

            class_S.append(suspicious_indices)

    class_taus = np.array(class_taus)
    for num in class_taus:
        print(num,end=',')
    median_tau = np.median(class_taus)

    #print('median_tau : %d' % median_tau)
    suspicious_indices = []
    max_tau = -99999
    for i in range(num_classes):
        #if class_taus[i] > max_tau:
        #    max_tau = class_taus[i]
        #    suspicious_indices = class_S[i]
        #print('class-%d, tau = %f' % (i, class_taus[i]))
        #if class_taus[i] > 2*median_tau:
        #    print('[large tau detected] potential poisons! Apply Filter!')
        for temp_index in class_S[i]:
            suspicious_indices.append(temp_index)

    return suspicious_indices

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=False, default='cifar10')
    parser.add_argument('-m', type=str, required=False, default='alexnet')
    parser.add_argument('-a', type=str, required=False, default='dc')
    args = parser.parse_args()

    args.mn = ""
    args.bn = "b_"

    if args.d == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        clean_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        if args.m == "alexnet":
            args.mn += "a"
        else:
            args.mn += "c"
        args.mn += "_"

        if args.a == "dd":
            args.mn += "dd"
        else:
            args.mn += "dc"

        args.mn += "_cifar"
        
        
    elif args.d == "stl10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        clean_set = torchvision.datasets.STL10(root='./data', split='test',
                                        download=False, transform=transform)

        if args.m == "alexnet":
            args.mn += "a"
        else:
            args.mn += "c"

        args.mn += "_"

        if args.a == "dd":
            args.mn += "dd"
        else:
            args.mn += "dc"

        args.mn += "_stl"
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4379104971885681, 0.44398033618927, 0.4729299545288086),(0.19803012907505035, 0.2010156363248825, 0.19703614711761475))])
        clean_set = torchvision.datasets.SVHN(root='./data', split='test',
                                        download=True, transform=transform)

        if args.m == "alexnet":
            args.mn += "a"
        else:
            args.mn += "c"

        args.mn += "_"

        if args.a == "dd":
            args.mn += "dd"
        else:
            args.mn += "dc"

        args.mn += "_svhn"

    args.bn += args.mn
    suspicious_indices = cleanser(1, args, clean_set)
    print(suspicious_indices)
