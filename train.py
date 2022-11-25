import os
import cv2
import json
import torch
import scipy
import argparse
import scipy.stats
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms


from spectre import _get_middle_output
from art.defences.detector import poison
from models import get_model, denoising_model
from art.defences.transformer import poisoning
from torch.utils.data import DataLoader, TensorDataset
from art.estimators.classification import PyTorchClassifier
from utlis import get_data, CIFAR10_noise, STL10_noise, SVHN_noise, FashionMNIST_noise


def train(model, target_label, train_loader, param, device):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    nc = param["nc"]
    trigger = torch.rand((nc, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    min_norm = np.inf
    min_norm_count = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(100):
        norm = 0.0
        for images, _ in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + 0.01 * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        # print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()

def reverse_engineer(device, args):
    model = get_model(args.param["name"], args.nc)
    model.load_state_dict(torch.load("pth/" + args.mn+'.pth'))
    model = model.to(device)
    _, _, x_test, y_test = get_data(args.param)
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    nc = args.param["nc"]
    if nc == 1:
        x_test = torch.unsqueeze(x_test, 1)
    train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.param["batch_size"], shuffle=False)
    mask_flatten = []
    norm_list = []
    idx_mapping = {}
    for label in range(args.param["num_classes"]):
        trigger, mask = train(model, label, train_loader, args.param, device)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        # plt.imshow(trigger)
        plt.savefig('mask/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        # plt.imshow(mask)
        plt.savefig('mask/mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask_flatten.append(mask.flatten())

        idx_mapping[label] = len(mask_flatten) - 1

    print(norm_list)

    outlier_detection(norm_list, idx_mapping)

def outlier_detection(l1_norm_list, idx_mapping):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

def Spectral_Signatures(device, args):
    
    x_train = torch.load('pth/' + args.mn + '_img.pth')

    model = get_model(args.param["name"], args.nc)
    model.load_state_dict(torch.load('pth/' + args.mn + '.pth'))
    model = model.to(device)

    y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1)

    images = x_train.to(device)
    labels = y_train.to(device)

    o_train = model(images)
    min_, max_ = 0.0, 255.0

    # torch.save(nn_model.state_dict(), "./nn_model.pth")

    # is_poison_train, x_train, y_train = generate_backdoor(x_raw, y_raw, 0.01, trigger, mask)
    # x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_, max_),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )


    defence = poison.SpectralSignatureDefense(classifier, x_train.numpy(), y_train.numpy())

    is_poison_train = np.zeros(np.shape(y_train.numpy()))
    is_poison_train[:10] = 1.

    print("------------------- Results using size metric -------------------")
    # print(defence.get_params())
    # report, is_clean_lst = defence.detect_poison()
    # cl = np.array(is_clean_lst)
    # mmsk = cl == 0
    # print(report)
    # print(len(cl[mmsk]))

    # is_clean = is_poison_train == 0

    # defence.plot_clusters(folder="/home/c02yuli/project/ddbd/dataset-distillation/")

    # confusion_matrix = defence.evaluate_defence(is_clean)
    # print("Evaluation defence results for size-based metric: ")
    # jsonObject = json.loads(confusion_matrix)
    # for label in jsonObject:
    #     print(label)
    #     print(jsonObject[label])

    # Visualize clusters:
    # x_train_l = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    # sprites_by_class = defence.visualize_clusters(x_train_l, "cifar10_poison_demo")

    # Show plots for clusters of class 5
    # n_class = 5
    # try:
    #     plt.imshow(sprites_by_class[n_class][0])
    #     plt.title("Class " + str(n_class) + " cluster: 0")
    #     plt.savefig("./1.pdf")
    #     plt.imshow(sprites_by_class[n_class][1])
    #     plt.title("Class " + str(n_class) + " cluster: 1")
    #     plt.savefig("./2.pdf")
    # except ImportError:
    #     print("matplotlib not installed. For this reason, cluster visualization was not displayed")

    # Try again using distance analysis this time:

    score = defence.spectral_signature_scores(o_train.detach().cpu().numpy())
    score = score[:,0]
    poisn_mean = score[:10].mean()
    clean_mean = score[10:].mean()
    print("poisn and clean:")
    print(poisn_mean)
    print(clean_mean)

def superimpose(background, overlay):
    added_image = cv2.addWeighted(background,1,overlay,1,0)
    return added_image

def entropyCal(background, n, x_train, model, device):
    entropy_sum = [0] * n
    x1_add = [0] * n
    for num in range(n):
        x1_add[num] = (superimpose(background, x_train[num]))

    x1_add = torch.from_numpy(np.asarray(x1_add)).to(device).float()
    py1_add = model(x1_add)
    py1_add = F.softmax(py1_add, dim=1)
    py1_add = py1_add.detach().cpu().numpy()
    EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    return EntropySum

def strip(device, args, clean_set):
    x_train = torch.load("pth/" + args.mn + "_img.pth")

    model = get_model(args.m, args.nc)
    model.load_state_dict(torch.load("pth/" + args.mn + ".pth"))
    model = model.to(device)

    # y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1)


    n_test = 10
    n_sample = 10
    
    entropy_trojan = [0] * n_test
    # x_poison = [0] * n_test

    x_train = x_train.numpy()

    index_overlay = np.random.randint(10,99, size=n_sample)
    num = np.array([i for i in range(10, 100)])
    rest = np.setdiff1d(num, index_overlay)

    x_benigh = x_train[rest]
    entropy_benigh = [0] *len(x_benigh)

    for j in range(len(x_benigh)):
        x_background = x_benigh[j]
        entropy_benigh[j] = entropyCal(x_background, n_sample, x_train[index_overlay], model, device)

    for j in range(n_test):
        x_poison = x_train[j]
        entropy_trojan[j] = entropyCal(x_poison, n_sample, x_train[index_overlay], model, device)

    entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
    entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs

    (mu, sigma) = scipy.stats.norm.fit(entropy_trojan)
    # print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    # print(threshold)

    FAR = sum(i > threshold for i in entropy_trojan)

    (mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
    # print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    # print(threshold)
    FRR = sum(i < threshold for i in entropy_benigh)
    print(FRR/n_test)
    print(FAR/n_test)
    

    # bins = 30
    # plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
    # plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
    # plt.legend(loc='upper right', fontsize = 20)
    # plt.ylabel('Probability (%)', fontsize = 20)
    # plt.title('normalized entropy', fontsize = 20)
    # plt.tick_params(labelsize=20)

    # fig1 = plt.gcf()
    # # plt.show()
    # # fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
    # fig1.savefig('EntropyDNNDist_T3.svg')

    # min_benign_entropy = np.mean(entropy_benigh)
    # max_trojan_entropy = np.mean(entropy_trojan)

    # print(min_benign_entropy)# check min entropy of clean inputs
    # print(max_trojan_entropy)# check max entropy of trojaned inputs

def get_poison(x_poison, trigger):
    trigger_loc = (29, 31)
    x_poison[:, 29:31, 29:31] = trigger[:, 29:31, 29:31]
    return x_poison

def strip_for_test(device, args, clean_set):
    model = get_model(args.m, args.nc)
    model.load_state_dict(torch.load("pth/" + args.mn + ".pth"))
    model = model.to(device)

    trigger = torch.load("pth/" + args.mn + "_trigger.pth")[0].detach().cpu().numpy()

    # y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1)
    train_loader = DataLoader(clean_set, batch_size=len(clean_set))
    train_dataset_array = next(iter(train_loader))[0].numpy()
    x_train = train_dataset_array

    l = len(x_train)
    n_test = 200
    n_sample = 10
    entropy_benigh = [0] * n_test
    entropy_trojan = [0] * n_test

    x_benigh = x_train[n_test+n_sample:]
    entropy_benigh = [0] *len(x_benigh)

    for j in range(n_test):
            x_background = x_train[j] 
            entropy_benigh[j] = entropyCal(x_background, n_sample, x_train[n_test*2:n_test*2+n_sample], model, device)

    for j in range(n_test):
        x_poison = x_train[j+n_test]
        x_poison = get_poison(x_poison, trigger)
        entropy_trojan[j] = entropyCal(x_poison, n_sample, x_train[n_test*2:n_test*2+n_sample], model, device)

    entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
    entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs

    (mu, sigma) = scipy.stats.norm.fit(entropy_trojan)
    # print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    # print(threshold)

    FAR = sum(i > threshold for i in entropy_trojan)

    (mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
    # print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    # print(threshold)
    FRR = sum(i < threshold for i in entropy_benigh)
    print(FRR/n_test)
    print(FAR/n_test)
    

    # bins = 30
    # plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
    # plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
    # plt.legend(loc='upper right', fontsize = 20)
    # plt.ylabel('Probability (%)', fontsize = 20)
    # plt.title('normalized entropy', fontsize = 20)
    # plt.tick_params(labelsize=20)

    # fig1 = plt.gcf()
    # # plt.show()
    # # fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
    # fig1.savefig('EntropyDNNDist_T3.svg')

    # min_benign_entropy = np.mean(entropy_benigh)
    # max_trojan_entropy = np.mean(entropy_trojan)

    # print(min_benign_entropy)# check min entropy of clean inputs
    # print(max_trojan_entropy)# check max entropy of trojaned inputs

def test_backdoor(device, args, clean_set):
    model = get_model(args.m, args.nc)
    model.load_state_dict(torch.load("pth/" + args.mn + ".pth"))
    model = model.to(device)
    model.eval()

    input_size = (args.param["image_size"][0], args.param["image_size"][1], args.nc)
    trigger_loc = (args.param["image_size"][0]-3, args.param["image_size"][0]-1)
    init_trigger = np.zeros(input_size)
    init_backdoor = np.random.randint(1, 256,(2, 2, args.nc))
    init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

    mask = torch.FloatTensor(np.float32(init_trigger > 0).transpose((2, 0, 1))).to(device)
    trigger = torch.load("pth/" + args.mn + "_trigger.pth")

    test_loader = DataLoader(clean_set, batch_size=len(clean_set))

    acc_avg = 0
    num_exp = 0

    for img, lab in test_loader:
        img = img.float().to(device)
        lab = lab.long().to(device)
        img[:] = img[:] * (1 - mask) + trigger[0] * mask
        lab[:] = 0
        output = model(img)
        n_b = lab.shape[0]
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        acc_avg += acc
        num_exp += n_b

    acc_avg /= num_exp

    print(acc_avg)

class SCAn:
    def __init__(self):
        self.EPS = 1e-2


    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:,1]
        ai = self.calc_anomaly_index(y/np.max(y))
        return ai


    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a-ma)
        mm = np.median(b)*1.4826
        index = b/mm
        return index


    def build_global_model(self, reprs,labels, n_classes):
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs,axis=0)
        X = reprs-mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L,M])
        for k in range(L):
            idx = (labels==k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N,M])
        e = np.zeros([N,M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]
            e[i] = X[i]-u[i]
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        #EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su+dist_Se > self.EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su,F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k]*Su+Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L,M])
            e = np.zeros([N,M])
            u = np.zeros([N,M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se,G),np.transpose(vec))
                u_m[k] = u_m[k]-np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec-u_m[k]
                u[i] = u_m[k]

            #max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su-last_Su
            dif_Se = Se-last_Se
            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_a

        self.gb_model = gb_model
        return gb_model


    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']
        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs,axis=0)
        X = reprs-mean_a

        class_score = np.zeros([L,3])
        u1 = np.zeros([L,M])
        u2 = np.zeros([L,M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels==k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = [k,i_sc,np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model


    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)
        if (N==1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        #EM
        steps = 0
        while (np.linalg.norm(subg-last_z1) > self.EPS) and (np.linalg.norm((1-subg)-last_z1) > self.EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            #max-step
            #calc u1 and u2
            idx1 = (subg>=0.5)
            idx2 = (subg<0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1,F),np.transpose(u1)) - np.matmul(np.matmul(u2,F),np.transpose(u2))
            e2 = u1-u2
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1,F),np.transpose(e2))
                if bias-2*delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)


    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]
        G = -np.linalg.pinv(N*Su+Se)
        mu = np.zeros([1,M])
        for i in range(N):
            vec = X[i]
            dd = np.matmul(np.matmul(Se,G),np.transpose(vec))
            mu = mu-dd

        b1 = np.matmul(np.matmul(mu,F),np.transpose(mu)) - np.matmul(np.matmul(u1,F),np.transpose(u1))
        b2 = np.matmul(np.matmul(mu,F),np.transpose(mu)) - np.matmul(np.matmul(u2,F),np.transpose(u2))
        n1 = np.sum(subg>=0.5)
        n2 = N-n1
        sc = n1*b1+n2*b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu-u1
            else:
                e2 = mu-u2
            sc -= 2*np.matmul(np.matmul(e1,F),np.transpose(e2))

        return sc/N
    
def test_SCAn(args, clean_set, device):
    model = get_model(args.m, args.nc)
    model.load_state_dict(torch.load("pth/" + args.mn + ".pth"))
    model = model.to(device)
    model.eval()

    x_train = torch.load('pth/' + args.mn + '_img.pth').to(device)
    y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1).numpy()
    #np.array([np.ones(10)*i for i in range(args.param['num_classes'])], dtype=np.int_).reshape(-1)
    o_train = _get_middle_output(x_train, model, -2)

    test_loader = DataLoader(clean_set, batch_size=len(clean_set))

    labs=[]
    testlocal=[]
    for batch_idx, (data, target) in enumerate(test_loader):
        testlocal.append(data)
        labs.append(target)
    datas=[]
    labels=[]
    for i in range(len(labs)):
        for j in range(len(labs[i])):
            if i*len(labs[i])+j>2999:
                break
            datas.append(testlocal[i][j])
            labels.append(labs[i][j])
    reals=[]

    with torch.no_grad():
        for i in range(len(datas)):
            predictions = _get_middle_output(torch.Tensor(datas[i].reshape(1,args.nc,32,32)).to(device), model, -2)
            reals.append(np.squeeze(predictions.cpu().numpy()))
    reals=np.array(reals)
    labels=np.array(labels)

    features= o_train.cpu().detach().numpy()
    fine_label=y_train

    scan=SCAn()
    gb=scan.build_global_model(reals,labels,args.param['num_classes'])
    lc=scan.build_local_model(features,fine_label,gb,args.param['num_classes'])
    ai=scan.calc_final_score(lc)
    print(ai)

def dataset_normalization(d):
    if d == 'cifar10' or d == 'stl10':
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    elif d == 'fmnist':
        mean, std = [0.2861,], [0.3530,]
    else:
        mean, std = [0.4379104971885681, 0.44398033618927, 0.4729299545288086], [0.19803012907505035, 0.2010156363248825, 0.19703614711761475]

    return mean, std

def train_epoch(mode, x_train, y_train, net, optimizer, criterion, device):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    n_b = y_train.shape[0]

    output = net(x_train)
    loss = criterion(output, y_train)
    acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), y_train.cpu().data.numpy()))

    loss_avg += loss.item()*n_b
    acc_avg += acc
    num_exp += n_b

    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def test_epoch(args, mode, dataloader, net, optimizer, criterion, device, tri=False):
    acc_avg, num_exp = 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    input_size = (args.param["image_size"][0], args.param["image_size"][1], args.nc)
    trigger_loc = (args.param["image_size"][0]-3, args.param["image_size"][0]-1)
    init_trigger = np.zeros(input_size)
    init_backdoor = np.random.randint(1, 256,(2, 2, args.nc))
    init_trigger[trigger_loc[0]:trigger_loc[1], trigger_loc[0]:trigger_loc[1], :] = init_backdoor

    mask = torch.FloatTensor(np.float32(init_trigger > 0).transpose((2, 0, 1))).to(device)
    trigger = torch.load("pth/" + args.mn + "_trigger.pth")

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        lab = datum[1].long().to(device)

        n_b = lab.shape[0]

        if tri:
            img[:] = img[:] * (1 - mask) + trigger[0] * mask
            lab[:] = 0

        output = net(img)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        acc_avg += acc
        num_exp += n_b

    acc_avg /= num_exp

    return acc_avg

def test_denoise(args, device, clean_set):
    mean, std = dataset_normalization(args.d)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32)), transforms.Normalize(mean=mean, std=std)])
    if args.d == "cifar10":
        dataset = CIFAR10_noise(root='../ddbd/dataset-distillation/data', train=True, download=True, transform=transform)
    elif args.d == "stl10":
        dataset = STL10_noise(root='../ddbd/dataset-distillation/data', split="train", download=True, transform=transform)
    elif args.d == "svhn":
        dataset = SVHN_noise(root='../ddbd/dataset-distillation/data', split="train", download=True, transform=transform)
    else:
        dataset = FashionMNIST_noise(root='../ddbd/dataset-distillation/data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=3)
    model=denoising_model(nc = args.param['nc']).to(device)
    criterion=nn.MSELoss()
    optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-5)
    
    for epoch in range(20):
        print("epoch:{}".format(str(epoch)))
        for dirty,clean,label in (trainloader):
            dirty=dirty.view(dirty.size(0),-1).type(torch.FloatTensor)
            clean=clean.view(clean.size(0),-1).type(torch.FloatTensor)
            dirty,clean=dirty.to(device),clean.to(device)
            
            #-----------------Forward Pass----------------------
            output=model(dirty)
            loss=criterion(output,clean)
            #-----------------Backward Pass---------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
    x_train = torch.load('pth/' + args.mn + '_img.pth').to(device)
    temp = x_train[:10]
    shape = temp.shape
    target_model = get_model(args.m, args.nc).to(device)
    x_train[:10] = model(temp.view(shape[0],-1)).reshape(shape)
    x_train = x_train.detach().requires_grad_()
    y_train = torch.arange(args.param['num_classes'], dtype=torch.long, device=device).repeat(10, 1)
    y_train = y_train.t().reshape(-1)
    # trainloader = torch.utils.data.DataLoader(x_train, y_train, batch_size=100, shuffle=True, num_workers=0)
    lr = 0.01
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    target_criterion = nn.CrossEntropyLoss().to(device)
    for i in range(300):
        _, acc_train = train_epoch('train', x_train, y_train, target_model, target_optimizer, target_criterion, device)
        if i ==151:
            lr *= 0.1
            optimizer = torch.optim.SGD(target_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    clean_test_loader = DataLoader(clean_set, batch_size=256)
    acc_test = test_epoch(args, 'test', clean_test_loader, target_model, target_optimizer, target_criterion, device)
    acc_test_trigger = test_epoch(args, 'test', clean_test_loader, target_model, target_optimizer, target_criterion, device, tri=True)
    print('ASR: %f' % acc_test_trigger)
    print('CTA: %f' % acc_test)



if __name__ == "__main__":
    np.random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=False, default='fmnist')
    parser.add_argument('-m', type=str, required=False, default='alexnet')
    parser.add_argument('-a', type=str, required=False, default='dc')
    args = parser.parse_args()

    # for ddd in ['cifar10', 'fmnist', 'stl10', 'svhn']:
    #     for mmm in ['alexnet', 'convnet']:
    #         args.d = ddd
    #         args.m = mmm

    print(args)

    args.mn = ""

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
        args.nc = 3
        
        
    elif args.d == "stl10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        clean_set = torchvision.datasets.STL10(root='./data', split='test',
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

        args.mn += "_stl"
        args.nc = 3

    elif args.d == "fmnist":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize((0.2861,), (0.3530,))])
        clean_set = torchvision.datasets.FashionMNIST(root='../ddbd/dataset-distillation/data', train=False,
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

        args.mn += "_fmnist"
        args.nc = 1
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

        args.nc = 3

    args.param = {
        "dataset": args.d,
        "Epochs": 100,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (32, 32),
        "name": args.m,
        "nc": args.nc
    }

    # test_backdoor(device, args, clean_set)

    # reverse_engineer(device, args)
    # Spectral_Signatures(device, args)
    # strip(device, args, clean_set)
    # strip_for_test(device, args, clean_set)
    # test_SCAn(args, clean_set, device)
    # test_denoise(args, device, clean_set)


