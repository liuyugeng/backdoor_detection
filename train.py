import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data import get_data
from model import *
import os
import json
import torch.optim as optim

from art.defences.transformer import poisoning
from art.defences.detector import poison
from art.defences.transformer import poisoning
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, preprocess
from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier, PyTorchClassifier

import math
import random
import numpy as np
import time
import scipy
import cv2
import scipy.stats
import torch.nn.functional as F

import argparse

import torchvision
import torchvision.transforms as transforms



def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()

##### Neural Cleanse #####

def reverse_engineer(device):
    param = {
        "dataset": "cifar10",
        "Epochs": 100,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (32, 32),
        "name": "alexnet"
    }
    model = get_model(param["name"])
    model.load_state_dict(torch.load('a_dc_cifar.pth'))
    model = model.to(device)
    _, _, x_test, y_test = get_data(param)
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    mask_flatten = []
    norm_list = []
    idx_mapping = {}
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
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


##### Spectral_Signatures #####

def Spectral_Signatures(device):
    param = {
        "dataset": "svhn",
        "Epochs": 100,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (32, 32),
        "name": "convnet"
    }
    x_train = torch.load('vis_DC_SVHN_ConvNet_10ipc_exp0_iter1000.pth')

    model = get_model(param["name"])
    model.load_state_dict(torch.load('c_dc_svhn.pth'))
    model = model.to(device)

    y_train = torch.tensor([np.ones(10)*i for i in range(10)], dtype=torch.long, requires_grad=False).view(-1)

    images = x_train.to(device)
    labels = y_train.to(device)

    o_train = model(images)
    (x_raw, y_raw), _, min_, max_ = load_cifar10(raw=True)

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
    report, is_clean_lst = defence.detect_poison()
    cl = np.array(is_clean_lst)
    mmsk = cl == 0
    print(report)
    print(len(cl[mmsk]))

    is_clean = is_poison_train == 0

    # defence.plot_clusters(folder="/home/c02yuli/project/ddbd/dataset-distillation/")

    confusion_matrix = defence.evaluate_defence(is_clean)
    print("Evaluation defence results for size-based metric: ")
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        print(jsonObject[label])

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

    print(poisn_mean)
    print(clean_mean)

##### strip #####

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
    x_train = torch.load(args.mn + "_img.pth")

    model = get_model(args.m)
    model.load_state_dict(torch.load(args.mn + ".pth"))
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
    print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    print(threshold)

    FAR = sum(i > threshold for i in entropy_trojan)

    (mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
    print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    print(threshold)
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

    min_benign_entropy = np.mean(entropy_benigh)
    max_trojan_entropy = np.mean(entropy_trojan)

    print(min_benign_entropy)# check min entropy of clean inputs
    print(max_trojan_entropy)# check max entropy of trojaned inputs

def get_poison(x_poison, trigger):
    trigger_loc = (29, 31)
    x_poison[:, 29:31, 29:31] = trigger[:, 29:31, 29:31]
    return x_poison

def strip_for_test(device, args, clean_set):
    model = get_model(args.m)
    model.load_state_dict(torch.load(args.mn + ".pth"))
    model = model.to(device)

    trigger = torch.load(args.mn + "_trigger.pth")[0].detach().cpu().numpy()

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
            entropy_benigh[j] = entropyCal(x_background, n_sample, x_train[n_test*2:n_test*2+n_sample], nn_model, state.device)

    for j in range(n_test):
        x_poison = x_train[j+n_test]
        x_poison = get_poison(x_poison, trigger)
        entropy_trojan[j] = entropyCal(x_poison, n_sample, x_train[n_test*2:n_test*2+n_sample], nn_model, state.device)

    entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
    entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs

    (mu, sigma) = scipy.stats.norm.fit(entropy_trojan)
    print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    print(threshold)

    FAR = sum(i > threshold for i in entropy_trojan)

    (mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
    print(mu, sigma)

    threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
    print(threshold)
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

    min_benign_entropy = np.mean(entropy_benigh)
    max_trojan_entropy = np.mean(entropy_trojan)

    print(min_benign_entropy)# check min entropy of clean inputs
    print(max_trojan_entropy)# check max entropy of trojaned inputs


if __name__ == "__main__":
    np.random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=False, default='cifar10')
    parser.add_argument('-m', type=str, required=False, default='alexnet')
    parser.add_argument('-a', type=str, required=False, default='dc')
    args = parser.parse_args()

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
    
    # reverse_engineer(device)
    # Spectral_Signatures(device)
    # strip(device, args, clean_set)
    strip_for_test(device, args, clean_set)
