import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from itertools import product
import math
import copy
import time
import logging
import pickle
import random

from datasets import MNIST_truncated, EMNIST_truncated, CIFAR10_truncated, CIFAR10_Poisoned, CIFAR10NormalCase_truncated, EMNIST_NormalCase_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_emnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    emnist_train_ds = EMNIST_truncated(datadir, train=True, download=True, transform=transform)
    emnist_test_ds = EMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    net_dataidx_map = None
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset.lower() == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]

    elif dataset == "shakespeare":
        net_dataidx_map = {}
        with open(datadir[0]) as json_file:
            train_data = json.load(json_file)

        with open(datadir[1]) as json_file:
            test_data = json.load(json_file)

        for j in range(n_nets):
            client_user_name = train_data["users"][j]

            client_train_data = train_data["user_data"][client_user_name]['x']
            num_samples_train = len(client_train_data)
            net_dataidx_map[j] = [i for i in range(num_samples_train)] # TODO: this is a dirty hack. needs modification
        return net_dataidx_map
    
    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        # XXX: logic of this part is a bit convoluted
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while (min_size < 10) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        if dataset == 'cifar10':
            if args.poison_type == 'howto' or args.poison_type == 'greencar-neo':
                green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
                #sanity_check_counter = 0
                for k, v in net_dataidx_map.items():
                    remaining_indices = [i for i in v if i not in green_car_indices]
                    #sanity_check_counter += len(remaining_indices)
                    net_dataidx_map[k] = remaining_indices

    return net_dataidx_map


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl



def get_dataloader_normal_case(dataset, datadir, train_bs, test_bs, 
                                dataidxs=None, 
                                user_id=0, 
                                num_total_users=200,
                                poison_type="southwest",
                                ardis_dataset=None,
                                attack_case='normal-case'):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        if dataset == 'emnist':
            dl_obj = EMNIST_NormalCase_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
        elif dataset == 'cifar10':
            dl_obj = CIFAR10NormalCase_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

        # this only supports cifar10 right now, please be super careful when calling it using other datasets
        # def __init__(self, root, 
        #                 dataidxs=None, 
        #                 train=True, 
        #                 transform=None, 
        #                 target_transform=None, 
        #                 download=False,
        #                 user_id=0,
        #                 num_total_users=200,
        #                 poison_type="southwest"):        
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True,
                                    user_id=user_id, num_total_users=num_total_users, poison_type=poison_type,
                                    ardis_dataset_train=ardis_dataset, attack_case=attack_case)
        
        test_ds = None #dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl



def load_poisoned_dataset(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.dataset in ("mnist", "emnist"):
        if args.fraction < 1:
            fraction=args.fraction  #0.1 #10
        else:
            fraction=int(args.fraction)

        with open("poisoned_dataset_fraction_{}".format(fraction), "rb") as saved_data_file:
            poisoned_dataset = torch.load(saved_data_file)
        num_dps_poisoned_dataset = poisoned_dataset.data.shape[0]
        
        # prepare fashionMNIST dataset
        fashion_mnist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        # prepare EMNIST dataset
        emnist_train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_dataset,
             batch_size=args.batch_size, shuffle=True, **kwargs)
        vanilla_test_loader = torch.utils.data.DataLoader(emnist_test_dataset,
             batch_size=args.test_batch_size, shuffle=False, **kwargs)
        targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
             batch_size=args.test_batch_size, shuffle=False, **kwargs)
        clean_train_loader = torch.utils.data.DataLoader(emnist_train_dataset,
                batch_size=args.batch_size, shuffle=True, **kwargs)

        if args.poison_type == 'ardis':
            # load ardis test set
            with open("./data/ARDIS/ardis_test_dataset.pt", "rb") as saved_data_file:
                ardis_test_dataset = torch.load(saved_data_file)

            targetted_task_test_loader = torch.utils.data.DataLoader(ardis_test_dataset,
                 batch_size=args.test_batch_size, shuffle=False, **kwargs)

    
    elif args.dataset == "cifar10":
        if args.poison_type == "southwest":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)

            if args.attack_case == "edge-case":
                with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            elif args.attack_case == "normal-case" or args.attack_case == "almost-edge-case":
                with open('./saved_datasets/southwest_images_adv_p_percent_edge_case.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./saved_datasets/southwest_images_p_percent_edge_case_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            else:
                raise NotImplementedError("Not Matched Attack Case ...")             

            #
            logger.info("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
            #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
            sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            
            logger.info("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
            #sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
            sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck



            # downsample the poisoned dataset #################
            if args.attack_case == "edge-case":
                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
                saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
                sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
                logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
            elif args.attack_case == "normal-case" or args.attack_case == "almost-edge-case":
                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(784,
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
            ######################################################


            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400 # M
            samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            # keep a copy of clean data
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ########################################################


            poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
            poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            logger.info("{}".format(poisoned_trainset.data.shape))
            logger.info("{}".format(poisoned_trainset.targets.shape))
            logger.info("{}".format(sum(poisoned_trainset.targets)))

            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader


def seed_experiment(seed=0):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything")
