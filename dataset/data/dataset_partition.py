import argparse
import json
import os

import numpy as np
from torchvision import datasets, transforms


def partition(args):
    # set seeds: TODO
    # raise NotImplemented("set seeds")

    # prepare datasets for then partition latter
    if args.dataset == 'MNIST':
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.MNIST(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = [str(c) for c in range(num_classes)]
    elif args.dataset == 'CIFAR10':
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.CIFAR100(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    else:
        exit(f'unknown dataset: f{args.dataset}')

    min_size = 0
    min_require_size = 10
    K = num_classes
    labels = np.array(dataset.targets, dtype='int64')
    N = labels.shape[0]

    dict_users = {}
    dict_classes = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(args.client_num)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.client_num))
            proportions = np.array([p * (len(idx_j) < N / args.client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(args.client_num):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    net_cls_counts = {}

    for net_i, dataidx in dict_users.items():
        dict_classes[net_i] = []
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        for c, cnt in tmp.items():
            if cnt >= 10:
                dict_classes[net_i].append(c)

    print('Data statistics: %s' % str(net_cls_counts))

    save_path = os.path.join(os.path.dirname(__file__), '../', 'split_file')
    file_name = 'f../{args.dataset}_client_num={args.client_num}_alpha={args.alpha}.json'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, file_name), 'w') as json_file:
        json.dump([dict_users[i].to_list() for i in range(args.client_num)], json_file, indent=4)
        json.dump([dict_classes[i].to_list() for i in range(args.client_num)], json_file, indent=4)

if __name__ == "__main__":
    partition_parser = argparse.ArgumentParser()

    partition_parser.add_argument("--dataset", type=str, default='CIFAR10')
    partition_parser.add_argument("--client_num", type=int, default=10)
    partition_parser.add_argument("--alpha", type=float, default=0.5)
    partition_parser.add_argument("--dataset_root", type=str, default='/home/yfy/datasets/torchvision')

    args = partition_parser.parse_args()
    partition(args)
