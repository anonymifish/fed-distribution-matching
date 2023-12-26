# This is an implementation of FedDM:Iterative Distribution Matching
# for Communication-Efficient Federated Learning (without Differential Privacy)
import copy
import json
import os

import torch
import wandb
from torch.utils.data import Subset

from client import Client
from server import Server
from config import parser
from dataset.data.dataset import get_dataset, PerLabelDatasetNonIID
from models import ResNet18, ConvNet
from utils import setup_seed

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    args = parser.parse_args()

    # set seeds and parse args, init wandb
    mode = "disabled" if args.debug else "online"
    wandb.init(
        project=f'FedDM_{args.dataset}_client{args.client_num}_{args.alpha}',
        name=f'{args.model}_cr{args.communication_rounds}_join_ratio{args.join_ratio}',
        mode=mode,
    )
    wandb.config.update(args)
    setup_seed(args.seed)
    device = torch.device(args.device)

    # get dataset and init models
    dataset_info, train_set, test_set, test_loader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    with open(args.split_file) as file:
        client_indices, client_classes = json.load(file)
    train_sets = [Subset(train_set, indices) for indices in client_indices]

    if args.model == "ConvNet":
        global_model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        )
    elif args.model == "ResNet":
        global_model = ResNet18(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes']
        )
    else:
        raise NotImplemented("only support ConvNet and ResNet")

    # init server and clients
    client_list = [Client(
        cid=i,
        train_set=PerLabelDatasetNonIID(
            train_sets[i],
            client_classes[i],
            dataset_info['channel'],
            device,
        ),
        classes=client_classes[i],
        dataset_info=dataset_info,
        ipc=args.ipc,
        rho=args.rho,
        dc_iterations=args.dc_iterations,
        real_batch_size=args.dc_batch_size,
        image_lr=args.image_lr,
        device=device,
    ) for i in range(args.client_num)]

    server = Server(
        global_model=global_model,
        clients=client_list,
        communication_rounds=args.communication_rounds,
        join_ratio=args.join_ratio,
        batch_size=args.batch_size,
        model_epochs=args.model_epochs,
        eval_gap=args.eval_gap,
        test_set=test_set,
        test_loader=test_loader,
        device=device,
    )
    print('Server and Clients have been created.')

    # fit the model
    server.fit()

if __name__ == "__main__":
    main()