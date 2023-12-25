# This is an implementation of FedDM:Iterative Distribution Matching
# for Communication-Efficient Federated Learning (except Differential Privacy)

import os
import json

from torch.utils.data import Subset

from config import parser

from dataset.data.dataset import get_dataset
from models import ResNet18, ConvNet
from client import Client
from server import Server

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # set seeds and parse args, init wandb

    args = parser.parse_args()

    # get dataset and init models
    dataset_info, trainset, testset, testloader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    with open(args.split_file) as file:
        client_indices, client_classes = json.load(file)
    train_sets = [Subset(trainset, indices) for indices in client_indices]

    [PerLabelDatasetNonIID(sub_train, classes, channel, args.device)

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

    )]
    server =

    # fit the model
    server.fit()

if __name__ == "__main__":
    main()