import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from client import Client


class Server:
    def __init__(
        self,
        global_model: nn.Module,
        clients: list[Client],
        # --- model training params ---
        communication_rounds: int,
        join_ratio: float,
        batch_size: int,
        model_epochs: int,
        # --- test and evaluation information ---
        eval_gap: int,
        test_set: object,
        test_loader: DataLoader,
        device: torch.device,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients

        self.communication_rounds = communication_rounds
        self.join_ratio = join_ratio
        self.batch_size = batch_size
        self.model_epochs = model_epochs

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader
        self.device = device

    def fit(self):
        for rounds in range(self.communication_rounds):
            # evaluate the model and make checkpoint
            print(f' ====== round {rounds} ======')
            if rounds % self.eval_gap == 0:
                acc = self.evaluate()
                print(f'round {rounds} evaluation: test acc is {acc}')

            start_time = time.time()

            # select client and train clients, number of synthetic images = # clients(10) * classes(10) * ipc(10)
            synthetic_data = []
            synthetic_label = []
            selected_clients = self.select_clients()
            for client in selected_clients:
                client.recieve_model(self.global_model)
                imgs, labels = client.train()
                synthetic_data.append(imgs)
                synthetic_label.append(labels)

            # update model parameters by SGD
            synthetic_data = torch.cat(synthetic_data, dim=0)
            synthetic_label = torch.cat(synthetic_label, dim=0)
            synthetic_dataset = TensorDataset(synthetic_data, synthetic_label)
            synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=4)
            self.global_model.train()
            model_optimizer = torch.optim.SGD()
            loss_function = torch.nn.CrossEntropyLoss()
            for epoch in range(self.model_epochs):
                for x, target in synthetic_dataloader:
                    pred = self.global_model(x)
                    loss = loss_function(pred, target)
                    model_optimizer.zero_grad()
                    loss.backward()
                    model_optimizer.step()

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def evaluate(self):
        self.global_model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.global_model(x)
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        return correct / float(total)

    def make_checkpoint(self, round):
        checkpoint = {
            'current_round': round,
            'model': self.global_model.state_dict()
        }
        return checkpoint
