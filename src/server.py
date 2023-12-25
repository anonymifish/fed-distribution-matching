import torch
import random
from torch.utils.data import DataLoader, TensorDataset
class Server():
    def __init__(
        self,
        global_model,
        clients,
        communication_rounds,
        join_ratio,
        batch_size,
        model_epochs,
        eval_gap,
        testset,
        device,
    ):
        super(self, Server).__init__()
        self.global_model = global_model
        self.clients = clients
        self.communication_rounds = communication_rounds
        self.join_ratio = join_ratio
        self.batch_size = batch_size
        self.model_epochs = model_epochs
        self.eval_gap = eval_gap
        self.testset = testset
        self.device = device
    def fit(self):
        for round in range(self.communication_rounds):
            # evaluate the model and make checkpoint
            if round % self.eval_gap == 0:
                self.evaluate()

            # select client and train clients, number of synthetic images = # clients(10) * classes(10) * ipc(10)
            synthetic_data = []
            synthetic_label = []
            selected_clients = self.select_clients()
            for client in selected_clients:
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
            test_dataloader = DataLoader(self.testset, 256, shuffle=True, num_workers=4)
            for x, target in test_dataloader:
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
