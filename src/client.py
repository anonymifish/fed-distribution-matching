import copy

import torch

from data.dataset import PerLabelDatasetNonIID
from utils import sample_random_model


class Client:
    def __init__(
        self,
        cid: int,
        # --- dataset information ---
        train_set: PerLabelDatasetNonIID,
        classes: list[int],
        dataset_info: dict,
        # --- data condensation params ---
        ipc: int,
        rho: float,
        dc_iterations: int,
        real_batch_size: int,
        image_lr: float,
        device: torch.device,
    ):
        self.cid = cid,

        self.train_set = train_set,
        self.classes = classes,
        self.dataset_info = dataset_info

        self.ipc = ipc
        self.rho = rho
        self.dc_iterations = dc_iterations
        self.real_batch_size = real_batch_size
        self.image_lr = image_lr

        self.device = device

        self.synthetic_images = torch.randn(
            size=(
                len(classes)*ipc,
                dataset_info['channel'],
                dataset_info['im_size'][0],
                dataset_info['im_size'][1],
            ),
            dtype=torch.float,
            requires_grad=True,
            device=self.device
        )

    def train(self):
        # initialize S_k from real examples and initialize optimizer
        for i, c in enumerate(self.classes):
            self.synthetic_images.data[i * self.ipc : (i + 1) * self.ipc] = self.trainset.get_images(c, self.ipc, avg=False).detach().data
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)
        optimizer_image.zero_grad()

        for dc_iteration in range(self.dc_iterations):
            # sample w ~ P_w(w_r)
            sample_model = sample_random_model(self.global_model, self.rho)

            # sample a mini-batch of real / synthetic image for each class then compute DC loss based on equation 8
            loss = torch.tensor(0.0)
            for i, c in enumerate(self.classes):
                real_image = self.trainset.get_image(c, self.real_batch_size)
                synthetic_image = self.synthetic_images[i*self.ipc : (i+1)*self.ipc].reshape(
                    (self.ipc, self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1]))

                real_feature = sample_model.embed(real_image).detach()
                synthetic_feature = sample_model.embed(synthetic_image)
                real_logits = sample_model(real_image).detach()
                synthetic_logits = sample_model(synthetic_image)

                loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0))**2)
                loss += torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0))**2)

            # update S_k
            optimizer_image.zero_grad()
            loss.backward()
            optimizer_image.step()

        # return S_k
        synthetic_labels = torch.cat([torch.ones(self.ipc) * c for c in self.classes])
        return copy.deepcopy(self.synthetic_images.detach()), synthetic_labels

    def recieve_model(self, global_model):
        self.global_model = copy.deepcopy(global_model).cpu()