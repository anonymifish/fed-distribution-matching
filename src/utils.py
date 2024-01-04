import copy
import random

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sample_random_model(model, rho):
    new_model = copy.deepcopy(model)
    parameters = new_model.parameters()

    mean = parameters.view(-1)
    multivariate_normal = MultivariateNormal(mean, torch.eye(mean.shape[0]))
    distance = rho + 1
    while distance > rho:
        sample = multivariate_normal.sample()
        distance = torch.sqrt(torch.sum((mean - sample)**2))

    new_parameters = sample.view(parameters.shape)
    for old_param, new_param in zip(parameters, new_parameters):
        with torch.no_grad():
            old_param.fill_(new_param)

    return new_model

def random_pertube(model, rho):
    new_model = copy.deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]), p=2)

    with torch.no_grad():
        scale = rho / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    return new_model