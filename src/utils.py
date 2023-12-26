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