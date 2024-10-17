import sys
import os 
sys.path.append(os.path.abspath('../'))
import torch
from time import time
from tqdm import tqdm
import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt 
from IPython.display import clear_output
import torch.nn as nn
import torch.nn.functional as F
import functorch
import faulthandler



from convolutions import *
from utils_fns import *
from utils_general import update_sigma_linear, run_scheduler_step, plt_errors, show_with_error
from optimizations import *
from utils_optim import run_optimization, run_grad_optimization, run_cg_optimization, run_bfgs_optimization
from utils_general import run_scheduler_step
from utils_mitsuba import get_mts_rendering, render_smooth, get_mts_rendering_mts
from read_scenes import create_scene_from_xml

if torch.cuda.is_available():
    device = 'cuda'
    print("is available")
    mi.set_variant('cuda_ad_rgb')
    
class CNN(nn.Module):
    def __init__(self, n_params):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.fc2 = nn.Linear(16*64, n_params)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 16*8*8)  # Flatten the tensor
        x = F.sigmoid(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc2(x)
        return x
    
n_params = 1  # Number of output parameters
model = CNN(n_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {pytorch_total_params}')

def jvp_model(model, x, v):
    '''
    JVP of model output wrt weights
    '''    
    def fmodel_wrt_param_dep(flattened_tensor):
        tensors = []
        index = 0
        for param in model.parameters():
            num_elements = param.numel()
            tensor = flattened_tensor[index:index + num_elements].view(param.size())
            tensors.append(tensor)
            index += num_elements
        
        return fmodel(tuple(tensors), x)
    
    def fmodel_wrt_param(flattened_tensor):
        unflattened_params = {}
        index = 0
        for name, param in dict(model.named_parameters()).items():
            num_elements = param.numel()
            unflattened_params[name] = flattened_tensor[index:index + num_elements].view(param.shape)
            index += num_elements
        
        return torch.func.functional_call(model, dict(unflattened_params), x)
    
    fmodel,_ = functorch.make_functional(model)
    flatten_weights = torch.cat([param.view(-1) for param in model.parameters()])
    print(torch.func.jvp(fmodel_wrt_param, (flatten_weights,), (v,))[1])
    return torch.func.jvp(fmodel_wrt_param_dep, (flatten_weights,), (v,))[1]

n_params = 1  # Number of output parameters
model = CNN(n_params)
model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {pytorch_total_params}')
print(jvp_model(model, torch.rand((3, 256, 256)).to(device), torch.rand((pytorch_total_params,)).to(device)))