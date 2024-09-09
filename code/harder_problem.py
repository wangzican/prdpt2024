import sys
import os 
sys.path.append(os.path.abspath('../'))
from optimizations import *
from convolutions import *
from utils_general import update_sigma_linear, run_scheduler_step, plt_errors, show_with_error
import torch
import numpy as np
import matplotlib.pyplot as plt 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scale = 500

# From Michael's code box example
def get_rendering(theta,  update_fn=None, ctx_args=None): 
  return draw_rect_3param(theta[0], theta[1], theta[2])

def draw_rect_3param(px, py, size):   
  s, exp = scale, 10
  w, h = min(abs(size), s/2), min(abs(size), s/2)
  px = max(0+h, min(s-h, px))
  py = max(0+w, min(s-w, py))
  ss = torch.arange(s, device=device)
  x, y = torch.meshgrid(ss, ss)
  image = 1 - 2*(abs(((py - x)/w))**exp + abs((y - px)/h)**exp)
  return torch.flipud(image.clamp(0, 1)).unsqueeze(-1)

def render_smooth(perturbed_theta, gt_img):
  '''
  Michael's code but without the average image
  '''
  with torch.no_grad():
      imgs, losses = [], []
      for j in range(perturbed_theta.shape[0]):       # for each sample
          perturbed_img = get_rendering(perturbed_theta[j, :])
          perturbed_loss = torch.nn.MSELoss()(perturbed_img, gt_img)
          imgs.append(perturbed_img)
          losses.append(perturbed_loss)
      loss = torch.stack(losses)
  return loss

def logging_box(theta, img_errors, param_errors, i, interval=5, **f_args):
    # plotting, logging, printing...
    theta = theta.squeeze()
    img_curr = get_rendering(theta)
    img_loss = torch.nn.MSELoss()(img_curr, ref_img).item()
    param_loss = torch.nn.MSELoss()(theta, gt_theta).item()
    img_errors.append(img_loss)
    param_errors.append(param_loss)

    print(f"Iter {i+1} - Img.Loss: {img_loss:.4f} - Param.Loss: {param_loss:.4f}")
    if (i+1) % interval == 0: 
        show_with_error(img_curr, ref_img, iter=i+1)
        plt_errors(img_errors, param_errors, title=f'Iter {i+1}')
    return img_errors, param_errors

torch.manual_seed(0)

# set up initial and gt translation:
theta = torch.tensor([0.5, 0.6, 0.3], device=device)*scale
gt_theta = torch.tensor([0.3, 0.33, 0.15], device=device)*scale

init_img = get_rendering(theta)
ref_img = get_rendering(gt_theta)


n_samples = 6
sigma=scale/10

ctx_args = {'nsamples':n_samples}
BFGS_box_hparams = {'sigma_annealing': True,
                    'sigma': sigma,
                    'epochs': 1000,
                    'anneal_const_first': 0,
                    'anneal_const_last': 0,
                    'anneal_sigma_min': 0.05,
                    'TR':True,
                    'TR_bound': 5,
                    'TR_rate': 0.2,
                    'learning_rate':0.2,
                    'line_search_fn': None, #'strong_wolfe',
                    'history_size': 1000,
                    'tol': 5e-6, # tolerance for newton
                    'plot_interval':1000 # number of iterations to plot
                    }

# --------------- run optimization
max_iter = BFGS_box_hparams['epochs']
f_args = {'gt_img': ref_img}
kernel_args = {'sigma': sigma}
sampler_args = {'sigma': sigma, 'is_antithetic': True}

# show_with_error(init_img, ref_img, iter=0)
BFGS_opt(render_smooth, theta, max_iter, log_func=logging_box, f_args=f_args, kernel_args=kernel_args, 
           sampler_args=sampler_args, opt_args=BFGS_box_hparams, ctx_args=ctx_args, device=device)
