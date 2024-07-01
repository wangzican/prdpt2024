import os

import fontTools.cffLib
import torch
import numpy as np
import matplotlib.pyplot as plt
from local_utils import normalize, get_initial_and_gt, get_hparams, get_defaults
from train_utils_modular import train_proxy_structured, set_same_samples

import scipy
x = np.array([-100, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 100])
y = np.array([9, 7.5, 7.0, 6.5, 6, 4, 3.5, 0, 2, 1, 6, 8, 9])
spline = scipy.interpolate.Akima1DInterpolator(x, y)

# remove old frames
framedir = 'results/frames'
[os.remove(os.path.join(framedir, x)) for x in os.listdir(framedir)]


def update_values(valdict, value, mode):
    valdict['curr_x'] = value
    return valdict


def render_fn(values):
    curr_val = values['curr_x'].detach().cpu().numpy()
    curr_f = spline(curr_val)[0]        # get item from np array
    return torch.full([1, 3, 2, 2], fill_value=curr_f)


def callback_fn(kwargs, curr_x, gt_x, sample_results):

    plot_n = 10
    if kwargs['current_iter'] % plot_n != 0:
        return
    # if kwargs['current_iter'] > 500: raise KeyError

    x_test = torch.linspace(0.0, 8.0, steps=200).reshape(200, 1)
    pred_loss = kwargs['proxy'].query_lossproxy(x_test).detach().cpu()

    # to calculate the real loss landscape, we must evaluate spline and
    # then subtract from the GT value

    xval = curr_x.detach().cpu().numpy()
    yval = spline(xval)

    gt_xval = gt_x.detach().cpu().numpy()
    gt_yval = spline(gt_xval)

    if kwargs['criterion'] == torch.nn.functional.mse_loss:
        real_loss = spline(x_test.numpy()) ** 2
        plt.ylim([-25.5, 50.0])
        func_surface = spline(x_test.numpy())
        plt.plot(x_test, func_surface, c='black', label='f(x)', alpha=0.2)
    elif kwargs['criterion'] == torch.nn.functional.l1_loss:
        real_loss = np.abs(spline(x_test.numpy()))
        # plt.ylim([-0.5, 10.0])
    else:
        raise NotImplementedError('unknown loss')

    loss = ((curr_x - gt_x)**2).mean()

    n_offs = 1.5
    normal = torch.distributions.Normal(loc=curr_x.item(), scale=kwargs['sigma'])
    pts_around_x = torch.linspace(start=curr_x.item()-n_offs, end=curr_x.item()+n_offs, steps=100)
    pts_at_normal = normal.log_prob(pts_around_x).exp() + yval

    plt.plot(x_test, real_loss, label='loss landscape')
    plt.plot(x_test, pred_loss, label='proxy')

    spread = kwargs['confidence']
    xval_confmin, xval_confmax = xval - spread * kwargs['sigma'], xval + spread * kwargs['sigma']
    yval_confmin, yval_confmax = yval - 4.0, yval + 4.0
    plt.plot([xval_confmin]*10, np.linspace(yval_confmin, yval_confmax, 10), linestyle='dashed', c='lime')
    plt.plot([xval_confmax]*10, np.linspace(yval_confmin, yval_confmax, 10), linestyle='dashed', c='lime', label='2sigma')

    plt.plot(pts_around_x, pts_at_normal, c='grey', alpha=0.75, label='Sampling Gaussian')
    plt.scatter(xval, yval, c='green', marker='o', s=50, label='current')
    plt.scatter(gt_xval, gt_yval, c='red', marker='o', s=30, label='minimum')
    plt.xlabel('Parameter')
    plt.ylabel('Loss Landscape')
    plt.legend(loc='upper right')
    plt.title(f'Iter {kwargs["current_iter"]} - Loss: {loss.item():.4f}')

    plt.show()
    # plt.savefig(f'results/frames/iter{kwargs["current_iter"]}.png')
    # plt.close('all')


if __name__ == '__main__':
    mode = 'toyexample'

    prox = 'NEURAL'
    samp = 'gaussian'

    hparams = get_hparams(mode)
    hparams['experiment_mode'] = mode
    defaults = get_defaults(hparams)
    hparams['proxy'] = prox
    hparams['sampler'] = samp.replace('smooth', '')
    hparams['smooth_loss'] = True if 'smooth' in samp else False

    hparams = set_same_samples(params=hparams, sampler=samp)

    # hparams['batchsize'] = 5
    hparams['epochs'] = 6000
    hparams['sigma'] = 0.33
    hparams['batchsize'] = 10
    hparams['inner_sigma'] = 0.1

    # hparams['lr_alpha'] = 1e-3

    # set shit up
    np.random.seed(42)
    torch.manual_seed(42)
    a = torch.tensor([1.0], device=hparams['device'])  # init cuda context

    # plotting ?
    plot_interval = 100000
    plot_initial = False
    plot_intermediate = False

    # setup initial translation and gt translation
    init_alpha, gt_alpha, ndim = get_initial_and_gt(hparams, seed=0)
    hparams['ndim'] = ndim
    hparams['alpha'] = init_alpha
    hparams['gt_alpha'] = gt_alpha

    idstring = f"toyexample"

    train_proxy_structured(mode=mode,
                           hparams=hparams,
                           defaults=defaults,
                           render_fn=render_fn,
                           update_fn=update_values,
                           normalize_fn=normalize,
                           plot_initial=plot_initial,
                           plot_interval=plot_interval,
                           plot_intermediate_or_final=plot_intermediate,
                           idstring=idstring,
                           callback=callback_fn)
