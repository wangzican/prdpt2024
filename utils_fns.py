import torch
import functools
import numpy as np


def grad_of_gaussiankernel(x, sigma, *args):
    grad_of_gauss = -(x / sigma ** 2) * calc_gauss(x, mu=0.0, sigma=sigma)
    return grad_of_gauss

def gauss(x, sigma, *args):
    return calc_gauss(x, mu=0.0, sigma=sigma)

def hess_of_gaussiankernel(x, sigma, diag_approx=False):
    '''
    Get the hessian of gaussian, diag_approx is to approximate with only diagonal
    x is nxm, n different samples each with m parameters
    return n hessians of mxm
    '''
    if diag_approx:
        m = x.shape[1]
        hess = (x**2/sigma**2 - 1) / sigma**2 * calc_gauss(x, mu=0.0, sigma=sigma)
        hess = hess.unsqueeze(2) * torch.eye(m).unsqueeze(0).to(x.device)
        return hess
    raise NotImplementedError("Full hessian not implemented yet")
    pos = torch.einsum('ij,ik->ijk', x, x)
    hess = (pos/sigma**2 - 1) / sigma**2 * calc_gauss(x, mu=0.0, sigma=sigma)
    return hess

def calc_gauss(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi)**0.5) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def calc_gauss_2d(x, y, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi)**0.5) * torch.exp(-0.5 * ((x - mu)**2 + (y - mu)**2) / sigma**2) 


def mc_estimate(f_xi, p_xi):
    N = f_xi.shape[0]
    estimate = 1. / N * (f_xi / p_xi).sum(dim=0)  # average along batch axis, leave dimension axis unchanged
    return estimate

def mc_estimate_hess(f_xi, p_xi):
    N = f_xi.shape[0]
    p_xi = torch.einsum('ij,ik->ijk', p_xi, p_xi)
    estimate = 1. / N * (f_xi / p_xi).sum(dim=0)  # average along batch axis, leave dimension axis unchanged
    return estimate

def convolve_mi(kernel_fn, render_fn, importance_fn, mc_fn, theta, nsamples, context_args, **kargs):
    # sample, get kernel(samples), get render(samples), return mc estimate of output
    # expect theta to be of shape [1, n], where n is dimensionality
    dim = theta.shape[-1]
    sigma = context_args['sigma']
    update_fn = context_args['update_fn']  # fn pointer to e.g. apply_rotation

    if context_args['sampler'] == 'uniform':
        tau = uniform(nsamples, context_args['antithetic'], dim, context_args['device'])

    if context_args['sampler'] == 'importance':
        # get importance-sampled taus
        tau, pdf = importance_fn(nsamples, sigma, context_args['antithetic'], dim, context_args['device'])
    
    # print(tau.shape)
    # get kernel weight at taus
    diag_approx = kargs.get('diag_approx', False)
    weights = kernel_fn(tau, sigma, diag_approx)
    
    # twice as many samples when antithetic
    if context_args['antithetic']:
        nsamples *= 2

    # shift samples around current parameter
    theta_p = torch.cat([theta] * nsamples, dim=0) - tau
    # print(theta_p)
    renderings, avg_img = render_fn(theta_p, update_fn, context_args)    # output shape [N]
    
    # get the dimension needed to expand x so that it broadcast to the weight
    # this is added for hessian calculation
    num_new_dims = len(weights.shape) - 1
    renderings = renderings.view((nsamples,) + (1,) * num_new_dims)

    # weight output by kernel, mc-estimate gradient
    output = renderings * weights
    if context_args['sampler'] == 'uniform':
        forward_output = output.mean(dim=0)
    if context_args['sampler'] == 'importance':
        forward_output = mc_fn(output, pdf)
    
    return forward_output, avg_img


def uniform(n_samples, is_antithetic, dim, device, low=-1.0, high=1.0):
    eps = 0.00001
    samples = torch.rand(n_samples, dim).to(device)
    
    if is_antithetic:
        samples = torch.cat([samples, low + high - samples])

    
    # avoid NaNs bc of numerical instabilities in log
    samples[torch.isclose(samples, torch.ones_like(samples))] -= eps
    samples[torch.isclose(samples, torch.zeros_like(samples))] += eps
    
    return samples

def importance_gradgauss_1d(n_samples, sigma, is_antithetic, dim, device):
    eps = 0.00001
    randoms = torch.rand(n_samples, dim).to(device)

    def icdf(x, sigma):
        res = torch.zeros_like(x).to(device)
        res[mask == 1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * (1.0 - x[mask == 1])))
        res[mask == -1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * x[mask == -1]))
        return res

    # samples and AT samples
    if is_antithetic:
        randoms = torch.cat([randoms, 1.0 - randoms])

    # avoid NaNs bc of numerical instabilities in log
    randoms[torch.isclose(randoms, torch.ones_like(randoms))] -= eps
    randoms[torch.isclose(randoms, torch.zeros_like(randoms))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=0.5))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=-0.5))] -= eps

    mask = torch.where(randoms < 0.5, -1.0, 1.0)
    x_i = icdf(randoms, sigma=sigma) * mask

    f_xi = torch.abs(x_i) * (1.0 / sigma ** 2) * calc_gauss(x_i, mu=0.0, sigma=sigma)
    f_xi[f_xi == 0] += eps
    p_xi = 0.5 * sigma * (2.0 * np.pi)**0.5 * f_xi

    return x_i, p_xi


def smoothFn(func=None, context_args=None, device='cuda'):
    if func is None:
        return functools.partial(smoothFn, context_args=context_args, device=device)

    @functools.wraps(func)
    def wrapper(input_tensor, context_args, *args):
        class SmoothedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_tensor, context_args, *args):

                original_input_shape = input_tensor.shape
                importance_fn = importance_gradgauss_1d

                forward_output, avg_img = convolve_mi(grad_of_gaussiankernel, func, importance_fn, mc_estimate, 
                                                   input_tensor, context_args['nsamples'], context_args, args=args)

                # save for bw pass
                ctx.fw_out = forward_output
                ctx.original_input_shape = original_input_shape

                return forward_output.mean(), avg_img

            @staticmethod
            def backward(ctx, dy, dz):
                # dz is grad for avg_img
                # Pull saved tensors
                original_input_shape = ctx.original_input_shape
                fw_out = ctx.fw_out
                grad_in_chain = dy * fw_out

                return grad_in_chain.reshape(original_input_shape), None

        return SmoothedFunc.apply(input_tensor, context_args, *args)

    return wrapper



def smoothFn_forward(func=None, context_args=None, device='cuda'):
    if func is None:
        return functools.partial(smoothFn, context_args=context_args, device=device)

    @functools.wraps(func)
    def wrapper(input_tensor, context_args, *args):
        class SmoothedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_tensor, context_args, *args):

                original_input_shape = input_tensor.shape
                importance_fn = importance_gradgauss_1d

                forward_output, avg_img = convolve_mi(gauss, func, importance_fn, mc_estimate, 
                                                   input_tensor, context_args['nsamples'], context_args, args=args)

                # save for bw pass
                ctx.fw_out = forward_output
                ctx.original_input_shape = original_input_shape

                return forward_output.mean(), avg_img

            @staticmethod
            def backward(ctx, dy, dz):
                # dz is grad for avg_img
                # Pull saved tensors
                original_input_shape = ctx.original_input_shape
                fw_out = ctx.fw_out
                grad_in_chain = dy * fw_out

                return grad_in_chain.reshape(original_input_shape), None

        return SmoothedFunc.apply(input_tensor, context_args, *args)

    return wrapper

def smoothFn_hess(func, input, context_args, device='cuda', **kargs):
    '''
    Get the hessian of the convolved function
    '''
    importance_fn = importance_gradgauss_1d
    hess, avg_img = convolve_mi(hess_of_gaussiankernel, func, importance_fn, mc_estimate_hess,
                                       input, context_args['nsamples'], context_args, diag_approx=kargs['diag_approx'])
    return hess

