import torch
import torch.optim as optim
from torch.autograd import Function
import matplotlib.pyplot as plt
from convolutions import *
import functools
import time
import numpy as np

def aggregate_dist(n, m):
    '''
    For n number of samples, m dimension
    '''
    base_n = n//m
    remainder = n%m
    slots = [base_n]*m
    extra_ind = np.random.choice(m, remainder, replace=False)
    for i in extra_ind:
        slots[i] += 1
    return slots

def smoothFn_gradient(func, sampler, n, f_args, kernel_args, sampler_args, aggregate=False, device='cuda'):
    '''
    Get the gradient of the convolved function,
    n is number of samples
    Returns a function that takes input tensor and returns the gradient
    '''
    def grad(input):
        '''
        input should be tensor of shape (1, m)
        '''
        dims = input.shape[1]
        result = torch.zeros(1, dims, device=device)
        for i in range(dims):
            sampler_args['dir'] = i
            result[:, i] = convolve(func, gauss_grad, input, n=n, sampler=sampler, 
                            f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)
        return result
    def grad_aggregate(input):
        '''
        input should be tensor of shape (1, m)
        '''
        dims = input.shape[1]
        result = torch.zeros(1, dims, device=device)
        aggregate_dims = torch.zeros(1, dims, device=device)
        zero_idx = []
        samples = aggregate_dist(n, dims)
        for i in range(dims):
            n_samples = samples[i]
            if n_samples == 0:
                zero_idx.append(i)
                continue
            sampler_args['dir'] = i
            conv_res = convolve(func, gauss_grad, input, n=n_samples, sampler=sampler, 
                            f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, aggregrate=aggregate, device=device)
            aggregate_dims = aggregate_dims + conv_res
            result[:, i] = conv_res[i]
        aggregate_dims = aggregate_dims/dims
        print(result[:, zero_idx].shape, aggregate_dims[:, zero_idx].shape)
        result[:, zero_idx] = aggregate_dims[zero_idx]
        return result
    if aggregate:
        return grad_aggregate
    return grad

def smoothFn_gradient_mi(func, n, f_args, kernel_args, sampler_args, device='cuda'):
    '''
    Get the gradient of the convolved function, but with n complexity
    '''
    def grad(input):
        '''
        input should be tensor of shape (1, m)
        '''
        result = convolve(func, gauss_grad_1d, input, n=n, sampler='importance_gradgauss_1d',
                          f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)
        return result.view(1, -1)
    return grad

def smoothFn_hv_fd(func, n, f_args, kernel_args, sampler_args, epsilon=1e-4, aggregate=False, device='cpu'):

    '''
    Uses importance grad gauss sampler
    return a function that does HV product using finite difference from gradient
    '''
    def HV(input, v):
        '''
        input should be tensor of shape (1, m)
        v should be tensor of shape (1, m)
        returns tensor of shape (1, m)
        '''
        v_norm = v.norm()
        unit_vec = v.view(1, -1)/v_norm
        grad = smoothFn_gradient(func, 'importance_gradgauss', n, f_args, kernel_args, sampler_args, aggregate=aggregate, device=device)
        unit_Hv = (grad(input + epsilon*unit_vec) - grad(input-epsilon*unit_vec))/2/epsilon
        return v_norm*unit_Hv
    return HV

def smoothFn_hessian(func, sampler, n, f_args, kernel_args, sampler_args, device='cuda'):
    '''
    Get the hessian of the convolved function,
    n is number of samples
    Returns a function that takes input tensor and returns the hessian
    '''
    # TODO: make more than 2D
    def hess(input):
        '''
        input should be tensor of shape (1, m)
        '''
        dims = input.shape[1]
        result = torch.zeros(dims, dims, device=device)
        for i in range(dims):
            for j in range(i, dims):
                sampler_args['dir'] = (i, j)
                result[i, j] = result[j, i] = convolve(func, gauss_hessian, input, n=n, sampler=sampler, 
                                f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)
        return result
    
    return hess

def smoothFn_hessian_diag(func, sampler, n, f_args, kernel_args, sampler_args, device='cuda'):
    '''
    Get the diagonal hessian of the convolved function,
    n is number of samples
    Returns a function that takes input tensor and returns the hessian
    '''
    def hess(input):
        '''
        input should be tensor of shape (1, m)
        '''
        dims = input.shape[1]
        result = torch.zeros(1, dims, device=device)
        for i in range(dims):
            sampler_args['dir'] = (i, i)
            result[0, i] = convolve(func, gauss_hessian, input, n=n, sampler=sampler, 
                            f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)
        return result
    return hess

def smoothFn_grad_AG_mi(func, n, f_args, clip=None, max_norm=10, device='cuda'):
    '''
    Provide n samples, and f_args here
    Provide input, sampler_args and kernel_args in the context_args later because they are varying
    '''

    @functools.wraps(func)
    def wrapper(input_tensor, kernel_args, sampler_args, *args):
        class SmoothedFun(Function):
            @staticmethod
            def forward(ctx, input_tensor, context_args, *args):
                grad_func = smoothFn_gradient_mi(func=func, n=n, f_args=f_args, kernel_args=kernel_args, 
                                              sampler_args=sampler_args, device=device)
                grad = grad_func(input_tensor)
                if clip:
                    grad = torch.clamp(grad, -max_norm, max_norm)
                #save for backward
                ctx.grad = grad
                ctx.original_input_shape = input_tensor.shape
                return grad.mean()

            @staticmethod
            def backward(ctx, dy):
                original_input_shape = ctx.original_input_shape
                fw_out = ctx.grad
                grad_in_chain = dy * fw_out
                return grad_in_chain.reshape(original_input_shape), None, None

        return SmoothedFun.apply(input_tensor, kernel_args, sampler_args, *args)
    return wrapper
    

def smoothFn_grad_AG(func, n, f_args, clip=None, max_norm=10, device='cuda'):
    '''
    Provide n samples, and f_args here
    Provide input, sampler_args and kernel_args in the context_args later because they are varying
    '''

    @functools.wraps(func)
    def wrapper(input_tensor, kernel_args, sampler_args, *args):
        class SmoothedFun(Function):
            @staticmethod
            def forward(ctx, input_tensor, context_args, *args):
                grad_func = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=n, f_args=f_args,kernel_args=kernel_args, 
                                              sampler_args=sampler_args, device=device)
                grad = grad_func(input_tensor)
                if clip:
                    grad = torch.clamp(grad, -max_norm, max_norm)
                #save for backward
                ctx.grad = grad
                ctx.original_input_shape = input_tensor.shape
                return grad.mean()

            @staticmethod
            def backward(ctx, dy):
                original_input_shape = ctx.original_input_shape
                fw_out = ctx.grad
                grad_in_chain = dy * fw_out
                return grad_in_chain.reshape(original_input_shape), None, None

        return SmoothedFun.apply(input_tensor, kernel_args, sampler_args, *args)
    return wrapper

# helpers
def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite using Cholesky factorization.
    
    Parameters:
    matrix (np.ndarray): The input matrix to check.
    
    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        np.linalg.cholesky(matrix.cpu().numpy())
        return True
    except np.linalg.LinAlgError:
        return False
    
# from Michael's
def update_sigma_linear(it, sigma_0, sigma_min, n=400, const_first=100):
    new_sig = sigma_0 - (it - const_first) * (sigma_0 - sigma_min) / (n - const_first)
    return new_sig if new_sig > sigma_min else sigma_min


# from Michael's
def run_scheduler_step(curr_sigma, curr_iter, sigma_initial, sigma_min, n, const_first_n, const_last_n=None):
    n_real = n - const_last_n if const_last_n else n
    newsigma = update_sigma_linear(curr_iter, sigma_initial, sigma_min, n_real, const_first_n)
    return newsigma


def sigma_scheduler(iter, hparams, sigma):
    if iter > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
        sigma = run_scheduler_step(sigma, curr_iter=iter+1, sigma_initial=hparams['sigma'], sigma_min=hparams['anneal_sigma_min'], 
                                   n=hparams['epochs'], const_first_n=hparams['anneal_const_first'], const_last_n=hparams['anneal_const_last'])
    # print(f'New sigma: {sigma}')
    return sigma




# --------- optimization functions ------------
def adam_opt(func, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    '''
    opt_args: 'epochs', 'learning_rate', 'tol', 'plot_interval', 'conv_thres', 'sigma_annealing', 'sigma', 'anneal_const_first', 'anneal_sigma_min', 'anneal_const_last'
    '''
    self_kernel_args = kernel_args.copy()
    self_sampler_args = sampler_args.copy()
    optim = torch.optim.Adam([x0], lr=opt_args['learning_rate'])
    conv_thres = opt_args.get('conv_thres', 10)
    diff_func = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=ctx_args['nsamples'], f_args=f_args,
                        kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    diff_func_check = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=500, f_args=f_args,
                        kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    img_errors, param_errors, iter_times = [], [], []
    convergence = 0
    converged = False
    sigma = self_sampler_args['sigma']
    for i in range(max_iter):
        start_time = time.time()
        optim.zero_grad()
        self_kernel_args['sigma'] = sigma
        self_sampler_args['sigma'] = sigma
        diff_func = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=ctx_args['nsamples'], f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device) 
        x0.grad = diff_func(x0.unsqueeze(0)).squeeze(0)
        optim.step()
        iter_time = time.time() - start_time
        iter_times.append(iter_time)
        img_errors, param_errors = log_func(x0.unsqueeze(0), img_errors, param_errors, i, interval=opt_args['plot_interval'], iter_time=iter_time)
        if torch.norm(diff_func(x0.unsqueeze(0))) < opt_args['tol']:
            if converged:
                convergence += 1
            else:
                convergence = 1
            converged = True
            # print(convergence)
            if convergence >= conv_thres:
                if torch.norm(diff_func_check(x0.unsqueeze(0))) < opt_args['tol']:
                    break
                else:
                    convergence = 0
        else:
            converged = False
        if opt_args.get('sigma_annealing', False):
            sigma = sigma_scheduler(i, opt_args, sigma)
    return x0, img_errors, param_errors, iter_times

def mi_opt(func, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    '''
    opt_args: 'epochs', 'learning_rate', 'tol', 'plot_interval', 'conv_thres', 'sigma_annealing', 'sigma', 'anneal_const_first', 'anneal_sigma_min', 'anneal_const_last'
    '''
    self_kernel_args = kernel_args.copy()
    self_sampler_args = sampler_args.copy()
    optim = torch.optim.Adam([x0], lr=opt_args['learning_rate'])
    conv_thres = opt_args.get('conv_thres', 10)
    diff_func = smoothFn_gradient_mi(func=func, n=ctx_args['nsamples'], f_args=f_args,
                        kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    diff_func_check = smoothFn_gradient_mi(func=func, n=500, f_args=f_args,
                        kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    img_errors, param_errors, iter_times = [], [], []
    convergence = 0
    converged = False
    sigma = self_sampler_args['sigma']
    for i in range(max_iter):
        start_time = time.time()
        optim.zero_grad()
        self_kernel_args['sigma'] = sigma
        self_sampler_args['sigma'] = sigma
        diff_func = smoothFn_gradient_mi(func=func, n=ctx_args['nsamples'], f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device) 
        x0.grad = diff_func(x0.unsqueeze(0)).squeeze(0)
        optim.step()
        iter_time = time.time() - start_time
        iter_times.append(iter_time)
        img_errors, param_errors = log_func(x0.unsqueeze(0), img_errors, param_errors, i, interval=opt_args['plot_interval'], iter_time=iter_time)
        if torch.norm(diff_func(x0.unsqueeze(0))) < opt_args['tol']:
            if converged:
                convergence += 1
            else:
                convergence = 1
            converged = True
            # print(convergence)
            if convergence >= conv_thres:
                if torch.norm(diff_func_check(x0.unsqueeze(0))) < opt_args['tol']:
                    break
                else:
                    convergence = 0
        else:
            converged = False
        if opt_args.get('sigma_annealing', False):
            sigma = sigma_scheduler(i, opt_args, sigma)
    return x0, img_errors, param_errors, iter_times
    

def newton_smooth(f, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    '''
    Non-linear CG for minimizing the function f.
    x0: initial guess as a row vector
    log_function should take (x, y_histroy, x_histroy, i, interval) and return updated y_histroy, x_histroy
    '''
    self_kernel_args = kernel_args.copy()
    self_sampler_args = sampler_args.copy()
    tol = opt_args['tol']
    start_lr = opt_args['learning_rate']
    modified = opt_args.get('hessian mod', False)
    interval = opt_args['plot_interval']
    n_samples = ctx_args['nsamples']
    sigma = self_kernel_args['sigma']
    TR = opt_args.get('TR', False)
    TR_bound = opt_args.get('TR_bound', 10)
    TR_rate = opt_args.get('TR_rate', 0.01)
    conv_thres = opt_args.get('conv_thres', 3)
    diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                            kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device) 
    
    hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    x_list = []
    x = x0.unsqueeze(0)
    x_copy = x0.cpu().squeeze().detach().numpy()
    x_list.append(x_copy)
    
    img_errors, param_errors, iter_times = [], [], []
    convergence = 0
    converged = False
    for i in range(max_iter):
        self_kernel_args['sigma'] = sigma
        self_sampler_args['sigma'] = sigma
        diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device) 
        
        hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                    kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
        start_time = time.time()
        # adaptive learning rate
        lr = start_lr*(1-i/max_iter+1e-1)
        # if TR_bound == 'dynamic':
        #     TR_bound = 2*max(sigma, 30)
        deriv = diff_func(x).T
        hessian = hess_func(x)
        if modified:
            while is_positive_definite(hessian) == False:
                hessian = TR_rate*torch.eye(hessian.shape[0]).to(device)+hessian

        step = torch.pinverse(hessian)@deriv
        if step.norm() > TR_bound and TR:
            step = step/step.norm()*TR_bound
        x = x - lr*step.T
        x_copy = x.cpu().squeeze().detach().numpy()
        x_list.append(x_copy)
        if len(x_list) >= 20:
            if torch.linalg.vector_norm(diff_func(x).T) < tol and np.sum((x_list[-1] - x_list[-2])**2) < tol:
                if converged:
                    convergence += 1
                else:
                    convergence = 1
                converged = True
                # print(convergence)
                if convergence >= conv_thres:
                    diff_func_check = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=500, f_args=f_args,
                                    kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
                    if diff_func_check(x).norm() < tol:
                        print("converged at: ", i)
                        break
                    else:
                        convergence = 0
            else:
                converged = False
        iter_times.append(time.time()-start_time)   
        img_errors, param_errors = log_func(x, img_errors, param_errors, i, interval=interval, iter_time=time.time()-start_time)
        if opt_args.get('sigma_annealing', False):
            sigma = sigma_scheduler(i, opt_args, sigma)
    return x, img_errors, param_errors, iter_times



def NCG_smooth(f, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    """
    Non-linear CG for minimizing the function f.
    x0: initial guess as a row vector
    log_function should take (x, y_histroy, x_histroy, i, interval) and return updated y_histroy, x_histroy
    ctx_args: 'nsamples'
    opt_args: convergence: 'epochs', 'conv_thres(whole number)', 'tol'
              line search: 'NR_max_iter', 'NR_tol', 'TR(T/F)', 'TR_bound(fixed value or 'dynamic')'
              hessian: 'HVP(T/F)', 'recompute',
              sigma: 'sigma_annealing', 'sigma', 'anneal_const_first', 'anneal_sigma_min', 'anneal_const_last'
              logging:  'plot_interval'
    """
    self_kernel_args = kernel_args.copy()
    self_sampler_args = sampler_args.copy()
    tol = opt_args['tol']
    NR_max_iter = opt_args['NR_max_iter']
    NR_tol = opt_args['NR_tol']
    recompute = opt_args['recompute']
    interval = opt_args['plot_interval']
    n_samples = ctx_args['nsamples']
    TR = opt_args.get('TR', False)
    conv_thres = opt_args.get('conv_thres', 2)
    TR_bound = opt_args.get('TR_bound', 'dynamic') # fixed value or dynamic
    Using_HVP = opt_args.get('HVP', True) # use faster HVP instead
    sigma = self_kernel_args['sigma']
    
    aggregate = False
    print('Starting NCG with sigma: {:2f}, TR: {}, TR_bound: {}, HVP: {}'.format(sigma, TR, TR_bound, Using_HVP))
    diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                            kernel_args=self_kernel_args, sampler_args=self_sampler_args, aggregate=aggregate, device=device) 
    
    hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    hvp_func = smoothFn_hv_fd(func=f, n=n_samples, f_args=f_args, kernel_args=self_kernel_args, sampler_args=self_sampler_args, epsilon=sigma/3, aggregate=aggregate, device=device)
    img_errors, param_errors, iter_times = [], [], []
    x = x0.unsqueeze(0)
    k = 0
    r = -diff_func(x).T # r should be column vector
    d = r
    delta_new = (r.T@r).item()
    tolerance = tol**2 * delta_new # when ||r_i|| <= tol * ||r_0||
    num_iter = max_iter
    x_list = []
    convergence = 0
    converged = False
    modified = False # controls the convergence
    for i in range(max_iter):
        start_time = time.time()
        self_kernel_args['sigma'] = sigma
        self_sampler_args['sigma'] = sigma
        if TR_bound == 'dynamic':
            TR_bound = 2*max(sigma, 5)
        diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, aggregate=aggregate, device=device) 

        hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                    kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
        hvp_func = smoothFn_hv_fd(func=f, n=n_samples, f_args=f_args, kernel_args=self_kernel_args, sampler_args=self_sampler_args, epsilon=sigma/3, aggregate=aggregate, device=device)
        delta_d = d.T@d
        
        for j in range(NR_max_iter): # newton-ralphson iterative approximation
            if Using_HVP:
                hvp = hvp_func(x, d).T
            else:
                hvp = hess_func(x)@d
            # print('full: ', hess_func(x)@d)
            # print('hvp: ', hvp_func(x, d).T)
            denom = d.T@hvp
            if denom > 0:
                modified = False
            else:
                # modify the hessian to be pd(follow the gradient)
                modified = True
                denom = 1/sigma#d.T@hessian@d
            alpha = -(diff_func(x)@d / denom).item()
            step = alpha*d.squeeze()
            if step.norm() > TR_bound*sigma and TR:
                step = step/step.norm()*TR_bound*sigma
                
            x = x + step
            if alpha**2 * delta_d <= NR_tol:
                break
        r = -diff_func(x).T
        delta_old = delta_new
        delta_new = r.T@r
        beta = delta_new/delta_old
        d = r + beta*d
        x_list.append(x.cpu().squeeze().detach().numpy())
        if k>=recompute or r.T@d <=0: # restart whenever a search direction is computed that is not descent direction
            k = 0
            d = r
        # if param_errors and param_errors[-1] < 0.03:
        #     hessian_epsilon = 8e-4
        end_time = time.time()
        iter_times.append(end_time-start_time)
        img_errors, param_errors = log_func(x, img_errors, param_errors, i, interval=interval, iter_time=end_time-start_time)
        if delta_new <= tolerance or (len(x_list) > 20 and torch.linalg.vector_norm(r) < tol and np.sum((x_list[-1] - x_list[-2])**2) < tol):
            if converged and not modified:
                convergence += 1
            else:
                convergence = 1
            converged = True
            # print(convergence)
            if convergence >= conv_thres:
                diff_func_check = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=500, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
                if diff_func_check(x).norm() < tol:
                    num_iter = i+1
                    print("Converged at ", i+1)
                    # img_errors, param_errors = log_func(x, img_errors, param_errors, i=i, interval=1)
                    break
                else:
                    convergence = 0
        else:
            converged = False
        if opt_args.get('sigma_annealing', False):
            sigma = sigma_scheduler(i, opt_args, sigma)
    return x, img_errors, param_errors, iter_times


def BFGS_opt_torch(func, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    '''
    BFGS for minimizing the function f.
    x0: initial guess as a row vector
    log_function should take (x, y_histroy, x_histroy, i, interval) and return updated y_histroy, x_histroy
    opt_args: convergence: 'epochs', 'conv_thres(whole number)', 'tol'
              line search: 'TR(T/F)', 'TR_bound(fixed value or 'dynamic')'
              hessian: 'memory_size'
              sigma: 'sigma_annealing', 'sigma', 'anneal_const_first', 'anneal_sigma_min', 'anneal_const_last'
              logging:  'plot_interval'
    '''
    self_kernel_args = kernel_args.copy()
    self_sampler_args = sampler_args.copy()
    x0 = x0.unsqueeze(0).requires_grad_(True)
    tol = opt_args['tol']
    start_lr = opt_args.get('learning_rate', 1.0)
    history_size = opt_args.get('history_size', 10)
    interval = opt_args['plot_interval']
    TR = opt_args.get('TR', False)
    TR_bound = opt_args.get('TR_bound', 10)
    n_samples = ctx_args['nsamples']
    sigma = self_kernel_args['sigma']
    line_search_fn = opt_args.get('line_search_fn', None)
        
    BFGS = optim.LBFGS([x0], lr=start_lr, line_search_fn=line_search_fn, max_iter=20, history_size=history_size)
    # func_ag = smoothFn_grad_AG_mi(func, n=n_samples, f_args=f_args, clip=TR, max_norm=TR_bound, device=device)
    func_ag = smoothFn_grad_AG(func, n=n_samples, f_args=f_args, clip=TR, max_norm=TR_bound, device=device)
    def closure():
        BFGS.zero_grad()
        loss = func_ag(x0, self_kernel_args, self_sampler_args)
        loss.backward()
        return loss
    img_errors, param_errors = [], []
    # BFGS.step(init_closure)
    for i in range(max_iter):
        self_kernel_args['sigma'] = sigma
        self_sampler_args['sigma'] = sigma
        BFGS.step(closure)
        img_errors, param_errors = log_func(x0, img_errors, param_errors, i, interval=interval)
        if opt_args.get('sigma_annealing', False):
            sigma = sigma_scheduler(i, opt_args, sigma)
            
    return x0, img_errors, param_errors

def lbfgs_two_loop_recursion(grad, s_history, y_history, gamma=None):
    """
    L-BFGS two-loop recursion for computing the product of the approximate inverse Hessian with the gradient.

    - grad: Current gradient (tensor) at iteration k.
    - s_history: List of tensors, storing the steps (s_k = x_{k+1} - x_k) from the past m iterations.
    - y_history: List of tensors, storing the gradient differences (y_k = ∇f(x_{k+1}) - ∇f(x_k)) from the past m iterations.
    - gamma: Optional scaling factor for the Hessian approximation. If None, it will be computed from the last step and gradient pair.

    Returns: The product of the inverse Hessian approximation with the gradient.
    """
    
    m = len(s_history)  # Number of stored vectors
    if m == 0:
        if gamma is not None:
            return gamma*grad
        return grad  # If no history, return gradient unchanged (identity Hessian approximation)
    
    # Initialize alpha and beta
    alpha = []
    rho = []
    
    # First loop (backward)
    q = grad.clone().squeeze()
    for i in range(m - 1, -1, -1):
        s_i = s_history[i]
        y_i = y_history[i]
        rho_i = 1.0 / torch.dot(y_i, s_i)
        rho.append(rho_i)
        alpha_i = rho_i * torch.dot(s_i, q)
        alpha.append(alpha_i)
        q -= alpha_i * y_i
    
    # Compute H0 gamma if not provided
    if gamma is None:
        gamma = torch.dot(s_history[-1], y_history[-1]) / torch.dot(y_history[-1], y_history[-1])
    
    # Scaling factor applied
    r = gamma.view(q.shape) * q
    
    # Second loop (forward)
    for i in range(m):
        s_i = s_history[i]
        y_i = y_history[i]
        beta_i = rho[i] * torch.dot(y_i, r)
        r += s_i * (alpha[i] - beta_i)
    
    return r.unsqueeze(1)


def BFGS_opt(func, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    '''
    BFGS for minimizing the function f.
    x0: initial guess as a row vector
    opt_args: convergence: 'epochs', 'conv_thres(whole number)', 'tol'
              line search: 'TR(T/F)', 'TR_bound(fixed value or 'dynamic')'
              hessian: 'memory_size'
              sigma: 'sigma_annealing', 'sigma', 'anneal_const_first', 'anneal_sigma_min', 'anneal_const_last'
              logging:  'plot_interval'
    '''
    self_kernel_args = kernel_args.copy()
    self_sampler_args = sampler_args.copy()
    n_samples = ctx_args['nsamples']
    
    tol = opt_args['tol']
    lr = opt_args.get('learning_rate', 1.0)
    history_size = opt_args.get('history_size', 10)
    interval = opt_args['plot_interval']
    TR = opt_args.get('TR', False)
    TR_bound = opt_args.get('TR_bound', 10)
    TR_rate = opt_args.get('TR_rate', 0.01)
    conv_thres = opt_args.get('conv_thres', 3)
    
    sigma = self_kernel_args['sigma']
    
    x_list = []
    x_copy = x0.cpu().squeeze().detach().numpy()
    x_list.append(x_copy)
    x = x0.unsqueeze(0)
    diff_func = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                            kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device) 
    hess_func = smoothFn_hessian_diag(func=func, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
    B = 1/hess_func(x).view(-1, 1)
    img_errors, param_errors = [], []
    convergence = 0
    converged = False
    s_hist, y_hist = [], []
    starting = True
    for i in range(max_iter):
        start_time = time.time()
        self_kernel_args['sigma'] = sigma
        self_sampler_args['sigma'] = sigma
        diff_func = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                                kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device) 
        hess_func = smoothFn_hessian_diag(func=func, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                    kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
        if not starting:
            old_deriv = deriv
        # adaptive learning rate
        deriv = diff_func(x).T
        if not starting:
            yk = (deriv - old_deriv).squeeze()
            # print(sk.shape, yk.shape)
            while torch.dot(sk, yk) <= 0:
                # print(yk, sk)
                yk = yk + TR_rate*sk
                # print(yk)
            y_hist.append(yk)
        starting = False
        # print('s_hist length: ', len(s_hist))
        # print('y_hist length: ', len(y_hist))
        step = lbfgs_two_loop_recursion(deriv, s_hist, y_hist, gamma=B)
            
        
        if step.norm() > TR_bound and TR:
            step = step/lr/step.norm()*TR_bound
            
        sk = -(lr*step).squeeze()
        s_hist.append(sk)
        if sk.norm() == 0:
            # print('here')
            B = 1/hess_func(x).view(-1, 1)
            s_hist, y_hist = [], []
            starting = True
        if len(s_hist) >= history_size:
            
            s_hist.pop(0)
            y_hist.pop(0)
        
        x = x - lr*step.T
        x_copy = x.cpu().squeeze().detach().numpy()
        x_list.append(x_copy)
        end_time = time.time()
        img_errors, param_errors = log_func(x, img_errors, param_errors, i, interval=interval, iter_time=end_time-start_time)
        if len(x_list) >= 20:
            if torch.linalg.vector_norm(diff_func(x)) < tol and np.sum((x_list[-1] - x_list[-2])**2) < tol:
                if converged:
                    convergence += 1
                else:
                    convergence = 1
                converged = True
                # print(convergence)
                if convergence >= conv_thres:
                    diff_func_check = smoothFn_gradient(func=func, sampler='importance_gradgauss', n=500, f_args=f_args,
                                    kernel_args=self_kernel_args, sampler_args=self_sampler_args, device=device)
                    if diff_func_check(x).norm() < tol:
                        print("converged at: ", i)
                        break
                    else:
                        convergence = 0
            else:
                converged = False
        if opt_args.get('sigma_annealing', False):
            sigma = sigma_scheduler(i, opt_args, sigma)
            
    return x0, img_errors, param_errors
