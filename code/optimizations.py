import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from convolutions import *
from utils_general import run_scheduler_step


def smoothFn_gradient(func, sampler, n, f_args, kernel_args, sampler_args, device='cuda'):
    '''
    Get the gradient of the convolved function,
    n is number of samples
    Returns a function that takes input tensor and returns the gradient
    '''
    # TODO: make more than 2D
    def grad(input):
        '''
        input should be tensor of shape (1, m)
        '''
        result = torch.zeros(1, input.shape[1], device=device)
        dir=0
        sampler_args['dir'] = dir
        result[:, dir] = convolve(func, gauss_grad, input, n=n, sampler=sampler, 
                        f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)[dir]
        dir=1
        sampler_args['dir'] = dir
        result[:, dir] = convolve(func, gauss_grad, input, n=n, sampler=sampler, 
                        f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)[dir]
        return result
    
    return grad

# def hessian_fd(func, x, ctx, epsilon=1e-4, device='cpu'):
# TODO: implement hessian_fd
#     '''
#     x is a row vector with one extra dimension [[a, b, c, ...]]
#     '''
#     new_x = x.clone()
#     new_x = new_x.view(1,-1) # ensures a row vector
#     n = new_x.size(1)
#     hessian_matrix = torch.zeros((n, n), device=device)
        
#     for i in range(n):
#         # Perturb the i-th element
#         x_pi = new_x.clone()
#         x_ni = new_x.clone()
#         x_pi[0,i] += epsilon
#         x_ni[0,i] -= epsilon
#         # Compute the gradient vectors at the perturbed point
#         # thess will be the column vector df/dx_(1~n) dx_i
#         par_pi = grad_fn(func, x_pi, ctx, epsilon)
#         par_ni = grad_fn(func, x_ni, ctx, epsilon)
#         # Compute the second-order partial derivatives
#         hessian_matrix[i] =  (par_pi - par_ni) / 2 / epsilon
    
#     # hessian_matrix[1,1] = 0
#     # hessian_matrix[0,1] = 0
#     # hessian_matrix[1,0] = 0
#     return hessian_matrix.T

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
        result = torch.zeros(input.shape[1], input.shape[1], device=device)
        dir=(0,0)
        sampler_args['dir'] = dir
        result[dir] = convolve(func, gauss_hessian, input, n=n, sampler=sampler, 
                        f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)[dir]
        dir=(1,1)
        sampler_args['dir'] = dir
        result[dir] = convolve(func, gauss_hessian, input, n=n, sampler=sampler, 
                        f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)[dir]
        dir=(1,0)
        sampler_args['dir'] = dir
        result[dir] = result[dir[1], dir[0]] = convolve(func, gauss_hessian, input, n=n, sampler=sampler, 
                        f_args=f_args, kernel_args=kernel_args, sampler_args=sampler_args, device=device)[dir]
        return result
    
    return hess

# optimizers
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
    

def sigma_scheduler(iter, hparams, sigma):
    if iter > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
        sigma = run_scheduler_step(sigma, curr_iter=iter+1, sigma_initial=hparams['sigma'], sigma_min=hparams['anneal_sigma_min'], 
                                   n=hparams['epochs'], const_first_n=hparams['anneal_const_first'], const_last_n=hparams['anneal_const_last'])
    return sigma

def newton_smooth(f, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    '''
    Non-linear CG for minimizing the function f.
    x0: initial guess as a row vector
    log_function should take (x, y_histroy, x_histroy, i, interval) and return updated y_histroy, x_histroy
    '''
    
    tol = opt_args['tol']
    start_lr = opt_args['learning_rate']
    modified = opt_args['hessian mod']
    interval = opt_args['plot_interval']
    n_samples = ctx_args['nsamples']
    sigma = kernel_args['sigma']
    diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                            kernel_args=kernel_args, sampler_args=sampler_args, device=device) 
    
    hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                kernel_args=kernel_args, sampler_args=sampler_args, device=device)
    x_list = []
    x = x0.unsqueeze(0)
    x_copy = x0.cpu().squeeze().detach().numpy()
    x_list.append(x_copy)
    
    img_errors, param_errors = [], []
    for i in range(max_iter):
        kernel_args['sigma'] = sigma
        sampler_args['sigma'] = sigma
        diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                                kernel_args=kernel_args, sampler_args=sampler_args, device=device) 
        
        hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                    kernel_args=kernel_args, sampler_args=sampler_args, device=device)
        # adaptive learning rate
        lr = start_lr*(1-i/max_iter+1e-4)
        deriv = diff_func(x).T
        hessian = hess_func(x)
        if is_positive_definite(hessian) or not modified:
            x = x - lr*(torch.pinverse(hessian)@deriv).T
        else:
            x = x - lr*deriv.T/torch.norm(deriv)
        x_copy = x.cpu().squeeze().detach().numpy()
        x_list.append(x_copy)
        if len(x_list) >= 20:
            if torch.linalg.vector_norm(deriv) < tol and np.sum((x_list[-1] - x_list[-2])**2) < tol:
                print("converged at: ", i)
                break
        img_errors, param_errors = log_func(x, img_errors, param_errors, i, interval=interval)
        sigma = sigma_scheduler(i, opt_args, sigma)
    return x, np.array(x_list)


def NCG_smooth(f, x0, max_iter, log_func, f_args, kernel_args, sampler_args, opt_args, ctx_args, device='cuda'):
    """
    Non-linear CG for minimizing the function f.
    x0: initial guess as a row vector
    log_function should take (x, y_histroy, x_histroy, i, interval) and return updated y_histroy, x_histroy
    """
    tol = opt_args['tol']
    NR_max_iter = opt_args['NR_max_iter']
    NR_tol = opt_args['NR_tol']
    recompute = opt_args['recompute']
    interval = opt_args['plot_interval']
    n_samples = ctx_args['nsamples']
    TR = opt_args.get('TR', False)
    TR_rate = opt_args.get('TR_rate', 0.1)
    sigma = kernel_args['sigma']
    diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                            kernel_args=kernel_args, sampler_args=sampler_args, device=device) 
    
    hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                kernel_args=kernel_args, sampler_args=sampler_args, device=device)
    img_errors, param_errors = [], []
    x = x0.unsqueeze(0)
    k = 0
    r = -diff_func(x).T # r should be column vector
    d = r
    delta_new = (r.T@r).item()
    tolerance = tol**2 * delta_new # when ||r_i|| <= tol * ||r_0||
    num_iter = max_iter
    x_list = []
    for i in range(max_iter):
        kernel_args['sigma'] = sigma
        sampler_args['sigma'] = sigma
        diff_func = smoothFn_gradient(func=f, sampler='importance_gradgauss', n=n_samples, f_args=f_args,
                                kernel_args=kernel_args, sampler_args=sampler_args, device=device) 
        
        hess_func = smoothFn_hessian(func=f, sampler='importance_hessgauss', n=n_samples, f_args=f_args,
                                    kernel_args=kernel_args, sampler_args=sampler_args, device=device)
        delta_d = d.T@d
        for j in range(NR_max_iter): # newton-ralph approximation
            hessian = hess_func(x)      
            if TR:
                hessian = TR_rate*torch.diag(torch.diag(hessian))+hessian
            denom = d.T@(hessian)@d
            # if i == 0:
            assert denom != 0, "try a new starting point"
            # TODO:fix this
            while denom < 0:
                hessian = hess_func(x)    
                if TR:
                    hessian = TR_rate*torch.diag(torch.diag(hessian))+hessian    
                denom = d.T@hessian@d
            alpha = -(diff_func(x)@d / denom).item()
            while (alpha*d).norm()> 0.03:
                hessian = hess_func(x)    
                if TR:
                    hessian = TR_rate*torch.diag(torch.diag(hessian))+hessian    
                denom = d.T@hessian@d
                alpha = -(diff_func(x)@d / denom).item()
            x = x + alpha*d.squeeze()
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
        if delta_new <= tolerance:
            num_iter = i+1
            print("Converged at ", i+1)
            img_errors, param_errors = log_func(x, img_errors, param_errors, i=i, interval=1)
            break
        img_errors, param_errors = log_func(x, img_errors, param_errors, i, interval=interval)
        sigma = sigma_scheduler(i, opt_args, sigma)
    return x, r, np.array(x_list)