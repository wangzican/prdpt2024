import torch
import numpy as np


# === Section 1: Gaussians, derivatives and respective distributions ===
def gauss_1d(x, mu=0.0, sigma=1.0):
    return 1.0 / (sigma * (2.0 * np.pi)**0.5) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gauss_grad_1d(x, sigma, *args, **kwargs):
    grad_of_gauss = -(x / sigma ** 2) * gauss_1d(x, mu=0.0, sigma=sigma)
    return grad_of_gauss

def gaussian_nd(x, sigma, **kwargs):
    """
    Assuming a isotropic gaussian distribution with mean at 0
    Takes x in forms of nxm, where n is the number of samples and m is the dimension
    Returns the gaussian value at each point with shape (n,)
    """
    batch_n, m = x.shape
    sigma2 = sigma ** 2
    norm_squared = torch.sum(x ** 2, dim=1)
    prefactor = 1 / ((2 * torch.pi)**0.5 * sigma)**m
    exp_component = torch.exp(-norm_squared / (2 * sigma2))
    
    return prefactor * exp_component

    
def gauss_grad(x, sigma, dir=None, **kwargs):
    '''
    Gradient of a nd gaussian function, the dimension is determined by the shape of x
    Takes in a tensor x of shape (n, m), where n is the number of samples and m is the dimension
    Returns the gradient of shape (n, m), where for each sample, each dimension has a gradient
    '''
    factor = x if dir is None else x[:, dir].view(-1, 1)
    
    return (-factor.T / sigma ** 2 * gaussian_nd(x, sigma)).T


def gauss_grad_as_pdf(x, sigma, dir=None, no_pos=False):
    '''
    Same as gauss grad, but the integral is 1, and positivised so that it is a valid distribution
    returns gradient of shape (n,m), each dimension has a gradient.
    '''
    m = x.shape[-1]
    sigma2 = sigma ** 2
    norm_squared = torch.sum(x ** 2, dim=1)
    prefactor = 1 / ((2 * torch.pi)**0.5 * sigma)**(m-1) # there is one dim need to be 0.5
    
    # if no_pos:
    #     prefactor = -prefactor*0.5 * x / sigma2
    # else:
    #     prefactor = prefactor*0.5 * torch.abs(x) / sigma2
    pre_x = x
    if dir is not None:
        pre_x = pre_x[:, dir].view(-1, 1)
    if no_pos:
        pre_x = pre_x
    else:
        pre_x = -torch.abs(pre_x)
        
    prefactor = -prefactor*0.5 * pre_x / sigma2
    broadcast_dims = pre_x.dim() - 1

    # broadcast_dims = x.dim() - 1
    exp_component = torch.exp(-norm_squared / (2 * sigma2)).view(-1, *([1] * broadcast_dims))
    
    return prefactor * exp_component


def gauss_grad_1d_as_cdf(x, sigma, no_pos=False):
    '''
    The cdf of second order derivative of gaussian, which is gauss grad positivised and normalized to 1
    '''
    x = torch.as_tensor(x, dtype=torch.float32)
    fac = 0.25/sigma * np.exp(0.5)
    grad = -x * torch.exp(-0.5 * (x / sigma) ** 2)
    before_pos = fac*grad
    if no_pos:
        return before_pos
    first0 = x<=-sigma
    second0 = (x > -sigma) & (x <= sigma)
    cdf = torch.where(first0, before_pos, torch.where(second0, 0.5 - before_pos, 1 + before_pos))
        
    return cdf

def gauss_hessian(x, sigma=1.0, dir=None, device='cpu'):
    '''
    Compute the hessian of a n dimensional isotropic gaussian function with mean at 0
    Takes x in forms of nxm, where n is the number of samples and m is the dimension
    Returns the hessian at each point with shape (n, m, m)
    '''
    n_sample, dim = x.shape

    sigma2 = sigma ** 2
    sigma4 = sigma ** 4
    
    if dir is None:
        grid = torch.einsum('ij,ik->ijk', x, x)
    else:
        x1 = x[:, dir[0]].view(-1, 1, 1)
        x2 = x[:, dir[1]].view(-1, 1, 1)
        result = x1*x2/sigma4
        if dir[0] == dir[1]:
            result = result - 1/sigma2
        result = result.view(-1, 1, 1) * gaussian_nd(x, sigma).view(-1, 1, 1)
        return result
    # diag of 1/sigma2
    diag = (1/sigma2) * torch.eye(dim)
    diag = diag.unsqueeze(0).to(device) # get (1,m,m) diag constants
    diag = diag.expand(n_sample, -1, -1)
    
    # non-diagonals
    hessians = grid/sigma4 - diag
    hessians = hessians * gaussian_nd(x, sigma).view(-1, 1, 1)
    return hessians

   
def gauss_hessian_as_pdf(x, sigma, dir=None, device='cpu'):
    '''
    For nxm input of x, each row is a point X in parameter space
    Returns a nxmxm tensor. n number of mxm matrices.
    each i,j element in the mxm matrix is the pdf of the i,j element of the hessian for the given xi and xj value
    (mxm distributions sort of)
    if dir is given (i,j), then only the i,j element of the hessian is calculated return a nx1x1 tensor
    '''
    
    n_sample, m = x.shape
    hessians = torch.zeros((n_sample, x.shape[1], x.shape[1])).to(device)
    
    sigma2 = sigma ** 2
    norm_squared = torch.sum(x ** 2, dim=1)
    
    if dir is not None: # for given element
        x1 = x[:, dir[0]].view(-1, 1)
        x2 = x[:, dir[1]].view(-1, 1)
        if dir[0] == dir[1]:
            prefactor = 1 / ((2 * torch.pi)**0.5 * sigma)**(m-1) # there is one dim need to be 0.5
            prefactor = prefactor * np.exp(0.5)*(x1*x2/sigma**2 - 1)*0.25/sigma
            pos = (x1 > -sigma) & (x1 <= sigma)
            result = torch.where(pos, -prefactor, prefactor).view(-1)   
        else:
            prefactor = 1 / ((2 * torch.pi)**0.5 * sigma)**(m-2) # there are two dims need to be 0.5
            result = (prefactor * torch.abs(x1 * x2) * (0.5/sigma2) **2).view(-1)
        result = (result*torch.exp(-norm_squared / (2 * sigma2))).view(-1, 1, 1)
        
        return result

    diag_mask = torch.eye(m, dtype=torch.bool).unsqueeze(0).to(device) # get (1,m,m) mask
    diag_mask = diag_mask.repeat(n_sample, 1, 1)

    non_diag_mask = ~diag_mask
    # diagonals
    prefactor = 1 / ((2 * torch.pi)**0.5 * sigma)**(m-1) # there is one dim need to be 0.5
    prefactor = prefactor * np.exp(0.5)*((x/sigma)**2 - 1)*0.25/sigma
    first0 = x<=-sigma
    second0 = (x > -sigma) & (x <= sigma)
    hessians[diag_mask] = torch.where(first0, prefactor, torch.where(second0, -prefactor, prefactor)).view(-1)
    
    # non-diagonals
    grid = torch.abs(torch.einsum('ij,ik->ijk', x, x))
    prefactor = 1 / ((2 * torch.pi)**0.5 * sigma)**(m-2) # there are two dims need to be 0.5
    hessians[non_diag_mask] = prefactor * grid[non_diag_mask] * (0.5/sigma2) **2
    
    
    broadcast_dims = hessians.dim() - 1
    exp_component = torch.exp(-norm_squared / (2 * sigma2)).view(-1, *([1] * broadcast_dims))
    hessians = hessians * exp_component
    
    return hessians


# === Section 2: Samplers ===
def uniform(n_samples, dim, min=-1.0, max=1.0, device='cpu'):
    '''
    min, max are the same for all dims
    '''
    samples = torch.rand(int(n_samples), dim).to(device)
    samples = min+(max-min)*samples
    
    return samples, 1/((max-min)**dim)

def grid(n_samples, dim, min, max, device='cpu'):
    '''
    min, max are the same for all dims
    '''
    size = int(n_samples ** (1/dim))
    grid = torch.linspace(min, max, size).to(device)
    grid = torch.meshgrid([grid] * dim, indexing='ij')
    grid = torch.stack(grid, dim=-1).view(-1, dim)
    return size**2, grid, 1/((max-min)**dim)


def importance_gauss_nd(n_samples, dim, sigma, is_antithetic=True, pdf=False):
    '''
    Returns a importance sampled points from nd gaussian 
    with shape (nsamples, dim)
    '''
    eps = 0.00001
    randoms = torch.rand(n_samples, dim)
    normal_dist = torch.distributions.normal.Normal(0, sigma)

    # samples and AT samples
    if is_antithetic:
        randoms = torch.cat([randoms, 1.0 - randoms])
        n_samples = n_samples * 2

    # avoid NaNs bc of numerical instabilities in log
    randoms[torch.isclose(randoms, torch.ones_like(randoms))] -= eps
    randoms[torch.isclose(randoms, torch.zeros_like(randoms))] += eps
    
    x_i = normal_dist.icdf(randoms)
    p_xi=1
    if pdf:
        # print('in func x_i: ', x_i.shape)
        p_xi = gaussian_nd(x_i, sigma=sigma)
    # print('in func p_xi: ', p_xi.shape)
    return n_samples, x_i, p_xi



def importance_gradgauss_1d(n_samples, dim, sigma, is_antithetic, pdf=False, **kwargs):
    '''
    Michael's grad gauss IS sampler, now slightly changed for 1d grad gauss
    '''
    eps = 0.00001
    randoms = torch.rand(n_samples, dim)

    def icdf(x, sigma):
        res = torch.zeros_like(x)
        res[mask == 1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * (1.0 - x[mask == 1])))
        res[mask == -1] = torch.sqrt(-2.0 * sigma ** 2 * torch.log(2.0 * x[mask == -1]))
        return res

    # samples and AT samples
    if is_antithetic:
        randoms = torch.cat([randoms, 1.0 - randoms])
        n_samples = n_samples * 2

    # avoid NaNs bc of numerical instabilities in log
    randoms[torch.isclose(randoms, torch.ones_like(randoms))] -= eps
    randoms[torch.isclose(randoms, torch.zeros_like(randoms))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=0.5))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=-0.5))] -= eps

    mask = torch.where(randoms < 0.5, -1.0, 1.0)
    x_i = icdf(randoms, sigma=sigma) * mask
    
    p_xi = 1
    if pdf:
        f_xi = torch.abs(x_i) * (1.0 / sigma ** 2) * gauss_1d(x_i, mu=0.0, sigma=sigma)
        f_xi[f_xi == 0] += eps
        p_xi = 0.5 * sigma * (2.0 * np.pi)**0.5 * f_xi

    return n_samples, x_i, p_xi


def importance_gradgauss_nd(n_samples, dim, sigma, is_antithetic=True, dir=0, device='cpu'):
    '''
    Dim is the dimension of tau (x)
    dir is the direction number (which gradient) x:0, y:1, ...
    '''
    assert dim >= 1
    gauss_dim = dim-1
    _, x_i, _ = importance_gradgauss_1d(n_samples, dim=1, sigma=sigma, is_antithetic=is_antithetic)
    if gauss_dim>0:
        _, x_otherdims, _ = importance_gauss_nd(n_samples, gauss_dim, sigma, is_antithetic)
        x_before = x_otherdims[:, :dir]
        x_after = x_otherdims[:, dir:]
        x_i = torch.cat((x_before, x_i, x_after), dim=1)
    if is_antithetic:
        n_samples *= 2
        x_i = x_i.to(device)
    p_x = gauss_grad_as_pdf(x_i, sigma=sigma, dir=dir)
    return n_samples, x_i, p_x

def importance_gradgauss_v(n_samples, v, dim, sigma, is_antithetic=True, device='cpu'):
    '''
    Importance sampling for the weighted sum of distribution for grad v product (directional derivative)
    '''
    n_samples, x_i, _ = importance_gradgauss_nd(n_samples, dim, sigma, is_antithetic)
    
    return

def inverse_cdf_hess(sigma):
    '''
    Return a function interpolate(y): given cdf interpolate the x from a table of cfd<->x
    '''
    x_range = torch.linspace(-10*sigma, 10*sigma, 2000)
    y_range = gauss_grad_1d_as_cdf(x_range, sigma)
    def interpolation(y):
        return torch.as_tensor(np.interp(y, y_range, x_range, left=0, right=1), dtype=torch.float32)
    return interpolation

def importance_hessgauss_diag(n_samples, sigma, is_antithetic, icdf):
    '''
    Importance sampling for the diagonal of hessian of gaussian
    Only for the diagonals (2ND ORDER DERIVATIVE), so dim is always 1 
    '''
    eps = 0.0001
    
    randoms = torch.rand(n_samples, 1)

    # samples and AT samples
    if is_antithetic:
        randoms = torch.cat([randoms, 1.0 - randoms])
        n_samples = n_samples * 2

    # avoid NaNs bc of numerical instabilities in log
    randoms[torch.isclose(randoms, torch.ones_like(randoms))] -= eps
    randoms[torch.isclose(randoms, torch.zeros_like(randoms))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=0.25))] += eps
    randoms[torch.isclose(randoms, torch.full_like(randoms, fill_value=0.75))] -= eps

    x_i = icdf(randoms)
    
    return n_samples, x_i

def importance_hessgauss_nd(n_samples, dim, sigma, is_antithetic=True, dir=(0,0), device='cpu'):
    '''
    Dim is dimension of tau (x) which is same as parameter space
    dir is the combination of direction. (i,j) means the partial derivative of the ith and jth dimensions
    Diagonal is interpolated result, non-diagonal is two gradgauss samples
    '''
    assert dim >= 1
    
    i, j = dir
    assert i < dim and j < dim, f"hessian entry {i},{j} is out of bounds for dim {dim}"
    
    if i == j: # for diagonals
        icdf = inverse_cdf_hess(sigma)
        gauss_dim = dim-1
        _, x_i = importance_hessgauss_diag(n_samples, sigma=sigma, is_antithetic=is_antithetic, icdf=icdf)
        if gauss_dim>0:
            _, x_otherdims, _ = importance_gauss_nd(n_samples, gauss_dim, sigma, is_antithetic)
            x_before = x_otherdims[:, :i]
            x_after = x_otherdims[:, i:]
            x_i = torch.cat((x_before, x_i, x_after), dim=1)
    else: # non diagonal is same as 2 grad gauss
        gauss_dim = dim-2
        _, x_i, _ = importance_gradgauss_1d(n_samples, dim=1, sigma=sigma, is_antithetic=is_antithetic)
        _, x_j, _ = importance_gradgauss_1d(n_samples, dim=1, sigma=sigma, is_antithetic=is_antithetic)
        if gauss_dim>0:
            _, x_otherdims, _ = importance_gauss_nd(n_samples, gauss_dim, sigma, is_antithetic)
            i, j = min(i, j), max(i, j)
            x_before_i = x_otherdims[:, :i]
            x_between_ij = x_otherdims[:, i:j]
            x_after_j = x_otherdims[:, j:]
            x_i = torch.cat((x_before_i, x_i, x_between_ij, x_j, x_after_j), dim=1)
        else:
            x_i = torch.cat((x_i, x_j), dim=1)
    if is_antithetic:
        n_samples *= 2
    x_i = x_i.to(device)
    p_x = gauss_hessian_as_pdf(x_i, sigma=sigma, dir=dir, device=device)

    return n_samples, x_i, p_x

def importance_hessgauss_v(n_samples, dim, sigma, is_antithetic=True, dir=0, device='cpu'):
    '''
    Importance sampling for the weighted sum of distribution for Hv product
    Dir
    '''
    return

def mc_estimate(f_xi, p_xi, n):
    '''
    n is the number of samplers
    f_xi should be with shape (n,...)
    '''
    estimate = 1. / n * (f_xi / p_xi).sum(axis=0)  # average along batch axis, leave dimension axis unchanged
    return estimate



# === Section 3: Convolution ===
def convolve(f, kernel, point, n, sampler='uniform', f_args={}, kernel_args={}, sampler_args={}, device='cpu'):
    """
    Convolves a function with a kernel
    The function f should take in a tensor of shape (n, m) and return a tensor of shape (n,)
    The kernel should also take in a tensor of shape (n, m) and return a tensor of shape (n, ...)
    Point should be a tensor of shape (1,m), which is a single point with same dimension as the input of f and kernel
    n is the number of samples around the input point to be sampled
    sampler can be: 'uniform', 'grid', 'importance_gradgauss', 'importance_hessgauss'
    
    Returns the result with the shape same as output of the kernel
    """
    
    dims = point.shape[1]
    
    # determine the sampler
    if sampler == 'uniform':
        tau, pdf = uniform(n, dims, device=device, **sampler_args)
    elif sampler == 'grid':
        # changes n to nearest square number
        n, tau, pdf = grid(n, dims, device=device, **sampler_args)
    elif sampler == 'importance_gauss':
        n, tau, pdf = importance_gauss_nd(n, dims, pdf=True, **sampler_args)
        tau=tau.to(device)
        pdf=pdf.to(device)
    elif sampler == 'importance_gradgauss_1d':
        n, tau, pdf = importance_gradgauss_1d(n, dims, pdf=True, **sampler_args)
        tau = tau.to(device)
        pdf=pdf.to(device)
    elif sampler == 'importance_gradgauss':
        n, tau, pdf = importance_gradgauss_nd(n, dims, device=device, **sampler_args)
    elif sampler == 'importance_hessgauss':
        n, tau, pdf = importance_hessgauss_nd(n, dims, device=device, **sampler_args)
    else:
        raise NotImplementedError('sampler not supported')
    
    # shift samples around current parameter
    
    points = torch.cat([point] * n, dim=0) - tau
    f_out = f(points, **f_args)
    if isinstance(f_out, tuple):
        f_values = f_out[0]
    else:
        f_values = f_out
    dir = sampler_args.get('dir', None)
    weights = kernel(x=tau, device=device, dir=dir, **kernel_args)
    
    # weights can be (n, ...), generating (...) of convolutions
    broadcast_dims = weights.dim() - 1
    f_values = f_values.view(-1, *([1] * broadcast_dims))
    
    # mc_estimate 
    convolved_value = mc_estimate(weights * f_values, pdf, n)
    
    return convolved_value


