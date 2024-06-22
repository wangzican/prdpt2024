import time
import torch
import utils_fns

from utils_general import show_with_error, plt_errors
from utils_mitsuba import get_mts_rendering, render_smooth


def run_optimization(hparams,
                     optim,
                     theta,
                     gt_theta,
                     ctx_args,
                     schedule_fn,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     plot_intermediate,
                     print_param=False):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image

    # --------------- set up smoothed renderer
    smooth_mts = utils_fns.smoothFn(render_smooth,
                                    context_args=None,
                                    device=ctx_args['device'])

    if plot_initial:
        show_with_error(initial_image, reference_image, 0)

    img_errors, param_errors = [], []
    img_errors.append(torch.nn.MSELoss()(initial_image, reference_image).item())
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")

    # --------------- run optimization
    for j in range(hparams['epochs']):
        start = time.time()
        optim.zero_grad()

        loss, _ = smooth_mts(theta.unsqueeze(0), ctx_args)
        loss.backward()

        optim.step()

        # potential sigma scheduling:
        if j > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
            sigma = schedule_fn(sigma, curr_iter=j + 1, n=hparams['epochs'],
                                sigma_initial=hparams['sigma'],
                                sigma_min=hparams['anneal_sigma_min'],
                                const_first_n=hparams['anneal_const_first'],
                                const_last_n=hparams['anneal_const_last'])
            ctx_args['sigma'] = sigma
        iter_time = time.time() - start

        # logging, timing, plotting, etc...
        with torch.no_grad():

            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta, update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())

            # plot intermediate
            if j % plot_interval == 0 and j > 0 and plot_intermediate:

                show_with_error(img_curr, ctx_args['gt_image'], j)

                if len(param_errors) > 1:
                    plt_errors(img_errors, param_errors, title=f'Ep {j + 1}')

            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {j + 1}/{hparams['epochs']}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")

    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print("Done.")



def run_lbfgs_optimization(hparams,
                     optim,
                     theta,
                     gt_theta,
                     ctx_args,
                     schedule_fn,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     plot_intermediate,
                     print_param=False):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image

    # --------------- set up smoothed renderer
    smooth_mts = utils_fns.smoothFn(render_smooth,
                                    context_args=None,
                                    device=ctx_args['device'])

    if plot_initial:
        show_with_error(initial_image, reference_image, 0)

    img_errors, param_errors = [], []
    img_errors.append(torch.nn.MSELoss()(initial_image, reference_image).item())
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")

    # --------------- run optimization
    for j in range(hparams['epochs']):
        start = time.time()        

        if j < 300:
            
            optim.zero_grad()

            loss, _ = smooth_mts(theta.unsqueeze(0), ctx_args)
            loss.backward()

            optim.step()
        else:
            optim = torch.optim.LBFGS([theta],
                                    lr=1,
                                    history_size=10, 
                                    max_iter=1, 
                                    line_search_fn="strong_wolfe")
            
            # L-BFGS
            def closure():
                optim.zero_grad()
                loss, _ = smooth_mts(theta.unsqueeze(0), ctx_args)
                loss.backward()
                return loss
            optim.step(closure)

        # potential sigma scheduling:
        if j > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
            sigma = schedule_fn(sigma, curr_iter=j + 1, n=hparams['epochs'],
                                sigma_initial=hparams['sigma'],
                                sigma_min=hparams['anneal_sigma_min'],
                                const_first_n=hparams['anneal_const_first'],
                                const_last_n=hparams['anneal_const_last'])
            ctx_args['sigma'] = sigma
        iter_time = time.time() - start

        # logging, timing, plotting, etc...
        with torch.no_grad():

            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta, update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())

            # plot intermediate
            if j % plot_interval == 0 and j > 0 and plot_intermediate:

                show_with_error(img_curr, ctx_args['gt_image'], j)

                if len(param_errors) > 1:
                    plt_errors(img_errors, param_errors, title=f'Ep {j + 1}')

            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {j + 1}/{hparams['epochs']}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")

    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print("Done.")


def compute_hessian(func, x, ctx):
    x = x.clone().detach().requires_grad_(True)  # Ensure x requires gradient
    y,_ = func(x, ctx)
    grads = torch.autograd.grad(y, x, create_graph=True)[0]
    hessian_matrix = []
    grads = grads.squeeze()
    for grad in grads:
        print(grad)
        grad.requires_grad_(True)
        grad2 = torch.autograd.grad(grad, x, retain_graph=True)[0]
        hessian_matrix.append(grad2)
    
    return torch.stack(hessian_matrix)

def torch_hessian(func, x, ctx):
    def fun(para):
        return func(para, ctx)[0]
    return torch.autograd.functional.hessian(fun, x)

def hessian_fd(func, x, ctx, epsilon=1e-5):
    
    x = x.clone().detach().requires_grad_(True)
    n = x.size(1)
    hessian_matrix = torch.zeros((n, n), device='cuda')
    
    # Compute the gradient at the original point
    y,_ = func(x, ctx)
    grad = torch.autograd.grad(y, x, create_graph=True)[0].squeeze()
    
    for i in range(n):
        for j in range(n):
            # Perturb the i-th element
            x_pj = x.clone()
            x_pi = x.clone()
            x_pj[0,j] += epsilon
            x_pi[0,i] += epsilon
            # Compute the function value at the perturbed point
            y_pj,_ = func(x_pj, ctx)
            y_pi,_ = func(x_pi, ctx)
            # Compute the gradient at the perturbed point
            grad_pij = torch.autograd.grad(y_pj, x_pj, create_graph=True)[0].squeeze()
            grad_pji = torch.autograd.grad(y_pi, x_pi, create_graph=True)[0].squeeze()
            # print(grad_pij[i] + grad_pji[j] - grad[0] - grad[1])
            # Compute the second-order partial derivatives
            hessian_matrix[i, j] =  (grad_pij[i] + grad_pji[j] - grad[0] - grad[1]) / 2 / epsilon
    
    return hessian_matrix


def run_cg_optimization(hparams,
                     optim,
                     theta,
                     gt_theta,
                     ctx_args,
                     schedule_fn,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     plot_intermediate,
                     print_param=False):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image

    # --------------- set up smoothed renderer
    smooth_mts = utils_fns.smoothFn(render_smooth,
                                    context_args=None,
                                    device=ctx_args['device'])

    if plot_initial:
        show_with_error(initial_image, reference_image, 0)

    img_errors, param_errors = [], []
    img_errors.append(torch.nn.MSELoss()(initial_image, reference_image).item())
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")

    
    # --------------- run optimization
    max_iter = hparams['epochs']
    tol=1e-5
    NR_max_iter=100
    NR_tol=1e-5
    recompute=50
    

    x = theta.requires_grad_(True)
    k = 0
    loss, _ = smooth_mts(x.unsqueeze(0), ctx_args)
    loss.backward()
    r = -x.grad.unsqueeze(0).T
    d = r
    delta_new = r.T@r
    tolerance = tol**2 * delta_new # when ||r_i|| <= tol * ||r_0||
    num_iter = max_iter
    for i in range(max_iter):
        start = time.time()
        delta_d = d.T@d
        for j in range(NR_max_iter): # newton-ralph approximation
            # hess = compute_hessian(smooth_mts, x.unsqueeze(0), ctx_args) # finite difference
            hess = hessian_fd(smooth_mts, x.unsqueeze(0), ctx_args) # pytorch
            hess = hess.squeeze()
            denom = d.T@hess@d 

            assert denom != 0, "try a new starting point"
            loss, _ = smooth_mts(x.unsqueeze(0), ctx_args)
            loss.backward()
            alpha = -(x.grad@d / denom)
            x = x + alpha*d.squeeze()
            
            if alpha**2 * delta_d <= NR_tol:
                break
        x=x.squeeze().detach().clone().requires_grad_(True)
        loss, _ = smooth_mts(x.unsqueeze(0), ctx_args)
        loss.backward()
        # print(x, x.grad)
        r = -x.grad.unsqueeze(0).T
        delta_old = delta_new
        delta_new = r.T@r
        beta = delta_new/delta_old
        d = r + beta*d
        if k>=recompute or r.T@d <=0: # restart whenever a search direction is computed that is not descent direction
            k = 0
            d = r

        theta = x
        if delta_new <= tolerance:
            num_iter = i+1
            break
        # potential sigma scheduling:
        if i > hparams['anneal_const_first'] and hparams['sigma_annealing'] and sigma >= hparams['anneal_sigma_min']:
            sigma = schedule_fn(sigma, curr_iter=j + 1, n=hparams['epochs'],
                                sigma_initial=hparams['sigma'],
                                sigma_min=hparams['anneal_sigma_min'],
                                const_first_n=hparams['anneal_const_first'],
                                const_last_n=hparams['anneal_const_last'])
            ctx_args['sigma'] = sigma
        iter_time = time.time() - start

        # logging, timing, plotting, etc...
        with torch.no_grad():

            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta, update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())

            # plot intermediate
            if j % plot_interval == 0 and j > 0 and plot_intermediate:

                show_with_error(img_curr, ctx_args['gt_image'], j)

                if len(param_errors) > 1:
                    plt_errors(img_errors, param_errors, title=f'Ep {j + 1}')

            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {i + 1}/{hparams['epochs']}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")

    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print("Done.")
