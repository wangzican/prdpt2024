import time
import torch
import utils_fns
import sys
import os 
this_dir = os.path.dirname(os.path.abspath(__file__))
code_path = os.path.join(this_dir, 'code')
sys.path.append(code_path)
# print(this_dir)
# print(code_path)
from utils_general import show_with_error, plt_errors
from utils_mitsuba import get_mts_rendering, render_smooth
from optimizations import adam_opt, NCG_smooth, BFGS_opt, mi_opt

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
    start = time.time()
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

    end = time.time() - start
    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print(f"Done. Time: {end:.4f}")
    print("Done.")

def run_grad_optimization(hparams,
                     theta,
                     gt_theta,
                     ctx_args,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     print_param=False):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image
    device = ctx_args['device']
    # adam
    n_args = {'nsamples':hparams['nsamples']}

    adam_box_params = {'sigma_annealing': hparams['sigma_annealing'],
            'sigma': sigma,
            'anneal_const_first': hparams['anneal_const_first'],
            'anneal_const_last': hparams['anneal_const_last'],
            'anneal_sigma_min': hparams['anneal_sigma_min'],
            'epochs': hparams['epochs'],
            'learning_rate':hparams['learning_rate'],
            'plot_interval':plot_interval, # number of iterations to plot
            'conv_thres': 10, # convergence threshold
            'tol':1e-7
            }
    # --------------- run optimization Adam
    f_args = {'update_fn': update_fn, 'ctx_args': ctx_args}
    kernel_args = {'sigma': sigma}
    sampler_args = {'sigma': sigma, 'is_antithetic': ctx_args['antithetic'], 'dir':(0,0)}
    
    def logging_func(theta, img_errors, param_errors, i, interval=5, **kwargs):
        with torch.no_grad():
            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta.squeeze(), update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())
            
            iter_time = kwargs.get('iter_time', 0)
            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {i + 1}/{hparams['epochs']}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")
            
            if (i+1) % interval == 0: 
                show_with_error(img_curr, reference_image, iter=i+1)
                plt_errors(img_errors, param_errors, title=f'Iter {i+1}')
        return img_errors, param_errors
    
    
    if plot_initial:
        show_with_error(initial_image, reference_image, 0)
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")
    start = time.time()
    x_adam, img_errors, param_errors, iter_times = adam_opt(render_smooth, theta, adam_box_params['epochs'], log_func=logging_func, f_args=f_args, kernel_args=kernel_args,
                                                sampler_args=sampler_args, opt_args=adam_box_params, ctx_args=n_args, device=device)
    end = time.time() - start

    img_curr = get_mts_rendering(x_adam.squeeze(), update_fn, ctx_args)
    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print(f"Done. Time: {end:.4f}")
    print("Done.")
    return img_errors, param_errors, iter_times


def run_cg_optimization(hparams,
                     theta,
                     gt_theta,
                     ctx_args,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     print_param=True):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image
    device = ctx_args['device']
    # adam
    n_args = {'nsamples':hparams['nsamples']}

    cg_box_hparams = {'sigma_annealing': hparams['sigma_annealing'],
           'sigma': sigma,
           'anneal_const_first': hparams['anneal_const_first'],
           'anneal_const_last': hparams['anneal_const_last'],
           'anneal_sigma_min':hparams['anneal_sigma_min'],
           'epochs': hparams['epochs'],
           'conv_thres': 5, # convergence threshold
           'tol': hparams['tol'], # tolerance for CG
           'TR':hparams['TR'], # trust region
           'TR_bound': hparams['TR_bound'], # number or 'dynamic'
           'HVP':hparams['HVP'], # using HVP or full hessian
           'NR_max_iter': hparams['NR_max_iter'], # max iter for NR line search in CG
           'NR_tol': hparams['NR_tol'], # tolerance for NR line search in CG
           'recompute': hparams['recompute'], # recompute the exact residual every n iterations
           'plot_interval':plot_interval # number of iterations to plot
           }
    # --------------- run optimization Adam
    f_args = {'update_fn': update_fn, 'ctx_args': ctx_args}
    kernel_args = {'sigma': sigma}
    sampler_args = {'sigma': sigma, 'is_antithetic': ctx_args['antithetic'], 'dir':(0,0)}
    
    def logging_func(theta, img_errors, param_errors, i, interval=5, **kwargs):
        with torch.no_grad():
            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta.squeeze(), update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())
            
            iter_time = kwargs.get('iter_time', 0)
            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {i + 1}/{hparams['epochs']}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")
            
            if (i+1) % interval == 0: 
                show_with_error(img_curr, reference_image, iter=i+1)
                plt_errors(img_errors, param_errors, title=f'Iter {i+1}')
        return img_errors, param_errors
    
    
    if plot_initial:
        show_with_error(initial_image, reference_image, 0)
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")
    start = time.time()
    x_cg, img_errors, param_errors, iter_times = NCG_smooth(render_smooth, theta, cg_box_hparams['epochs'], log_func=logging_func, f_args=f_args, kernel_args=kernel_args,
                                                sampler_args=sampler_args, opt_args=cg_box_hparams, ctx_args=n_args, device=device)
    end = time.time() - start

    # img_curr = get_mts_rendering(x_cg.squeeze(), update_fn, ctx_args)
    # plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    # show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    # print(f"Done. Time: {end:.4f}")
    # print("Done.")
    return img_errors, param_errors, iter_times
    
    
    

def run_bfgs_optimization(hparams,
                     theta,
                     gt_theta,
                     ctx_args,
                     update_fn,
                     plot_initial,
                     plot_interval,
                     print_param=False):
    sigma = hparams['sigma']

    reference_image = get_mts_rendering(gt_theta, update_fn, ctx_args)
    initial_image = get_mts_rendering(theta, update_fn, ctx_args)
    ctx_args['gt_image'] = reference_image
    device = ctx_args['device']
    # adam
    n_args = {'nsamples':hparams['nsamples']}

    bfgs_hparams = {'sigma_annealing': hparams['sigma_annealing'],
           'sigma': sigma,
           'anneal_const_first': hparams['anneal_const_first'],
           'anneal_const_last': hparams['anneal_const_last'],
           'anneal_sigma_min':hparams['anneal_sigma_min'],
           'epochs': hparams['epochs'],
           'conv_thres': 5, # convergence threshold
           'tol': 5e-5, # tolerance for CG
           'TR':hparams['TR'], # trust region
           'TR_bound': hparams['TR_bound'], # number or 'dynamic'
           'TR_rate': hparams['TR_rate'],
           'learning_rate':hparams['learning_rate'],
           'history_size': hparams['history_size'],
           'line_search_fn': hparams['line_search_fn'], #'strong_wolfe',
           'plot_interval':plot_interval # number of iterations to plot
           }
    # --------------- run optimization Adam
    f_args = {'update_fn': update_fn, 'ctx_args': ctx_args}
    kernel_args = {'sigma': sigma}
    sampler_args = {'sigma': sigma, 'is_antithetic': ctx_args['antithetic'], 'dir':(0,0)}
    
    def logging_func(theta, img_errors, param_errors, i, interval=5, **kwargs):
        with torch.no_grad():
            # calc loss btwn rendering with current parameter (non-blurred)
            img_curr = get_mts_rendering(theta.squeeze(), update_fn, ctx_args)
            img_errors.append(torch.nn.MSELoss()(img_curr, ctx_args['gt_image']).item())
            param_errors.append(torch.nn.MSELoss()(theta, gt_theta).item())
            
            iter_time = kwargs.get('iter_time', 0)
            pstring = ' - CurrentParam: {}'.format(theta.tolist()) if print_param else ''
            print(f"Iter {i + 1}/{hparams['epochs']}, ParamLoss: {param_errors[-1]:.6f}, "
                  f"ImageLoss: {img_errors[-1]:.8f} - Time: {iter_time:.4f}{pstring}")
            
            if (i+1) % interval == 0: 
                show_with_error(img_curr, reference_image, iter=i+1)
                plt_errors(img_errors, param_errors, title=f'Iter {i+1}')
        return img_errors, param_errors
    
    
    if plot_initial:
        show_with_error(initial_image, reference_image, 0)
    print(f"Running {hparams['epochs']} epochs with {hparams['nsamples']} samples and sigma={hparams['sigma']}")
    start = time.time()
    x_cg, img_errors, param_errors = BFGS_opt(render_smooth, theta, bfgs_hparams['epochs'], log_func=logging_func, f_args=f_args, kernel_args=kernel_args,
                                                sampler_args=sampler_args, opt_args=bfgs_hparams, ctx_args=n_args, device=device)
    end = time.time() - start

    img_curr = get_mts_rendering(x_cg.squeeze(), update_fn, ctx_args)
    plt_errors(img_errors, param_errors, title=f'Final, after {hparams["epochs"]} iterations')
    show_with_error(img_curr, ctx_args['gt_image'], hparams['epochs'])
    print(f"Done. Time: {end:.4f}")
    print("Done.")