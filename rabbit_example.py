import torch
import drjit as dr
import mitsuba as mi
import numpy as np

from utils_optim import run_optimization, run_grad_optimization, run_cg_optimization, run_bfgs_optimization
from utils_general import run_scheduler_step
from utils_mitsuba import setup_rabbitscene

if torch.cuda.is_available():
    print("is available")
    mi.set_variant('cuda_ad_rgb')
    
    # device = 'cpu'
    # mi.set_variant('scalar_rgb')

def apply_transformation(theta, p, mat_id, init_vpos):
    if isinstance(theta, torch.Tensor):
        theta = theta.tolist()
    trafo = mi.Transform4f.translate([float(theta[0]), float(theta[1]), 0.0]).rotate([0, 1, 0], float(theta[2]) *100.0)

    p[mat_id] = dr.ravel(trafo @ init_vpos)
    p.update()

if __name__ == '__main__':

    hparams = {'resx': 32,
               'resy': 32,
               'nsamples': 1,
               'sigma': 0.004,
               'render_spp': 32,
               'initial_translation': [0.25, -0.25, 0.25],
               'gt_translation': [0, 0, 0],
               'integrator': 'direct_projective',
               'integrator': 'path',
               'max_depth': 6,
               'reparam_max_depth': 2}
    
    mi_hparams = {'resx': hparams['resx'],
               'resy': hparams['resy'],
               'nsamples': hparams['nsamples'],
               'sigma': hparams['sigma'],
               'render_spp': hparams['render_spp'],
               'integrator': hparams,
               'max_depth': hparams['max_depth'],
               'reparam_max_depth': hparams['reparam_max_depth'],
               'epochs': 300,
               'learning_rate': 2e-2, # 1st order
               'sigma_annealing': False,
               'anneal_const_first': 0,
               'anneal_const_last': 300,
               'anneal_sigma_min': 0.01}
    
    adam_hparams = {'resx': hparams['resx'],
               'resy': hparams['resy'],
               'nsamples': hparams['nsamples'],
               'sigma': hparams['sigma'],
               'render_spp': hparams['render_spp'],
               'initial_translation': hparams['initial_translation'],
               'gt_translation': hparams['gt_translation'],
               'integrator': hparams,
               'max_depth': hparams['max_depth'],
               'reparam_max_depth': hparams['reparam_max_depth'],
               'epochs': 100,
               'learning_rate': 3e-2, # 1st order
               'sigma_annealing': False,
               'anneal_const_first': 50,
               'anneal_const_last': 100,
               'anneal_sigma_min': 1e-2
               }
    
    cg_hparams = {'resx': hparams['resx'],
                    'resy': hparams['resy'],
                    'nsamples': hparams['nsamples'],
                    'sigma': hparams['sigma'],
                    'render_spp': hparams['render_spp'],
                    'initial_translation': hparams['initial_translation'],
                    'gt_translation': hparams['gt_translation'],
                    'integrator': hparams,
                    'max_depth': hparams['max_depth'],
                    'reparam_max_depth': hparams['reparam_max_depth'],
                    'sigma_annealing': True,
                    'anneal_const_first':  0,
                    'anneal_const_last': 0,
                    'anneal_sigma_min': 3e-3,
                    'epochs': 400,
                    'conv_thres': 700, # convergence threshold
                    'tol': 1e-10, # tolerance for CG
                    'TR':True,
                    'TR_bound': 0.2, # number or 'dynamic'
                    'HVP':True, # using HVP or full hessian
                    'NR_max_iter': 5, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 5, # recompute the exact residual every n iterations
                    }    
    # cg_hparams = {'resx': hparams['resx'],
    #                 'resy': hparams['resy'],
    #                 'nsamples': hparams['nsamples'],
    #                 'sigma': hparams['sigma'],
    #                 'render_spp': hparams['render_spp'],
    #                 'initial_translation': hparams['initial_translation'],
    #                 'gt_translation': hparams['gt_translation'],
    #                 'integrator': hparams,
    #                 'max_depth': hparams['max_depth'],
    #                 'reparam_max_depth': hparams['reparam_max_depth'],
    #                 'sigma_annealing': True,
    #                 'anneal_const_first': 0,
    #                 'anneal_const_last': 25,
    #                 'anneal_sigma_min': 1e-2,
    #                 'epochs': 40,
    #                 'conv_thres': 4, # convergence threshold
    #                 'tol': 1e-4, # tolerance for CG
    #                 'TR':True,
    #                 'TR_bound': 6, # number or 'dynamic'
    #                 'HVP':True, # using HVP or full hessian
    #                 'NR_max_iter': 1, # max iter for NR line search in CG
    #                 'NR_tol': 1e-3, # tolerance for NR line search in CG
    #                 'recompute': 40, # recompute the exact residual every n iterations
    #                 }
    cg_hparams2 = {'resx': hparams['resx'],
                    'resy': hparams['resy'],
                    'nsamples': hparams['nsamples'],
                    'sigma': hparams['sigma'],
                    'render_spp': hparams['render_spp'],
                    'integrator': hparams,
                    'max_depth': hparams['max_depth'],
                    'reparam_max_depth': hparams['reparam_max_depth'],
                    'sigma_annealing': True,
                    'anneal_const_first': 0,
                    'anneal_const_last': 15,
                    'anneal_sigma_min': 0.01,
                    'epochs': 30,
                    'conv_thres': 55, # convergence threshold
                    'tol': 5e-9, # tolerance for CG
                    'TR':True,
                    'TR_bound': 4, # number or 'dynamic'
                    'HVP':True, # using HVP or full hessian
                    'NR_max_iter': 2, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 2, # recompute the exact residual every n iterations
                    }
    
    BFGS_box_hparams = {'resx': hparams['resx'],
                        'resy': hparams['resy'],
                        'nsamples': hparams['nsamples'],
                        'sigma': hparams['sigma'],
                        'render_spp': hparams['render_spp'],
                        'integrator': hparams,
                        'max_depth': hparams['max_depth'],
                        'reparam_max_depth': hparams['reparam_max_depth'],
                        'sigma_annealing': True,
                        'epochs': 20,
                        'anneal_const_first': 0,
                        'anneal_const_last': 0,
                        'anneal_sigma_min': 0.05,
                        'TR':True,
                        'TR_bound': 0.5,
                        'TR_rate': 0.2,
                        'learning_rate':5e-2,
                        'line_search_fn': None, #'strong_wolfe',
                        'history_size': 10,
                        'tol': 5e-6, # tolerance for newton
                        'plot_interval':1000 # number of iterations to plot
                        }
    plot_initial = False
    plot_intermediate = False
    plot_interval = 2000

    device = 'cuda'
    # torch.manual_seed(0)
    update_fn = apply_transformation
    
    n_starting_points = 20
    # starting_points = (torch.rand(n_starting_points, 3, device=device)*torch.tensor([1], device=device) + torch.tensor([0], device=device) + torch.tensor(hparams['initial_translation'], device=device))
    # print(starting_points)
    # np.save('./code/results/mug/mug_sp.npy', starting_points.cpu())
    # initial_translations = np.load('./code/results/mug/mug_sp.npy')
    # initial_translations = torch.tensor(initial_translations, device=device)
    
    # --------------- set up initial and gt translation:
    # initial_translation = torch.tensor(hparams['initial_translation'], requires_grad=True, device=device)
    gt_translation = torch.tensor(hparams['gt_translation'], device=device)

    # --------------- set up optimizer: only for Michael's
    # optim = torch.optim.Adam([initial_translation], lr=mi_hparams['learning_rate'])
    

    # --------------- set up scene:
    scene, params, mat_id, initial_vertex_positions = setup_rabbitscene(hparams)
    dr.disable_grad(params)

    # --------------- set up ctx_args
    ctx_args = {'scene': scene, 'params': params, 'spp': hparams['render_spp'],                     # rendering
                'init_vpos': initial_vertex_positions, 'mat_id': mat_id, 'update_fn': update_fn,    # rendering
                'sampler': 'importance', 'antithetic': True, 'nsamples': hparams['nsamples'],       # ours
                'sigma': hparams['sigma'], 'device': device}                                        # ours

    # from utils_mitsuba import get_mts_rendering
    # from utils_general import show_with_error, plt_errors
    # for i in range(n_starting_points):
    #     print(initial_translations[i])
    #     reference_image = get_mts_rendering(gt_translation, update_fn, ctx_args)
    #     initial_image = get_mts_rendering(initial_translations[i], update_fn, ctx_args)
    #     show_with_error(initial_image, reference_image, 0)
    
    # Michael original
    
    # for i in range(n_starting_points):
    #     initial_translation = initial_translations[i].clone().detach().requires_grad_(True)
    #     optim = torch.optim.Adam([initial_translation], lr=mi_hparams['learning_rate'])
    #     run_optimization(hparams=mi_hparams.clone(),
    #                     optim=optim,
    #                     theta=initial_translation,
    #                     gt_theta=gt_translation,
    #                     ctx_args=ctx_args,
    #                     schedule_fn=run_scheduler_step,
    #                     update_fn=apply_translation,
    #                     plot_initial=plot_initial,
    #                     plot_interval=plot_interval,
    #                     plot_intermediate=plot_intermediate)
    #     np.save(f'./code/results/shadow/shadow_cg_HVP/shadow_cg_HVP_f_loss_{i}.npy', func_loss)
    #     np.save(f'./code/results/shadow/shadow_cg_HVP/shadow_cg_HVP_param_loss_{i}.npy', param_loss)
    #     np.save(f'./code/results/shadow/shadow_cg_HVP/shadow_cg_HVP_times_{i}.npy', iter_times)
    
    #my FR:
    # change in run_grad function  adam_opt->mi_opt
    # i = 0
    # # initial_translation = initial_translations[i].clone().detach().requires_grad_(True)
    # initial_translation = torch.tensor(hparams['initial_translation'],device=device)
    # print(f"Starting point {i} is {initial_translation}")
    
    # func_loss, param_loss, iter_times = run_grad_optimization(hparams=mi_hparams.copy(),
    #                 theta=initial_translation,
    #                 gt_theta=gt_translation,
    #                 ctx_args=ctx_args.copy(),
    #                 update_fn=update_fn,
    #                 plot_initial=plot_initial,
    #                 plot_interval=plot_interval)
    # length = len(param_loss)
    # for j, loss_i in enumerate(param_loss):
    #     if loss_i <= 0.00002:
    #         idx = j#min(j+6, length-1)
    #         func_loss = func_loss[:idx+1]
    #         param_loss = param_loss[:idx+1]
    #         iter_times = iter_times[:idx+1]
    
    # np.save(f'./code/results/rabbit/rabbit_mi/rabbit_mi_f_loss_{i}.npy', func_loss)
    # np.save(f'./code/results/rabbit/rabbit_mi/rabbit_mi_param_loss_{i}.npy', param_loss)
    # np.save(f'./code/results/rabbit/rabbit_mi/rabbit_mi_times_{i}.npy', iter_times)
    
    # My adam:
    # i = 0
    # print(f"Starting point {i}")
    # # initial_translation = initial_translations[i].clone().detach().requires_grad_(True)
    # initial_translation = torch.tensor(hparams['initial_translation'],device=device)
    
    # func_loss, param_loss, iter_times = run_grad_optimization(hparams=adam_hparams.copy(),
    #                 theta=initial_translation,
    #                 gt_theta=gt_translation,
    #                 ctx_args=ctx_args.copy(),
    #                 update_fn=update_fn,
    #                 plot_initial=plot_initial,
    #                 plot_interval=plot_interval)
    # length = len(func_loss)
    # for j, loss_i in enumerate(func_loss):
    #     if loss_i < 0.00002:
    #         idx = j#min(j+6, length-1)
    #         func_loss = func_loss[:idx+1]
    #         param_loss = param_loss[:idx+1]
    #         iter_times = iter_times[:idx+1]
    
    # np.save(f'./code/results/rabbit/rabbit_adam/rabbit_adam_f_loss_{i}.npy', func_loss)
    # np.save(f'./code/results/rabbit/rabbit_adam/rabbit_adam_param_loss_{i}.npy', param_loss)
    # np.save(f'./code/results/rabbit/rabbit_adam/rabbit_adam_times_{i}.npy', iter_times)

    # My CG:
    i = 0
    
    # initial_translation = initial_translations[i].clone()
    # s = 10, 10, 0.5
    
    initial_translation = torch.tensor(hparams['initial_translation'],device=device)
    
    func_loss, param_loss, iter_times = run_cg_optimization(hparams=cg_hparams.copy(),
                                                            theta=initial_translation.clone(),
                                                            gt_theta=gt_translation,
                                                            ctx_args=ctx_args.copy(),
                                                            update_fn=update_fn,
                                                            plot_initial=plot_initial,
                                                            plot_interval=plot_interval,
                                                            print_param=True)
    length = len(func_loss)
    for j, loss_i in enumerate(func_loss):
        if loss_i < 0.00001:
            idx = j#min(j+6, length-1)
            func_loss = func_loss[:idx+1]
            param_loss = param_loss[:idx+1]
            iter_times = iter_times[:idx]
    iter_times = np.insert(iter_times, 0, 0)
    np.save(f'./code/results/rabbit/rabbit_cg_HVP/rabbit_cg_HVP_f_loss_{i}.npy', func_loss)
    np.save(f'./code/results/rabbit/rabbit_cg_HVP/rabbit_cg_HVP_param_loss_{i}.npy', param_loss)
    np.save(f'./code/results/rabbit/rabbit_cg_HVP/rabbit_cg_HVP_times_{i}.npy', iter_times)
    
    # My BFGS
    # run_bfgs_optimization(hparams=BFGS_box_hparams,
    #                  theta=initial_translation,
    #                  gt_theta=gt_translation,
    #                  ctx_args=ctx_args,
    #                  update_fn=apply_translation,
    #                  plot_initial=plot_initial,
    #                  plot_interval=plot_interval)