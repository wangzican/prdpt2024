import torch
import drjit as dr
import mitsuba as mi
import numpy as np
import time

from utils_optim import run_optimization, run_grad_optimization, run_cg_optimization, run_bfgs_optimization
from utils_general import run_scheduler_step, show_with_error
from utils_mitsuba import setup_shadowscene, get_mts_rendering_mts

if torch.cuda.is_available():
    print("is available")
    mi.set_variant('cuda_ad_rgb')
    
    # device = 'cpu'
    # mi.set_variant('scalar_rgb')


def apply_translation(theta, p, mat_id, init_vpos):
    if isinstance(theta, torch.Tensor):
        theta = theta.tolist()
    trans = mi.Transform4f.translate([0.0, theta[0], theta[1]])
    p[mat_id] = dr.ravel(trans @ init_vpos)
    p.update()


if __name__ == '__main__':

    hparams = {'resx': 256,
               'resy': 192,
               'nsamples': 1,
               'sigma': 0.5,
               'render_spp': 32,
               'initial_translation': [-0.5, 2.5],
               'gt_translation': [-1.5, 1.0],
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
               'epochs': 350,
               'learning_rate': 2e-2, # 1st order
               'sigma_annealing': True,
               'anneal_const_first': 150,
               'anneal_const_last': 0,
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
               'epochs': 130,
               'learning_rate': 2.5e-2, # 1st order
               'sigma_annealing': True,
               'anneal_const_first': 50,
               'anneal_const_last': 70,
               'anneal_sigma_min': 1e-4,
               'plot_interval': 60}
    
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
                    'anneal_const_first': 0,
                    'anneal_const_last': 15,
                    'anneal_sigma_min': 1e-2,
                    'epochs': 40,
                    'conv_thres': 4, # convergence threshold
                    'tol': 1e-4, # tolerance for CG
                    'TR':True,
                    'TR_bound': 3, # number or 'dynamic'
                    'HVP':True, # using HVP or full hessian
                    'NR_max_iter': 1, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 10, # recompute the exact residual every n iterations
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
                    'sigma': 0.6,#hparams['sigma'],
                    'render_spp': hparams['render_spp'],
                    'integrator': hparams,
                    'max_depth': hparams['max_depth'],
                    'reparam_max_depth': hparams['reparam_max_depth'],
                    'sigma_annealing': True,
                    'anneal_const_first': 20,
                    'anneal_const_last': 10,
                    'anneal_sigma_min': 0.01,
                    'epochs': 110,
                    'conv_thres': 5, # convergence threshold
                    'tol': 5e-5, # tolerance for CG
                    'TR':True,
                    'TR_bound': 4, # number or 'dynamic'
                    'HVP':False, # using HVP or full hessian
                    'NR_max_iter': 10, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 5, # recompute the exact residual every n iterations
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
    plot_interval = 600

    device = 'cuda'
    torch.manual_seed(0)
    update_fn = apply_translation
    
    n_starting_points = 20
    # starting_points = (torch.rand(n_starting_points, 2, device=device)*torch.tensor([-0.2, 0.2], device=device) + torch.tensor([0.1, -0.1], device=device) + torch.tensor(hparams['initial_translation'], device=device))
    # print(starting_points)
    # np.save('./code/results/shadow/shadow_sp.npy', starting_points.cpu())
    initial_translations = np.load('./code/results/shadow/shadow_sp.npy')
    initial_translations = torch.tensor(initial_translations, device=device)
    # --------------- set up initial and gt translation:
    # initial_translation = torch.tensor(hparams['initial_translation'], requires_grad=True, device=device)
    gt_translation = torch.tensor(hparams['gt_translation'], device=device)

    # --------------- set up optimizer: only for Michael's
    # optim = torch.optim.Adam([initial_translation], lr=mi_hparams['learning_rate'])
    

    # --------------- set up scene:
    scene, params, mat_id, initial_vertex_positions = setup_shadowscene(hparams)
    dr.disable_grad(params) # comment for others except for Mitsuba

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
    
    
    # Mitsuba
    
    # dr.enable_grad(params) 
    # mi.set_variant('cuda_ad_rgb')
    # def mse(image):
    #     loss = dr.sum(dr.sqr(image - reference_image))
    #     return loss 
    # def update_fn_mitsuba():
    #     trans = mi.Transform4f.translate(mitsuba_opt['translation'])
    #     params[mat_id] = dr.ravel(trans @ initial_vertex_positions)
    #     params.update()
        
    # mat_id = 'PLYMesh_1.vertex_positions'
    # reference_image = get_mts_rendering_mts(gt_translation, update_fn, ctx_args)
    # torch_reference_image = torch.tensor(reference_image, dtype=torch.float32, device=device)
    
    # i = 0
    # print(f'initial translation: {initial_translations[i,0]}, {initial_translations[i,1]}')
    # print(f'gt translation: {gt_translation[0]}, {gt_translation[1]}')
    # start = mi.Vector3f(0, float(initial_translations[i, 0]), float(initial_translations[i, 1]))
    # gt = mi.Vector3f(0, float(gt_translation[0]), float(gt_translation[1]))
    
    # mitsuba_opt = mi.ad.Adam(lr=0.05)
    # mitsuba_opt['translation'] = start
    
    # # update params
    # update_fn_mitsuba()
    # rendering = mi.render(ctx_args['scene'], params, seed=0, spp=ctx_args['spp'])
    # img_loss = mse(rendering)
    # param_loss = dr.mean(dr.sqr(mitsuba_opt['translation'] - gt))
    # print(f"img_Loss at start: {img_loss[0]}")
    # print(f'param_loss at start: {param_loss[0]}')
    # torch_rendering = torch.tensor(rendering, dtype=torch.float32, device=device)
    # show_with_error(torch_rendering, torch_reference_image, 0)
    
    # max_iter = 100
    # start_time = time.time()
    # for it in range(max_iter):
    #     print(f'iter {it}')
        
    #     rendering = mi.render(ctx_args['scene'], params, seed=0, spp=ctx_args['spp'])
    #     if (it+1) % 100 == 0:
    #         torch_rendering = torch.tensor(rendering, dtype=torch.float32, device=device)
    #         show_with_error(torch_rendering, torch_reference_image, it)
    #     img_loss = mse(rendering)
    #     param_loss = dr.mean(dr.sqr(mitsuba_opt['translation'] - gt))
    #     dr.backward(img_loss)
    #     mitsuba_opt.step()
    #     update_fn_mitsuba()
    #     print(f"img_Loss at iter {it}: {img_loss[0]}")
    #     print(f'param_loss at iter {it}: {param_loss[0]}')
    #     mitsuba_opt.step()
    #     if img_loss[0] < 0.000005:
    #         print(f'converged at iter {it}')
    #         break
    # end_time = time.time()
    # print(f'time: {end_time - start_time}')
    
    
    # Michael original
    
    # for i in range(n_starting_points):
    #     initial_translation = initial_translations[i].clone().detach().requires_grad_(True)
    #     optim = torch.optim.Adam([initial_translation], lr=mi_hparams['learning_rate'])
        # run_optimization(hparams=mi_hparams.clone(),
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
    
    # i = 0
    # print(f"Starting point {i}")
    # initial_translation = initial_translations[i].clone().detach().requires_grad_(True)
    
    # func_loss, param_loss, iter_times = run_grad_optimization(hparams=mi_hparams.copy(),
    #                 theta=initial_translation,
    #                 gt_theta=gt_translation,
    #                 ctx_args=ctx_args.copy(),
    #                 update_fn=apply_translation,
    #                 plot_initial=plot_initial,
    #                 plot_interval=plot_interval)
    # length = len(func_loss)
    # for j, loss_i in enumerate(func_loss):
    #     if loss_i < 0.000005:
    #         idx = j#min(j+6, length-1)
    #         func_loss = func_loss[:idx+1]
    #         param_loss = param_loss[:idx+1]
    #         iter_times = iter_times[:idx+1]
    # np.save(f'./code/results/shadow/shadow_mi/shadow_mi_f_loss_{i}.npy', func_loss)
    # np.save(f'./code/results/shadow/shadow_mi/shadow_mi_param_loss_{i}.npy', param_loss)
    # np.save(f'./code/results/shadow/shadow_mi/shadow_mi_times_{i}.npy', iter_times)
    
    # My adam:
    # i = 19
    # print(f"Starting point {i}")
    # initial_translation = initial_translations[i].clone().detach().requires_grad_(True)
    
    # func_loss, param_loss, iter_times = run_grad_optimization(hparams=adam_hparams.copy(),
    #                 theta=initial_translation,
    #                 gt_theta=gt_translation,
    #                 ctx_args=ctx_args.copy(),
    #                 update_fn=apply_translation,
    #                 plot_initial=plot_initial,
    #                 plot_interval=plot_interval)
    # length = len(func_loss)
    # for j, loss_i in enumerate(func_loss):
    #     if loss_i < 0.000005:
    #         idx = j#min(j+6, length-1)
    #         func_loss = func_loss[:idx+1]
    #         param_loss = param_loss[:idx+1]
    #         iter_times = iter_times[:idx+1]
    # np.save(f'./code/results/shadow/shadow_adam/shadow_adam_f_loss_{i}.npy', func_loss)
    # np.save(f'./code/results/shadow/shadow_adam/shadow_adam_param_loss_{i}.npy', param_loss)
    # np.save(f'./code/results/shadow/shadow_adam/shadow_adam_times_{i}.npy', iter_times)

    # My CG:
    cg_hparams_HVP_agg = {'resx': hparams['resx'],
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
                    'anneal_const_first': 0,
                    'anneal_const_last': 10,
                    'anneal_sigma_min': 5e-2,
                    'epochs': 30,
                    'conv_thres': 4, # convergence threshold
                    'tol': 1e-4, # tolerance for CG
                    'TR':True,
                    'TR_bound': 4, # number or 'dynamic'
                    'HVP':True, # using HVP or full hessian
                    'aggregate': True,
                    'NR_max_iter': 1, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 5, # recompute the exact residual every n iterations
                    }    
    i = 3
    torch.manual_seed(0)
    initial_translation = initial_translations[i].clone()
    func_loss, param_loss, iter_times = run_cg_optimization(hparams=cg_hparams_HVP_agg.copy(),
                                                            theta=initial_translation.clone(),
                                                            gt_theta=gt_translation,
                                                            ctx_args=ctx_args.copy(),
                                                            update_fn=apply_translation,
                                                            plot_initial=plot_initial,
                                                            plot_interval=plot_interval)
    length = len(func_loss)
    for j, loss_i in enumerate(func_loss):
        if loss_i < 0.000008:
            idx = j#min(j+6, length-1)
            func_loss = func_loss[:idx+1]
            param_loss = param_loss[:idx+1]
            iter_times = iter_times[:idx]
    iter_times = np.insert(iter_times, 0, 0)
    np.save(f'./code/results/shadow/shadow_cg_HVP_agg/shadow_cg_HVP_agg_f_loss_{i}.npy', func_loss)
    np.save(f'./code/results/shadow/shadow_cg_HVP_agg/shadow_cg_HVP_agg_param_loss_{i}.npy', param_loss)
    np.save(f'./code/results/shadow/shadow_cg_HVP_agg/shadow_cg_HVP_agg_times_{i}.npy', iter_times)
    # My BFGS
    # run_bfgs_optimization(hparams=BFGS_box_hparams,
    #                  theta=initial_translation,
    #                  gt_theta=gt_translation,
    #                  ctx_args=ctx_args,
    #                  update_fn=apply_translation,
    #                  plot_initial=plot_initial,
    #                  plot_interval=plot_interval)