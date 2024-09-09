import torch
import drjit as dr
import mitsuba as mi

from utils_optim import run_optimization, run_grad_optimization, run_cg_optimization, run_bfgs_optimization
from utils_general import run_scheduler_step
from utils_mitsuba import setup_shadowscene

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
               'initial_translation': hparams['initial_translation'],
               'gt_translation': hparams['gt_translation'],
               'integrator': hparams,
               'max_depth': hparams['max_depth'],
               'reparam_max_depth': hparams['reparam_max_depth'],
               'epochs': 400,
               'learning_rate': 2e-2, # 1st order
               'sigma_annealing': True,
               'anneal_const_first': 50,
               'anneal_const_last': 200,
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
               'learning_rate': 5e-2, # 1st order
               'sigma_annealing': True,
               'anneal_const_first': 50,
               'anneal_const_last': 0,
               'anneal_sigma_min': 0.01}
    
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
                    'anneal_const_last': 10,
                    'anneal_sigma_min': 0.005,
                    'epochs': 20,
                    'conv_thres': 5, # convergence threshold
                    'tol': 5e-3, # tolerance for CG
                    'TR':True,
                    'TR_bound': 1.5, # number or 'dynamic'
                    'HVP':True, # using HVP or full hessian
                    'NR_max_iter': 3, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 2, # recompute the exact residual every n iterations
                    }
    cg_hparams2 = {'resx': hparams['resx'],
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
                    'anneal_const_first': 10,
                    'anneal_const_last': 0,
                    'anneal_sigma_min': 0.01,
                    'epochs': 50,
                    'conv_thres': 5, # convergence threshold
                    'tol': 5e-5, # tolerance for CG
                    'TR':True,
                    'TR_bound': 5, # number or 'dynamic'
                    'HVP':False, # using HVP or full hessian
                    'NR_max_iter': 10, # max iter for NR line search in CG
                    'NR_tol': 1e-3, # tolerance for NR line search in CG
                    'recompute': 1, # recompute the exact residual every n iterations
                    }
    
    BFGS_box_hparams = {'resx': hparams['resx'],
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
                        'epochs': 100,
                        'anneal_const_first': 10,
                        'anneal_const_last': 60,
                        'anneal_sigma_min': 0.05,
                        'TR':True,
                        'TR_bound': 0.5,
                        'TR_rate': 0.2,
                        'learning_rate':5e-2,
                        'line_search_fn': None, #'strong_wolfe',
                        'history_size': 20,
                        'tol': 5e-6, # tolerance for newton
                        'plot_interval':1000 # number of iterations to plot
                        }
    plot_initial = True
    plot_intermediate = False
    plot_interval = 500

    device = 'cuda'
    torch.manual_seed(0)
    update_fn = apply_translation

    # --------------- set up initial and gt translation:
    initial_translation = torch.tensor(hparams['initial_translation'], requires_grad=True, device=device)
    gt_translation = torch.tensor(hparams['gt_translation'], device=device)

    # --------------- set up optimizer: only for Michael's
    optim = torch.optim.Adam([initial_translation], lr=mi_hparams['learning_rate'])
    

    # --------------- set up scene:
    scene, params, mat_id, initial_vertex_positions = setup_shadowscene(hparams)
    dr.disable_grad(params)

    # --------------- set up ctx_args
    ctx_args = {'scene': scene, 'params': params, 'spp': hparams['render_spp'],                     # rendering
                'init_vpos': initial_vertex_positions, 'mat_id': mat_id, 'update_fn': update_fn,    # rendering
                'sampler': 'importance', 'antithetic': True, 'nsamples': hparams['nsamples'],       # ours
                'sigma': hparams['sigma'], 'device': device}                                        # ours

    # Michael original
    # run_optimization(hparams=mi_hparams,
    #                  optim=optim,
    #                  theta=initial_translation,
    #                  gt_theta=gt_translation,
    #                  ctx_args=ctx_args,
    #                  schedule_fn=run_scheduler_step,
    #                  update_fn=apply_translation,
    #                  plot_initial=plot_initial,
    #                  plot_interval=plot_interval,
    #                  plot_intermediate=plot_intermediate)
    
    # My adam:
    run_grad_optimization(hparams=adam_hparams,
                     theta=initial_translation,
                     gt_theta=gt_translation,
                     ctx_args=ctx_args,
                     update_fn=apply_translation,
                     plot_initial=plot_initial,
                     plot_interval=plot_interval)

    # My CG:
    # run_cg_optimization(hparams=cg_hparams,
    #                  theta=initial_translation,
    #                  gt_theta=gt_translation,
    #                  ctx_args=ctx_args,
    #                  update_fn=apply_translation,
    #                  plot_initial=plot_initial,
    #                  plot_interval=plot_interval)
    
    # My BFGS
    # run_bfgs_optimization(hparams=BFGS_box_hparams,
    #                  theta=initial_translation,
    #                  gt_theta=gt_translation,
    #                  ctx_args=ctx_args,
    #                  update_fn=apply_translation,
    #                  plot_initial=plot_initial,
    #                  plot_interval=plot_interval)