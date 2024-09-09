import os
import drjit as dr
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
from mitsuba import Transform4f as T

mi.set_variant('llvm_ad_rgb')


def create_occlusion_scene(res_x, res_y, integr='path', maxdepth=6, reparam_max_depth=2, include_walls=False,
                           isglasswall=False):
    if integr == 'path':
        integrator = {'type': 'path', 'max_depth': maxdepth}
    elif 'reparam' in integr:
        integrator = {'type': integr, 'max_depth': maxdepth, 'reparam_max_depth': reparam_max_depth}

    lightpos = [0.0, 4.20, 2.0]
    lightInt = 25.6 * 10
    wallcolor = (0.8, 0.8, 0.8)
    texturepath = 'mitsuba_scenes_used/occlusion/grid_50.png'

    plastic_bsdf = {'type': 'plastic',
                    'diffuse_reflectance': {'type': 'bitmap', 'filename': texturepath},
                    'specular_reflectance': {'type': 'rgb', 'value': (0.0, 0.0, 0.0)}}

    scenedict = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(
                origin=(0, 0, 2),
                target=(0, 0, 0),
                up=(0, 1, 0)
            ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': res_x,
                'height': res_y,
                'rfilter': {'type': 'gaussian'},
                'sample_border': True
            },
        },
        'wall': {
            'type': 'obj',
            'filename': 'mitsuba_scenes_used/occlusion/rectangle.obj',
            'to_world': T.translate([0, 10, -12]).scale(20.0),
            'face_normals': True,
            'bsdf': plastic_bsdf
        },
        'light': {
            'type': 'obj',
            'filename': 'mitsuba_scenes_used/occlusion/rectangle.obj',
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [lightInt, lightInt, lightInt]}
            },
            'to_world': T.translate(lightpos).rotate([1, 0, 0], 90)
        },
        'sphereHidden': {
            'type': 'obj',
            'filename': 'mitsuba_scenes_used/occlusion/sphere.obj',
            'to_world': T.translate([0.0, 0.0, -3.2]).scale(0.75),
            'face_normals': True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (0.75, 0.1, 0.1)}
            }
        },
        'sphereOccluding': {
            'type': 'obj',
            'filename': 'mitsuba_scenes_used/occlusion/sphere.obj',
            'to_world': T.translate([0.0, 0.0, -5.0]).scale(0.75),
            'face_normals': True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (0.1, 0.1, 0.75)},
            }
        }
    }

    lightgrey = wallcolor
    if include_walls:
        for j in range(5):
            if j == 0:
                translation, rotation, angle, c = [4.25, 0.0, -2.0], [0, 1, 0], -90, lightgrey
            elif j == 1:
                translation, rotation, angle, c = [-4.25, 0.0, -2.0], [0, 1, 0], 90, lightgrey
            elif j == 2:
                translation, rotation, angle, c = [0.0, 4.20, -2.0], [1, 0, 0], 90, lightgrey
            elif j == 3:
                translation, rotation, angle, c = [0.0, -(1.5 if isglasswall else 4.25), -2.0], [1, 0,
                                                                                                 0], -90, lightgrey
            elif j == 4:
                translation, rotation, angle, c = [0.0, 0.0, 5.0], [0, 0, 0], 0, lightgrey

            if j == 3 and isglasswall:
                # if isglasswall, make lower wall reflecting glass
                bsdf_dict = {'type': 'dielectric', 'int_ior': 1.504, 'ext_ior': 1.0}
            else:
                bsdf_dict = plastic_bsdf

            scenedict['wall{}'.format(j)] = {
                'type': 'obj',
                'filename': 'mitsuba_scenes_used/occlusion/rectangle.obj',
                'to_world': T.translate(translation).rotate(rotation, angle).scale(7.0),
                'face_normals': True,
                'bsdf': bsdf_dict
            }
    scene = mi.load_dict(scenedict)

    return scene


def create_scene_from_xml(xmlpath,
                          resx=512, resy=512,
                          integrator='path', maxdepth=6, reparam_max_depth=2, meshpath=None):
    lines = open(xmlpath, 'r').readlines()
    for idx in range(len(lines)):
        line = lines[idx]
        if 'resx' in lines[idx]:
            lines[idx] = line.replace('resolution_x', str(resx))
        if 'resy' in lines[idx]:
            lines[idx] = line.replace('resolution_y', str(resy))
        if 'integrator' in lines[idx]:
            lines[idx] = line.replace('integrator_type', integrator)
        if 'max_depth' in lines[idx]:
            if integrator == 'direct':
                lines[idx] = ''
            else:
                lines[idx] = line.replace('depth_value', str(maxdepth))
        if 'reparam_max_depth' in lines[idx]:
            if integrator == 'prb_reparam':
                lines[idx] = line.replace('reparam_depth_value', str(reparam_max_depth))
            else:
                lines[idx] = ''
        if meshpath and 'meshpath' in line:
            lines[idx] = line.replace('meshpath', meshpath)

    tmppath = os.path.join(os.path.split(xmlpath)[0], 'tmp.xml')
    open(tmppath, 'w').writelines(lines)
    scene = mi.load_file(tmppath)
    # os.remove(tmppath)
    return scene


def setup_occlusion_scene(hparams):
    scene = create_occlusion_scene(res_x=hparams['resx'], res_y=hparams['resy'], integr=hparams['integrator'],
                                   maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'],
                                   include_walls=True, isglasswall=True)
    params = mi.traverse(scene)
    mat_id = 'sphereHidden.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def setup_coffecup_scene(hparams):
    xmlpath = 'mitsuba_scenes_used/coffee_mug/classic-mug-backdrop-camMoreFrontal.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = 'PLYMesh_1.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def setup_caustic_scene(hparams):
    xmlpath = 'mitsuba_scenes_used/caustic/caustic_sphere_autoReadable.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = 'Cube.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def setup_shadowscene(hparams):
    xmlpath = 'mitsuba_scenes_used/sphere_shadows/sphereshadows.xml'
    print("Running: ", xmlpath, '\n', hparams)
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = 'PLYMesh_1.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def setup_cbox_scene(hparams, inset_only):
    if inset_only:
        xmlpath = 'mitsuba_scenes_used/global_illumination/cbox_aspect_cameraCloser_autoReadable.xml'
    else:
        xmlpath = 'mitsuba_scenes_used/global_illumination/cbox_aspect_autoReadable.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = ['PLYMesh.vertex_positions', 'mat-LeftWallBSDF.brdf_0.reflectance.value', 'PLYMesh_7.vertex_positions']
    initial_vertex_positions = [dr.unravel(mi.Point3f, params[mat_id[0]]),
                                dr.unravel(mi.Point3f, params[mat_id[2]])]
    return scene, params, mat_id, initial_vertex_positions


def setup_highdim_scene(hparams):
    scene = create_highdim_scene(res_x=hparams['resx'], res_y=hparams['resy'],
                                 integr=hparams['integrator'], maxdepth=hparams['max_depth'],
                                 reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    ndim = len([x for x in params.keys() if 'cube' in x and 'vertex_positions' in x])
    ids = ['cube{}.vertex_positions'.format(j) for j in range(ndim)]
    init_vpos = []
    for j in range(ndim):
        init_vpos.append(dr.unravel(mi.Point3f, params[ids[j]]))
    return scene, params, ids, init_vpos


def create_highdim_scene(res_x, res_y, integr='path', maxdepth=6, reparam_max_depth=2):
    if integr == 'path':
        integrator = {'type': 'path', 'max_depth': maxdepth}
    elif 'reparam' in integr:
        integrator = {'type': integr, 'max_depth': maxdepth, 'reparam_max_depth': reparam_max_depth}

    lightpos = [0.0, 8.20, 14.0]
    lightInt = 350e2

    scenedict = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(
                origin=(0, 0, 2),
                target=(0, 0, 0),
                up=(0, 1, 0)
            ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': res_x,
                'height': res_y,
                'rfilter': {'type': 'gaussian'},
                'sample_border': True
            },
        },

        'backdrop': {
            'type': 'ply',
            'filename': 'mitsuba_scenes_used/highdim/Floor.ply',
            'to_world': T.translate([0, -2, 0]).scale(2),
            'bsdf': {'type': 'principled',
                     'base_color': {
                         'type': 'rgb',
                         'value': [0.800000, 0.80950, 0.83509]
                     },
                     'metallic': 0.0,
                     'specular': 0.5,
                     'roughness': 0.5,
                     'spec_tint': 0.0,
                     'anisotropic': 0.0,
                     'sheen': 0.0,
                     'sheen_tint': 0.5,
                     'clearcoat': 0.0,
                     'clearcoat_gloss': 0.173205,
                     'spec_trans': 0.0}
        },

        'light': {
            'type': 'obj',
            'filename': 'mitsuba_scenes_used/occlusion/rectangle.obj',
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [lightInt, lightInt, lightInt]}
            },
            'to_world': T.translate(lightpos).rotate([1, 0, 0], 90).scale(0.15)
        },
        'emitter': {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': (0.3, 0.3, 0.4)}
        }
    }

    num_primitives = 65
    num_red, num_blue, num_orange = 25, 20, 20
    scale, trans_z = 0.1, -8.0
    colors = [(0.9, 0., 0.), (1.0, 0.5, 0.), (0., 0., 0.9)]

    offsets = np.random.uniform(low=-1.0, high=1.0, size=(6, num_primitives)) * 0.75
    for j in range(num_primitives):
        if j < num_red:
            c = colors[0]
            translation = [1.5 - offsets[0, j], 1.5 - offsets[1, j], trans_z]  # red
        elif j < num_red + num_blue:
            c = colors[2]
            translation = [-0.8 - offsets[2, j], -1.5 - offsets[3, j], trans_z]  # blue
        else:
            c = colors[1]
            translation = [-0.8 - offsets[4, j], +1.2 - offsets[5, j], trans_z]  # yellow

        scenedict['{}{}'.format('cube', j)] = {
            'type': 'cube',
            'to_world': T.translate(translation).scale(scale),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': c}}
        }

    scene = mi.load_dict(scenedict)

    return scene


if __name__ == '__main__':

    # some random parameters, adapt as needed 
    hparams = {'resx': 512,
               'resy': 384,
               'integrator': 'prb_reparam',
               'max_depth': 6,
               'reparam_max_depth': 2,
               'render_spp': 64,
               'learning_rate': 0.05}

    # scene, params, ids, initial_vertex_positions = setup_coffecup_scene(hparams=hparams)
    # scene, params, ids, initial_vertex_positions = setup_shadowscene(hparams=hparams)
    # scene, params, ids, initial_vertex_positions = setup_caustic_scene(hparams=hparams)
    # scene, params, ids, initial_vertex_positions = setup_occlusion_scene(hparams=hparams)
    # scene, params, ids, initial_vertex_positions = setup_cbox_scene(hparams=hparams, inset_only=True)
    scene, params, ids, initial_vertex_positions = setup_highdim_scene(hparams=hparams)

    img_reference = mi.render(scene, seed=0, spp=512)
    plt.imshow(img_reference ** .4545)
    plt.show()
