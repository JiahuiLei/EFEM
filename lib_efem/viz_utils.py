# debug code

import sys, os
import os.path as osp

from .render_helper import render
import trimesh
from colorsys import hsv_to_rgb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import logging
import os
import imageio
import numpy as np


def viz_scene_jointly(
    background_pcl,
    group1_object_list,
    group2_object_list,
    shape=(640, 640),
    group1_pcl_alpha=0.8,
    group1_mesh_alpha=1.0,
    group2_pcl_alpha=0.2,
    group2_mesh_alpha=0.3,
    background_pts_r=0.001,
    pts_r=0.005,
    corner_r=0.008,
    mode="ins",
    mesh_flag=True,
    **kwargs,
):
    assert mode in ["ins", "sem"]
    cnt = 0
    for g in [group1_object_list, group2_object_list]:
        for l in g:
            cnt += len(l)
    if cnt == 0:
        return None, None

    bbox_edge_start_ind = [0, 0, 2, 1, 3, 3, 5, 6, 0, 1, 4, 2]
    bbox_edge_end_ind = [1, 2, 4, 4, 6, 5, 7, 7, 3, 6, 7, 5]

    pcl_list, pcl_no_obj_list = [background_pcl], [background_pcl]
    pcl_radius_list, pcl_no_obj_radius_list = [background_pts_r], [background_pts_r]
    pcl_color_list, pcl_no_obj_color_list = [[0.8, 0.8, 0.8, 1.0]], [[0.8, 0.8, 0.8, 1.0]]
    line_list, line_color_list = [], []
    arrow_colors, arrow_start, arrow_directions = [], [], []
    mesh_list, mesh_colors_list = [], []

    # * Two group of objects
    for obj_list, pcl_alpha, mesh_alpha in zip(
        [group1_object_list, group2_object_list],
        [group1_pcl_alpha, group2_pcl_alpha],
        [group1_mesh_alpha, group2_mesh_alpha],
    ):
        cat_obj_list = []
        for l in obj_list:
            cat_obj_list += l

        if mode == "ins":
            pcl_colors = [
                list(hsv_to_rgb(h, 0.7, 0.7)) + [pcl_alpha]
                for h in np.linspace(start=0.0, stop=1.0, num=len(cat_obj_list) + 1)[:-1]
            ]
            box_colors = [
                list(hsv_to_rgb(h, 1.0, 1.0)) + [1.0]
                for h in np.linspace(start=0.0, stop=1.0, num=len(cat_obj_list) + 1)[:-1]
            ]
            mesh_colors = [
                list(hsv_to_rgb(h, 0.8, 0.9)) + [mesh_alpha]
                for h in np.linspace(start=0.0, stop=1.0, num=len(cat_obj_list) + 1)[:-1]
            ]
        else:
            N_cate = max(len(group1_object_list), len(group2_object_list))
            hue_list = np.linspace(start=0.0, stop=1.0, num=N_cate + 1)[:-1]
            pcl_colors, box_colors, mesh_colors = [], [], []
            for l, h in zip(obj_list, hue_list):
                pcl_colors += [list(hsv_to_rgb(h, 0.7, 0.7)) + [pcl_alpha]] * len(l)
                box_colors += [list(hsv_to_rgb(h, 1.0, 1.0)) + [1.0]] * len(l)
                mesh_colors += [list(hsv_to_rgb(h, 0.8, 0.9)) + [mesh_alpha]] * len(l)

        for i in range(len(cat_obj_list)):
            pred = cat_obj_list[i]
            pcl_list += [pred["pcl"]]
            pcl_no_obj_list += [pred["bbox_8pts"]]
            pcl_radius_list += [pts_r]
            pcl_no_obj_radius_list += [corner_r]
            pcl_color_list += [pcl_colors[i]]
            pcl_no_obj_color_list += [box_colors[i]]

            viz_bbox_corner = pred["bbox_8pts"]
            bbox_start = viz_bbox_corner[bbox_edge_start_ind]
            bbox_end = viz_bbox_corner[bbox_edge_end_ind]
            line_list += [(bbox_start, bbox_end)]
            line_color_list += [box_colors[i]]

            basis, bbox_3vale = pred["B"], pred["bbox"]
            _arrow_start = np.tile(pred["t"].squeeze(0), (3, 1))
            _arrow_directions = basis * abs(bbox_3vale)[:, None]
            _arrow_colors = np.ones((3, 4))
            _arrow_colors[:3, :3] = np.eye(3)
            arrow_start.append(_arrow_start)
            arrow_directions.append(_arrow_directions)
            arrow_colors.append(_arrow_colors)
            mesh_list.append(pred["mesh_world"])
            mesh_colors_list.append(mesh_colors[i])

    arrow_start = np.concatenate(arrow_start, 0)
    arrow_directions = np.concatenate(arrow_directions, 0)
    arrow_colors = np.concatenate(arrow_colors, 0)

    if not mesh_flag:
        # the output is captured by pcl, for UDF applications
        pcl_no_obj_list = pcl_no_obj_list + mesh_list
        pcl_no_obj_radius_list = pcl_no_obj_radius_list + [pts_r] * len(mesh_list)
        pcl_no_obj_color_list = pcl_no_obj_color_list + mesh_colors
        mesh_list, mesh_colors = [], []
    rgb1 = render(
        # pcl
        pcl_list=pcl_list,
        pcl_radius_list=pcl_radius_list,
        pcl_color_list=pcl_color_list,
        shape=shape,
        **kwargs,
    )

    rgb2 = render(
        # mesh
        mesh_list=mesh_list,
        mesh_color_list=mesh_colors_list,
        # pcl
        pcl_list=pcl_no_obj_list,
        pcl_radius_list=pcl_no_obj_radius_list,
        pcl_color_list=pcl_no_obj_color_list,
        # bbox
        line_list=line_list,
        lines_color_list=line_color_list,
        # pose
        arrow_tuples=(arrow_start, arrow_directions),
        arrow_colors=arrow_colors,
        arrow_radius=pts_r,
        shape=shape,
        **kwargs,
    )

    return rgb1, rgb2


def viz_single_proposal(
    W,
    full_pcl,
    id_list,
    viz_dir,
    step,
    # optional viz
    center=None,
    est_sRt_pcl=None,
    est_sRt_pcl_radius=0.005,
    est_shape_pcl=None,
    est_shape_pcl_radius=0.01,
    bbox_8pts=None,
    bbox_3vale=None,
    basis=None,
    mesh_list=None,
    cam_dist=0.5,
    cam_dist_can=1.5,
    cam_angle_pitch=np.pi / 6.0,
    cam_angle_yaw=0.0,
    pts_r=0.005,
    corner_r=0.008,
    fn_prefix="",
    fn_postfix="",
    centered_shape_pcl=False,
    viz_detailed_recon=True,
    scene_viz_cfg={},
):
    logging.info("vizing proposals ...")
    os.makedirs(viz_dir, exist_ok=True)
    if mesh_list is not None:
        mesh_flag = isinstance(mesh_list[0], trimesh.Trimesh)
    for i in tqdm(range(len(W))):
        w = W[i]
        active_mask = w > 0
        viz_w = w[active_mask].detach().cpu()
        active_pcl = full_pcl[active_mask].detach().cpu()
        if center is not None:
            viz_center = center[i].squeeze(1).squeeze(0).detach().cpu()
        else:
            viz_center = active_pcl.mean(0)
        viz_full_pcl = active_pcl - viz_center[None, ...]
        if mesh_list is not None:
            if mesh_flag:
                viz_mesh = deepcopy(mesh_list[i])
                viz_mesh.apply_translation(-viz_center)
            else:
                viz_mesh = mesh_list[i] - viz_center[None, :].detach().cpu().numpy()
        if est_sRt_pcl is not None:
            viz_sRt_input = est_sRt_pcl[i].detach().cpu() - viz_center
        if est_shape_pcl is not None:
            viz_shape_input = est_shape_pcl[i].detach().cpu()
            if centered_shape_pcl:
                viz_shape_input = viz_shape_input - viz_center

        if bbox_8pts is not None:
            viz_bbox_corner = bbox_8pts[i].detach().cpu() - viz_center
            r"""
            The bbox corners orders should be like this
            [
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ]
            """
            bbox_edge_start_ind = [0, 0, 2, 1, 3, 3, 5, 6, 0, 1, 4, 2]
            bbox_edge_end_ind = [1, 2, 4, 4, 6, 5, 7, 7, 3, 6, 7, 5]
            bbox_start = viz_bbox_corner[bbox_edge_start_ind]
            bbox_end = viz_bbox_corner[bbox_edge_end_ind]

            arrow_start = np.zeros((3, 3))
            arrow_directions = (basis[i].detach() * abs(bbox_3vale[i])[:, None]).cpu().numpy()
            arrow_colors = np.ones((3, 4))
            arrow_colors[:3, :3] = np.eye(3)

        # first make a full scene viz
        viz_full_scene_proposal = render(
            pcl_list=[full_pcl.detach().cpu()],
            pcl_radius_list=[-1.0],
            pcl_color_list=[w.detach().cpu()],
            **scene_viz_cfg,
        )
        imageio.imsave(
            osp.join(
                viz_dir,
                f"{fn_prefix}_full_scene_prop{id_list[i]}_iter{step}_{fn_postfix}.png",
            ),
            viz_full_scene_proposal,
        )
        if not viz_detailed_recon:
            continue

        viz_list = []
        viz_pcl_value = render(
            pcl_list=[viz_full_pcl],
            pcl_color_list=[viz_w],
            pcl_radius_list=[-1.0],
            cam_angle_pitch=cam_angle_pitch,
            cam_angle_yaw=cam_angle_yaw,
            cam_dist=cam_dist,
        )
        viz_list.append(viz_pcl_value)
        imageio.imsave(
            osp.join(
                viz_dir,
                fn_prefix + f"_single_viz_value_prop{id_list[i]}_iter{step}_" + fn_postfix + ".png",
            ),
            viz_pcl_value,
        )

        if bbox_8pts is not None:
            viz_det = render(
                pcl_list=[viz_full_pcl, viz_bbox_corner],
                pcl_radius_list=[pts_r, corner_r],
                pcl_color_list=[[0.9, 0.9, 0.9, 0.3], [0.2, 0.9, 0.8, 1.0]],
                line_list=[(bbox_start, bbox_end)],
                lines_color_list=[[0.2, 0.9, 0.8, 1.0]],
                arrow_tuples=(arrow_start, arrow_directions),
                arrow_colors=arrow_colors,
                arrow_radius=0.005,
                cam_angle_pitch=cam_angle_pitch,
                cam_angle_yaw=cam_angle_yaw,
                cam_dist=cam_dist,
            )
            viz_list.append(viz_det)

        if est_sRt_pcl is not None:
            viz_sRt_input = render(
                pcl_list=[viz_sRt_input],
                pcl_color_list=[[0.5, 0.5, 1.0]],
                pcl_radius_list=[est_sRt_pcl_radius],
                cam_angle_pitch=cam_angle_pitch,
                cam_angle_yaw=cam_angle_yaw,
                cam_dist=cam_dist,
            )
            viz_list.append(viz_sRt_input)

        if est_shape_pcl is not None:
            viz_shape_input = render(
                pcl_list=[viz_shape_input],
                pcl_color_list=[[255, 183, 44]],
                pcl_radius_list=[est_shape_pcl_radius],
                cam_angle_pitch=cam_angle_pitch,
                cam_angle_yaw=cam_angle_yaw,
                cam_dist=cam_dist_can,
            )
            viz_list.append(viz_shape_input)
            imageio.imsave(
                osp.join(
                    viz_dir,
                    fn_prefix
                    + f"_single_viz_pcl_prop{id_list[i]}_iter{step}_"
                    + fn_postfix
                    + ".png",
                ),
                viz_shape_input,
            )

        if mesh_list is not None:
            if mesh_flag:
                viz_recon = render(
                    mesh_list=[viz_mesh],
                    mesh_color_list=[[65/255.0, 163/255.0, 189/255.0, 1.0]], #[[0.7, 0.7, 0.7, 1.0]],
                    cam_angle_pitch=cam_angle_pitch,
                    cam_angle_yaw=cam_angle_yaw,
                    cam_dist=cam_dist,
                    light_intensity=4.0,
                )
                imageio.imsave(
                    osp.join(
                        viz_dir,
                        fn_prefix
                        + f"_single_viz_mesh_prop{id_list[i]}_iter{step}_"
                        + fn_postfix
                        + ".png",
                    ),
                    viz_recon,
                )
                viz_joint = render(
                    mesh_list=[viz_mesh],
                    mesh_color_list=[[65/255.0, 163/255.0, 189/255.0, 0.7]], #[[0.7, 0.7, 0.7, 0.7]],
                    pcl_list=[viz_full_pcl],
                    pcl_color_list=[viz_w],
                    pcl_radius_list=[-1.0],
                    cam_angle_pitch=cam_angle_pitch,
                    cam_angle_yaw=cam_angle_yaw,
                    cam_dist=cam_dist,
                    light_intensity=4.0,
                )
                imageio.imsave(
                    osp.join(
                        viz_dir,
                        fn_prefix
                        + f"_single_viz_shape_prop{id_list[i]}_iter{step}_"
                        + fn_postfix
                        + ".png",
                    ),
                    viz_joint,
                )
            else:
                # viz_recon = render(
                #     pcl_list=[viz_mesh],
                #     pcl_color_list=[[1.0, 1.0, 1.0, 0.8]],
                #     pcl_radius_list=[pts_r],
                #     cam_angle_pitch=cam_angle_pitch,
                #     cam_angle_yaw=cam_angle_yaw,
                #     cam_dist=cam_dist,
                #     light_intensity=4.0,
                # )
                viz_joint = render(
                    pcl_list=[viz_full_pcl, viz_mesh],
                    pcl_color_list=[viz_w, [1.0, 0.2, 0.2, 0.7]],
                    pcl_radius_list=[-1.0, pts_r],
                    cam_angle_pitch=cam_angle_pitch,
                    cam_angle_yaw=cam_angle_yaw,
                    cam_dist=cam_dist,
                    light_intensity=4.0,
                )
            # viz_list.append(viz_recon)
            viz_list.append(viz_joint)

        viz_cat = np.concatenate(viz_list, 1)

        imageio.imsave(
            osp.join(viz_dir, fn_prefix + f"_prop{id_list[i]}_iter{step}_" + fn_postfix + ".png"),
            viz_cat,
        )


def viz_database(
    id_list,
    viz_dir,
    step,
    layout,
    fn_prefix="",
    fn_postfix="",
    pred_pcl=None,  # B,N,3
    knn_pcl=None,  # B,K,N,3
    pcl_radius=0.004,
    pcl_viz_sample_n=512,
    cam_dist=0.5,
    cam_angle_pitch=np.pi / 6.0,
    cam_angle_yaw=0.0,
    best_fit_id=None,
    knn_basis=None,
    pred_basis=None,
    pred_bbox=None,
    pred_bbox_center=None,
    bar=False,
):
    # logging.info("vizing database ...")
    B, K, _, _ = knn_pcl.shape
    assert layout[0] * layout[1] >= K
    os.makedirs(viz_dir, exist_ok=True)
    ret_list = []
    it = tqdm(range(B)) if bar else range(B)
    for bid in it:
        _pcl = pred_pcl[bid]
        choice = np.random.choice(pcl_viz_sample_n, len(_pcl), replace=True)
        query_rgb = render(
            pcl_list=[_pcl[choice]],
            pcl_color_list=[[0.2, 0.2, 0.9]],
            pcl_radius_list=[pcl_radius],
            cam_angle_pitch=cam_angle_pitch,
            cam_angle_yaw=cam_angle_yaw,
            cam_dist=cam_dist,
        )
        viz_list = []
        for k in range(K):
            _pcl = knn_pcl[bid][k]
            choice = np.random.choice(pcl_viz_sample_n, len(_pcl), replace=True)
            if knn_basis is not None:
                arrow_start = np.zeros((3, 3))
                arrow_directions = knn_basis[bid][k]  # numbasis, 3
                arrow_colors = np.ones((3, 4))
                arrow_colors[:3, :3] = np.eye(3)
                arrow_len = abs(_pcl @ (knn_basis[bid][k].T)).max(0)
                arrow_directions = arrow_directions * arrow_len[:, None]
            else:
                arrow_start, arrow_directions, arrow_colors = None, None, None
            rgb = render(
                pcl_list=[_pcl[choice]],
                pcl_color_list=[[0.9, 0.2, 0.2]],
                pcl_radius_list=[pcl_radius],
                cam_angle_pitch=cam_angle_pitch,
                cam_angle_yaw=cam_angle_yaw,
                cam_dist=cam_dist,
                arrow_tuples=(arrow_start, arrow_directions),
                arrow_colors=arrow_colors,
                arrow_radius=0.005,
            )
            viz_list.append(rgb)
        # make a joint viz
        L = rgb.shape[0]
        canvas = np.ones((L * layout[0], L * layout[1] + L, 3)).astype(np.uint8) * 255
        canvas[:L, -L:] = query_rgb
        if best_fit_id is not None:
            canvas[L : 2 * L, -L:] = render(
                pcl_list=[knn_pcl[bid][best_fit_id[bid]][choice]],
                pcl_color_list=[[0.2, 0.9, 0.2]],
                pcl_radius_list=[pcl_radius],
                cam_angle_pitch=cam_angle_pitch,
                cam_angle_yaw=cam_angle_yaw,
                cam_dist=cam_dist,
            )
        if pred_basis is not None:
            assert pred_bbox is not None
            if pred_bbox_center is not None:
                arrow_start = np.tile(pred_bbox_center[bid], (3, 1))
            else:
                arrow_start = np.zeros((3, 3))
            arrow_directions = pred_basis[bid]  # numbasis, 3
            arrow_colors = np.ones((3, 4))
            arrow_colors[:3, :3] = np.eye(3)
            arrow_directions = arrow_directions * (pred_bbox[bid][:, None] + 1e-6)

            bbox8pts_weight = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                ]
            )
            bbox_pts_coordi = bbox8pts_weight * pred_bbox[bid][None, :]  # 8,3
            bbox_pts = bbox_pts_coordi @ pred_basis[bid] + arrow_start[:1]
            bbox_edge_start_ind = [0, 0, 2, 1, 3, 3, 5, 6, 0, 1, 4, 2]
            bbox_edge_end_ind = [1, 2, 4, 4, 6, 5, 7, 7, 3, 6, 7, 5]
            bbox_start = bbox_pts[bbox_edge_start_ind]
            bbox_end = bbox_pts[bbox_edge_end_ind]

            canvas[2 * L : 3 * L, -L:] = render(
                pcl_list=[pred_pcl[bid], bbox_pts],
                pcl_color_list=[[0.2, 0.2, 0.9], [0.05, 0.05, 1.0]],
                pcl_radius_list=[-1.0, pcl_radius * 2],
                line_list=[(bbox_start, bbox_end)],
                lines_color_list=[[0.2, 0.9, 0.8, 1.0]],
                cam_angle_pitch=cam_angle_pitch,
                cam_angle_yaw=cam_angle_yaw,
                cam_dist=cam_dist,
                arrow_tuples=(arrow_start, arrow_directions),
                arrow_colors=arrow_colors,
                arrow_radius=0.005,
            )

        for i, rgb in enumerate(viz_list):
            col = i % layout[1]
            row = int((i - col) / layout[1])
            canvas[row * L : row * L + L, col * L : col * L + L] = rgb
            canvas[row * L : row * L + L, col * L] = 0
            canvas[row * L : row * L + L, col * L + L - 1] = 0
            canvas[row * L, col * L : col * L + L] = 0
            canvas[row * L + L - 1, col * L : col * L + L] = 0
        ret_list.append(canvas)
        imageio.imsave(
            osp.join(viz_dir, fn_prefix + f"prop{id_list[bid]}_iter{step}" + fn_postfix + ".png"),
            canvas,
        )
    return ret_list
