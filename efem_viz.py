# new centroid version

import sys

sys.path.append("..")

import os
import os.path as osp
import torch
import yaml
from tqdm import tqdm
import imageio

from lib_efem.data_utils import get_dataset
from lib_efem.misc import create_log_dir, cfg_with_default, setup_seed
from lib_efem.model_utils import load_models_dict
from lib_efem.database import load_database
from lib_efem.viz_utils import viz_scene_jointly

from lib_efem.solver import Solver
import shutil
import logging
from lib_efem.misc import config_logging

import numpy as np
import pandas as pd


def viz_solution_th(sol, th, data_dict, solver, output_path=None):
    accept_object_dict = {}
    for prior_key in sol.keys():
        if prior_key == "prop_cnt_statistic" or prior_key == "meta_info":
            continue
        l = sol[prior_key][1:]
        for it in l:
            sem = it["cate"]
            try:
                confidence = it["score"]["precompute_confidence"]
            except:
                confidence = it["score"]["score_stationary"]
            if confidence < th:
                logging.info(f"conf {confidence}<{th}, skipped")
                continue
            if sem not in accept_object_dict.keys():
                accept_object_dict[sem] = [it]
            else:
                accept_object_dict[sem].append(it)
    accept_object_list = [v for v in accept_object_dict.values()]
    full_pcl = data_dict["pointcloud"].detach().cpu().numpy()
    ret = {}
    for mode in solver.scene_viz_args["viz_modes"]:
        rgb_pcl, rgb_msh = viz_scene_jointly(
            background_pcl=full_pcl,
            group1_object_list=accept_object_list,
            group2_object_list=[],
            mode=mode,
            **solver.scene_viz_args,
        )
        if rgb_pcl is None:
            continue
        if output_path is not None:
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            output_fn1 = output_path + f"_{mode}_pcl.png"
            output_fn2 = output_path + f"_{mode}_msh.png"
            imageio.imsave(output_fn1, rgb_pcl)
            imageio.imsave(output_fn2, rgb_msh)
        ret[mode] = (rgb_pcl, rgb_msh)
    return ret


def main(cfg, device, th):
    # create log
    log_dir = osp.join(cfg["working_dir"], cfg["log_dir"])
    viz_dir = osp.join(log_dir, f"viz_confth={th:.3f}")
    results_dir = osp.join(log_dir, "results")
    os.makedirs(viz_dir, exist_ok=True)
    assert osp.exists(results_dir)
    config_logging(osp.join(log_dir, "logs"), debug=False, log_fn="start.log")
    # Load One data example
    dataset = get_dataset(cfg)
    solver = Solver(cfg)
    solver.viz_flag = True

    results = [fn[:-4] for fn in os.listdir(results_dir) if fn.endswith(".npz")]

    for scene_id in tqdm(results):
        if scene_id not in dataset.scene_id_list:
            continue
        ind = dataset.scene_id_list.index(scene_id)
        data_dict = dataset[ind]
        data_dict = dataset.to_device(data_dict, device)
        # render

        viz_prefix = f"{ind}_{scene_id}"
        solver.viz_input(data_dict, viz_dir, viz_prefix=viz_prefix)
        _sol = np.load(osp.join(results_dir, f"{scene_id}.npz"), allow_pickle=True)

        sol = {}
        for f in _sol.files:
            sol[f] = _sol[f].tolist()
        viz_solution_th(
            sol,
            th,
            data_dict,
            solver,
            output_path=osp.join(viz_dir, f"{ind}_{scene_id}_s{data_dict['scene']}_output"),
        )


if __name__ == "__main__":
    import argparse

    setup_seed(12345)

    arg_parser = argparse.ArgumentParser(description="Run")
    arg_parser.add_argument("--config", "-c", required=True)
    arg_parser.add_argument("--confidence_th", "-t", type=float, default=0.0)
    args = arg_parser.parse_args()

    # * Note: All the path is relative to this files dir
    cwd = osp.dirname(__file__)
    with open(osp.join(cwd, args.config), "r") as f:
        cfg = yaml.full_load(f)
    cfg["working_dir"] = cwd
    main(cfg, device=torch.device("cuda"), th=args.confidence_th)
