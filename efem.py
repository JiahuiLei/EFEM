# this should be an elegant and almost final version

import os
import os.path as osp
import torch
import yaml
import numpy as np
import shutil
import logging

from lib_efem.data_utils import get_dataset
from lib_efem.misc import (
    create_log_dir,
    cfg_with_default,
    setup_seed,
    config_logging,
    save_scannet_format,
)
from lib_efem.model_utils import load_models_dict
from lib_efem.database import load_database
from lib_efem.solver import Solver
import time

SEED_DEFAULT = 12345


def main(cfg, device, i=0, m=1, SEED=12345):  # i, m are to split the dataset to several jobs
    if SEED != SEED_DEFAULT:
        cfg["log_dir"] = cfg["log_dir"] + f"_seed={SEED}"

    log_resume_flag = cfg_with_default(cfg, ["log_resume"], False)

    # create log
    log_dir, viz_dir, bck_dir = create_log_dir(
        osp.join(cfg["working_dir"], cfg["log_dir"]), resume=log_resume_flag
    )
    output_dir = osp.join(log_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # load finished
    if log_resume_flag:
        finished_scene_id = [d[:-4] for d in os.listdir(output_dir) if d.endswith(".npz")]
    else:
        finished_scene_id = []

    config_logging(osp.join(log_dir, "logs"), debug=False, log_fn="init.log")
    os.system(f"cp {__file__} {bck_dir}")
    os.system(f"cp -r lib_efem configs {bck_dir}")

    logging.info(f"Use seed {SEED}")

    # Load Model
    model = load_models_dict(cfg, device)

    dataset = get_dataset(cfg)

    solver = Solver(cfg)

    # build database if necessary
    if cfg_with_default(cfg, ["use_database"], True):
        database_dict = load_database(cfg)
    else:
        database_dict = None

    iter = cfg_with_default(cfg, ["iter"], range(len(dataset)))
    time_list = []
    for scene_id in iter:
        if scene_id % m != i:
            continue  # other job will handle this

        if dataset.scene_id_list[scene_id] in finished_scene_id:
            continue

        config_logging(osp.join(log_dir, "logs"), debug=False, log_fn=f"{scene_id}.log")
        logging.info("=" * max(shutil.get_terminal_size()[0] - 100, 30))
        logging.info(f"scene_id {scene_id}")
        logging.info("=" * max(shutil.get_terminal_size()[0] - 100, 30))
        data_dict = dataset[scene_id]
        data_dict = dataset.to_device(data_dict, device)

        # solve
        start_time = time.time()
        ppt_dict, prop_cnt_statistic = solver.solve(
            model_dict=model,
            data_dict=data_dict,
            database_dict=database_dict,
            viz_prefix=f"{scene_id}_s{data_dict['scene']}",
            viz_dir=viz_dir,
            seed=SEED,
        )
        solver_time = time.time() - start_time
        time_list.append(solver_time)
        logging.info(f"Solver takes {solver_time:.3f}s")

        # save solution
        solution = {}
        for k, v in ppt_dict.items():
            solution[k] = v.fetch_output()
        np.savez_compressed(
            osp.join(output_dir, f"{data_dict['scene']}.npz"),
            **solution,
            meta_info={"prop_cnt_statistic": prop_cnt_statistic, "solver_time": solver_time},
        )

        # save to scannet format
        save_scannet_format(
            solution,
            data_dict['scene'],
            dst_dir=output_dir + "_eval",
            scannet_flag=cfg_with_default(cfg, ["scannet_flag"], False),
        )

    time_list = np.array(time_list)
    average_time = time_list.mean()
    np.save(osp.join(log_dir, "time.npy"), time_list)
    np.savetxt(osp.join(log_dir, "ave_time.txt"), [average_time])
    logging.info(f"ave time = {average_time}")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Run")
    arg_parser.add_argument("--config", "-c", required=True)
    arg_parser.add_argument("-i", type=int, default=0)
    arg_parser.add_argument("-m", type=int, default=1)
    arg_parser.add_argument("-s", type=int, default=SEED_DEFAULT)
    args = arg_parser.parse_args()
    seed = args.s
    setup_seed(seed)

    # * Note: All the path is relative to this files dir
    cwd = osp.dirname(__file__)
    with open(osp.join(cwd, args.config), "r") as f:
        cfg = yaml.full_load(f)
    cfg["working_dir"] = cwd
    main(cfg, device=torch.device("cuda"), i=args.i, m=args.m, SEED=seed)
