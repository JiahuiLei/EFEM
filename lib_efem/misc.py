# MISC helper functions
import numpy as np
import random
import os
import os.path as osp
import torch
from datetime import datetime
import logging
import platform


CLASSES = (
    "cabinet",  # 3
    "bed",  # 4
    "chair",  # 5
    "sofa",  # 6
    "table",  # 7
    "door",  # 8
    "window",  # 9
    "bookshelf",  # 10
    "picture",  # 11
    "counter",  # 12
    "desk",  # 14
    "curtain",  # 16
    "refrigerator",  # 24
    "shower curtain",  # 28
    "toilet",  # 33
    "sink",  # 34
    "bathtub",  # 36
    "otherfurniture",  # 39
)
NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)


def save_scannet_format(sol, scene_id, dst_dir, scannet_flag=False):
    if scannet_flag:
        semantic_list = CLASSES
        nyu_id = NYU_ID
        store_npy = False
    else:
        semantic_list = list(sol.keys())
        assert len(semantic_list) == 1
        nyu_id = None
        store_npy = True
    # if set fg_mode, all fg sem label will be 1
    save_dir = osp.join(dst_dir, "scannet_format")
    save_mask_dir = osp.join(save_dir, "predicted_masks")
    os.makedirs(save_mask_dir, exist_ok=True)

    f = open(osp.join(save_dir, f"{scene_id}.txt"), "w")
    # sol = np.load(osp.join(src_dir, f"{scene_id}.npz"), allow_pickle=True)
    cnt = 0
    for prior_name in sol.keys():
        if prior_name == "prop_cnt_statistic" or prior_name == "meta_info":
            continue
        for pred in sol[prior_name][1:]:
            label = pred["cate"]

            if nyu_id is None:
                label_id = semantic_list.index(label) + 1
            else:
                # convert to NYU id
                label_id = nyu_id[semantic_list.index(label)]
            score_dict = pred["score"]
            score = score_dict["score_stationary"]

            if "precompute_confidence" in score_dict.keys():
                score = score_dict["precompute_confidence"]
            else:
                coverage_th = 0.4
                if score_dict["score_coverage"] < coverage_th:
                    s = (
                        np.clip(score_dict["score_coverage"], a_min=0.0, a_max=coverage_th)
                        / coverage_th
                    )
                    score *= s

            mask = pred["mask"].numpy().astype(np.uint8)
            rel_fn = f"predicted_masks/{scene_id}_{cnt:03d}.txt"
            f.write(f"{rel_fn} {label_id} {score:.4f}\n")
            mask_path = osp.join(save_dir, rel_fn)
            if store_npy:
                np.save(mask_path + ".npy", mask)
            else:
                np.savetxt(mask_path, mask, fmt="%d")
            cnt += 1
    f.close()
    np.savez(osp.join(dst_dir, "semantic.npz"), class_names=semantic_list, nyu_id=nyu_id)
    return


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_log_dir(path, resume=False):
    if osp.exists(path):
        if resume:
            viz_dir = osp.join(path, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            back_dir = osp.join(path, "backup")
            os.makedirs(back_dir, exist_ok=True)
            return path, viz_dir, back_dir
        else:
            print("warning, old log exists, not resume, move to bck")
            os.makedirs(path + "_bck", exist_ok=True)
            time_stamp = datetime.now().strftime("%H_%M_%S")
            os.system(f"mv {path} {osp.join(path+'_bck', osp.basename(path)+time_stamp)}")
    os.makedirs(path, exist_ok=False)
    viz_dir = osp.join(path, "viz")
    os.makedirs(viz_dir)
    back_dir = osp.join(path, "backup")
    os.makedirs(back_dir)
    return path, viz_dir, back_dir


def cfg_with_default(cfg, key_list, default):
    root = cfg
    for k in key_list:
        if k in root.keys():
            root = root[k]
        else:
            return default
    return root


def count_param(net):
    return sum(param.numel() for param in net.parameters())


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def config_logging(log_dir, debug=False, log_fn="running.log"):
    logging.getLogger().handlers.clear()
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    logger_handler.addFilter(HostnameFilter())
    formatter = logging.Formatter(
        "| %(hostname)s | %(levelname)s | %(asctime)s | %(message)s   [%(filename)s:%(lineno)d]",
        "%b-%d-%H:%M:%S",
    )
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)
    os.makedirs(log_dir, exist_ok=True)
    file_logger_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
    logger_handler.addFilter(HostnameFilter())
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)
    return
