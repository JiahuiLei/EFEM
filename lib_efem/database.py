# Construct database
import sys


sys.path.append("../lib_shape_prior/")

from .misc import cfg_with_default
import os
import os.path as osp
import torch
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .model_utils import CatePrior
from torch.utils.data import DataLoader
from dataset import get_dataset
import time
import logging


def load_database(cfg):
    ret = dict()
    for name in cfg["shape_priors"].keys():
        assert "working_dir" in cfg["shape_priors"][name].keys()
        ret[name] = Database(cfg["shape_priors"][name])
    return ret


class Database:
    def __init__(self, cfg) -> None:
        self.working_dir = cfg["working_dir"]
        self.cache_fn = osp.join(self.working_dir, cfg["database_cache_fn"])

        self.saved_pcl_n = cfg_with_default(cfg, ["database_n_pcl_store"], -1)
        self.saved_pcl_flag = self.saved_pcl_n > 0

        self.k_inv, self.k_so3 = cfg["database_k"]["inv"], cfg["database_k"]["so3"]
        self.k_joint = cfg["database_k"]["joint"]

        try:
            npdata = np.load(self.cache_fn, allow_pickle=True)
            DATA = {}
            for f in npdata.files:
                DATA[f] = npdata[f]
            if self.saved_pcl_flag:
                assert DATA["pcl"][0].shape[0] == self.saved_pcl_n
        except:
            DATA = self.encode(cfg)
            os.makedirs(osp.dirname(self.cache_fn), exist_ok=True)
            np.savez_compressed(self.cache_fn, **DATA)
            npdata = np.load(self.cache_fn, allow_pickle=True)
            DATA = {}
            for f in npdata.files:
                DATA[f] = npdata[f]
        self.DATA = DATA
        self.prepare_database(DATA)
        return

    def __len__(self):
        return self.N

    def prepare_database(self, data):
        self.id = data["id"]
        self.N = len(self.id)
        self.center = data["center"]
        self.scale = data["scale"]
        self.z_so3, self.z_inv = data["z_so3"], data["z_inv"]
        self.z_so3_proj, self.z_so3_basis = data["z_so3_proj"], data["z_so3_basis"]
        if self.saved_pcl_flag:
            self.pcl = data["pcl"]
        try:
            self.basis, self.bbox = data["basis"], data["bbox"]
        except:
            pass

        self.z_inv_knn = NearestNeighbors(n_neighbors=self.k_inv, algorithm="kd_tree")
        self.z_inv_knn.fit(self.z_inv)
        self.z_so3_knn = NearestNeighbors(n_neighbors=self.k_so3, algorithm="kd_tree")
        self.z_so3_knn.fit(self.z_so3_proj.reshape(self.N, -1))
        self.z_joint_knn = NearestNeighbors(n_neighbors=self.k_joint, algorithm="kd_tree")
        self.z_joint_knn.fit(np.concatenate([self.z_so3_proj.reshape(self.N, -1), self.z_inv], -1))

        return

    @torch.no_grad()
    def encode(self, cfg, bs=8, workers=8):
        model = CatePrior(cfg, model_id="dataset building", use_double=False).cuda()
        model_config_fn = cfg["field_cfg"]
        with open(osp.join(cfg["working_dir"], model_config_fn), "r") as f:
            model_cfg = yaml.full_load(f)

        # modify the config
        model_cfg["dataset"]["balanced_class"] = False
        model_cfg["dataset"]["use_augmentation"] = False
        model_cfg["dataset"]["ram_cache"] = False
        model_cfg["dataset"]["input_mode"] = "pcl"
        model_cfg["dataset"]["n_pcl_fewer"] = model_cfg["dataset"]["n_pcl"]
        # # ! warning, fix the std here!!! 2022.9.11
        # ! warning! leave this bug here, from x0.5 exp, seems using the std noisy pcl input to build database get a little better results on naive dataset
        # model_cfg["dataset"]["noise_std"] = 0.0

        if self.saved_pcl_flag:
            model_cfg["dataset"]["n_pcl"] = self.saved_pcl_n
        if cfg_with_default(cfg, ["use_current_cwd"], False):
            model_cfg["root"] = os.getcwd()
        else:
            # model_cfg["root"] = os.path.dirname(os.getcwd())
            model_cfg["root"] = osp.join(osp.dirname(__file__), "../lib_shape_prior/")

        DatasetClass = get_dataset(model_cfg)
        dataset = DatasetClass(model_cfg, mode="train")
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        ret = {
            "id": [],
            "z_so3": [],
            "z_inv": [],
            "center": [],
            "scale": [],
            "z_so3_proj": [],
            "z_so3_basis": [],
            "z_so3_var": [],
            "bbox": [],
            "bbox_c": [],
            "pcl": [],
            "cls": [],
        }
        for batch in tqdm(dataloader):
            input_pcl = batch[0]["inputs_fewer"]
            input_pcl = input_pcl.transpose(2, 1).float().cuda()
            centroid = input_pcl.mean(-1)  # B,3
            input_pcl = input_pcl - centroid[..., None]

            if self.saved_pcl_flag:
                ret["pcl"].append(batch[0]["inputs"])

            bbox_h, bbox_l = input_pcl.max(dim=-1).values, input_pcl.min(dim=-1).values
            bbox = (bbox_h - bbox_l) / 2.0
            bbox_c = (bbox_h + bbox_l) / 2.0
            _ret = model.encoder(input_pcl)
            if len(_ret) == 3:  # no center prediction
                for _i, name in enumerate(["scale", "z_so3", "z_inv"]):
                    ret[name].append(_ret[_i].float().detach().cpu())
                ret["center"].append(centroid.unsqueeze(1).float().detach().cpu())
                z_so3_id = 1
            else:
                for _i, name in enumerate(["center", "scale", "z_so3", "z_inv"]):
                    ret[name].append(_ret[_i].float().detach().cpu())
                z_so3_id = 2
            # compute an rotation invariant characteristic of the vector feature
            z_so3_proj, z_so3_basis, var = self.z_so3_pca(_ret[z_so3_id].float())
            ret["z_so3_proj"].append(z_so3_proj.float().detach().cpu())
            ret["z_so3_basis"].append(z_so3_basis.float().detach().cpu())
            ret["z_so3_var"].append(var.float().detach().cpu())
            ret["cls"].append(batch[1]["cls"].detach().cpu())
            ret["bbox"].append(bbox.float().detach().cpu())
            ret["bbox_c"].append(bbox_c.float().detach().cpu())
            ret["id"] += batch[1]["obj_id"]

        empty_key = []
        for k in ret.keys():
            if len(ret[k]) == 0:
                empty_key.append(k)
                continue
            if isinstance(ret[k][0], torch.Tensor):
                ret[k] = torch.cat(ret[k], 0).numpy()
        for k in empty_key:
            del ret[k]
        return ret

    def z_so3_pca(self, z_so3):
        z_W = (z_so3[..., None] @ z_so3.unsqueeze(2)).sum(1)
        success = False
        while not success:
            try:
                var, z_so3_basis = torch.linalg.eigh(z_W.double())  # B,3,basis
                success = True
            except:
                logging.warning("torch.linalg.eigh fail, retry")
        z_so3_basis = z_so3_basis.float()
        z_so3_proj = torch.einsum("bci,bij->bcj", z_so3, z_so3_basis)
        return z_so3_proj, z_so3_basis, var.float()

    def query(
        self,
        z_so3,
        z_inv,
        use_inv=True,
        use_so3=False,
        use_joint=False,
        device=torch.device("cuda"),
    ):
        B = z_so3.shape[0]
        z_inv = z_inv.detach().cpu().numpy()
        z_so3_proj, z_so3_basis, _ = self.z_so3_pca(z_so3)
        z_so3_proj = z_so3_proj.detach().cpu().numpy()
        z_so3_basis = z_so3_basis.detach().cpu().numpy()
        dist, idx = [], []
        if use_inv:
            knn_inv_dist, knn_inv_idx = self.z_inv_knn.kneighbors(z_inv)
            dist.append(knn_inv_dist)
            idx.append(knn_inv_idx)
        if use_so3:
            knn_so3_dist, knn_so3_idx = self.z_so3_knn.kneighbors(z_so3_proj.reshape(B, -1))
            dist.append(knn_so3_dist)
            idx.append(knn_so3_idx)
        if use_joint:
            joint_query = np.concatenate([z_so3_proj.reshape(B, -1), z_inv], -1)
            knn_joint_dist, knn_joint_idx = self.z_joint_knn.kneighbors(joint_query)
            dist.append(knn_joint_dist)
            idx.append(knn_joint_idx)
        dist, idx = np.concatenate(dist, -1), np.concatenate(idx, -1)
        # prepare neighbors info
        nn_info = {}
        for k, v in self.DATA.items():
            if isinstance(v, np.ndarray) and k not in ["id"]:
                nn_info[k] = torch.from_numpy(v[idx]).to(device).float()
            # elif isinstance(v, list):
            #     nn_info[k] = [v[i] for i in idx]
            else:
                nn_info[k] = v[idx]

        return idx, nn_info, dist

    @property
    def mean_scale(self):
        return self.scale.mean()
