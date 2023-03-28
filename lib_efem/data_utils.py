import os
import os.path as osp
import torch
import numpy as np
import torch.utils.data as data
import trimesh
from matplotlib import cm
from .misc import cfg_with_default
from copy import deepcopy
import json


def get_dataset(cfg):
    cfg["dataset"]["working_dir"] = cfg["working_dir"]
    dataset_mode = cfg_with_default(cfg, ["dataset_mode"], "sapien_processed")
    if dataset_mode == "sapien_processed":
        dataset = SapienProcessedDataset(cfg["dataset"])
    elif dataset_mode == "scannet_processed":
        dataset = ScannetProcessedDataset(cfg["dataset"])
    else:
        raise NotImplementedError()
    return dataset


class SapienProcessedDataset(data.Dataset):
    # Now only load pts, later can add lables, colors and normals
    def __init__(self, cfg) -> None:
        super().__init__()
        self.phase = cfg["phase"]
        self.data_root = osp.join(cfg["data_dir"], self.phase)
        self.postfix = cfg["postfix"]
        self.scene_id_list = [
            d[: -len(self.postfix)] for d in os.listdir(self.data_root) if d.endswith(self.postfix)
        ]
        self.scene_id_list.sort()
        semantic = np.load(osp.join(cfg["data_dir"], "semantic.npz"), allow_pickle=True)
        self.label2cate = semantic["LABEL2CATE"].tolist()
        self.cate2label = semantic["CATE2LABEL"].tolist()
        self.invalid_label = int(semantic["INVALID_LABEL"])
        return

    def __len__(self):
        return len(self.scene_id_list)

    def __getitem__(self, index):
        scene_id = self.scene_id_list[index]

        # fn = osp.join(self.data_root, scene_id, "mesh_repeat.npz")
        data_fn = osp.join(self.data_root, f"{scene_id}{self.postfix}")
        pts, _, sem, ins = torch.load(data_fn)
        # ! ins has -100 as bg, others starting from 1; but in enff old code, the ins start id is 0
        ins[sem < 0] = self.invalid_label  # remove invalid semantic as bg as well
        label = ins
        # ! also need mesh
        mesh_fn = osp.join(self.data_root, f"{scene_id}.obj")
        mesh = trimesh.load(mesh_fn, process=False)
        nrm = np.asarray(mesh.vertex_normals)
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 127.5 - 1
        # ! warning, here is only one category, in the future should change this!!!
        cate_list = []
        for ins_id in np.unique(label):
            if ins_id == self.invalid_label:
                continue
            sem_id = np.unique(sem[ins == ins_id])
            assert len(sem_id) == 1
            cate_list.append(self.label2cate[int(sem_id)])

        valid_mask = np.ones_like(pts[:, -1]) > 0  # all valid
        pts = pts[valid_mask]
        colors = colors[valid_mask]
        nrm = nrm[valid_mask]
        label = label[valid_mask]

        pts = pts[:, [0, 2, 1]]
        nrm = nrm[:, [0, 2, 1]]
        pts[:, 2] *= -1.0  # make right hand coordinate
        nrm[:, 2] *= -1.0  # make right hand coordinate

        # also load the mesh
        vtx = np.asarray(mesh.vertices)
        vtx = vtx - vtx.mean(0)[None, :]
        vtx = vtx[:, [0, 2, 1]]
        vtx[:, 2] *= -1.0  # make right hand coordinate
        new_scene_mesh = trimesh.Trimesh(
            vertices=vtx, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors
        )

        # # debug
        # np.savetxt("../debug/scannet_pts.txt", pts)
        ret = {
            "pointcloud": torch.from_numpy(pts).float(),
            "normals": torch.from_numpy(nrm).float(),
            "color": torch.from_numpy(colors).float(),
            "scene": scene_id,
            "view": "0-3",
            "mesh": new_scene_mesh,
            "label": label,
            "cate_list": cate_list,
        }
        return ret

    def to_device(self, data_dict, device):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(device)
        return data_dict


class ScannetProcessedDataset(data.Dataset):
    # Now only load pts, later can add lables, colors and normals
    def __init__(self, cfg) -> None:
        super().__init__()
        self.phase = cfg["phase"]
        self.data_root = osp.join(cfg["data_dir"], self.phase)
        self.postfix = cfg["postfix"]
        self.scene_id_list = [
            d[: -len(self.postfix)] for d in os.listdir(self.data_root) if d.endswith(self.postfix)
        ]
        self.scene_id_list.sort()

        self.ins_18_classes = [
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "shower curtain",
            "toilet",
            "sink",
            "bathtub",
            "otherfurniture",
        ]
        self.sem_20_classes = ["wall", "floor"] + deepcopy(self.ins_18_classes)

        self.NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
        # self.label2cate = {k: v for k, v in zip(NYU_ID, CLASSES)}
        # self.cate2label = {k: v for k, v in zip(CLASSES, NYU_ID)}
        self.invalid_label = -100

        self.sem_pred_dir = cfg_with_default(cfg, ["sem_dir"], None)
        self.use_sem_pred = False
        if self.sem_pred_dir is not None:
            self.sem_pred_dir = osp.join(self.sem_pred_dir, self.phase)
            self.use_sem_pred = True
        return

    def __len__(self):
        return len(self.scene_id_list)

    def __getitem__(self, index):
        scene_id = self.scene_id_list[index]
        data_fn = osp.join(self.data_root, f"{scene_id}{self.postfix}")
        data = torch.load(data_fn)[:2]
        pts, colors = data
        mesh_fn = osp.join(self.data_root, f"{scene_id}_vh_clean_2.ply")
        mesh = trimesh.load(mesh_fn, process=False)
        nrm = np.asarray(mesh.vertex_normals)

        # seg_fn = osp.join(self.data_root, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        # with open(seg_fn) as f:
        #     per_point_segment_ids = json.load(f)
        # per_point_segment_ids = np.asarray(per_point_segment_ids["segIndices"], dtype="int32")
        # uni_seg_ids = np.unique(per_point_segment_ids)
        # new_seg_ids = np.arange(len(uni_seg_ids))
        # super_seg = -np.ones_like(per_point_segment_ids)
        # super_seg_vtx_color = np.zeros((len(super_seg), 3))
        # seg_viz_color = cm.jet(
        #     new_seg_ids[np.random.permutation(len(new_seg_ids))] / (len(uni_seg_ids) + 5)
        # )[:, :3]
        # for old_id, new_id, viz_color in zip(uni_seg_ids, new_seg_ids, seg_viz_color):
        #     seg_mask = per_point_segment_ids == old_id
        #     super_seg_vtx_color[seg_mask] = viz_color
        #     super_seg[seg_mask] = new_id

        valid_mask = np.ones_like(pts[:, -1]) > 0
        pts = pts[valid_mask]
        colors = colors[valid_mask]
        nrm = nrm[valid_mask]
        # label = label[valid_mask]

        pts = pts[:, [0, 2, 1]]
        nrm = nrm[:, [0, 2, 1]]
        pts[:, 2] *= -1.0  # make right hand coordinate
        nrm[:, 2] *= -1.0  # make right hand coordinate

        # also load the mesh
        vtx = np.asarray(mesh.vertices)
        vtx = vtx - vtx.mean(0)[None, :]
        vtx = vtx[:, [0, 2, 1]]
        vtx[:, 2] *= -1.0  # make right hand coordinate
        new_scene_mesh = trimesh.Trimesh(
            vertices=vtx, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors
        )
        # viz_overseg_mesh = trimesh.Trimesh(
        #     vertices=vtx,
        #     faces=mesh.faces,
        #     vertex_colors=(super_seg_vtx_color * 255).astype(np.uint8),
        # )

        # # debug
        # np.savetxt("../debug/scannet_pts.txt", pts)
        ret = {
            "pointcloud": torch.from_numpy(pts).float(),
            "normals": torch.from_numpy(nrm).float(),
            # "seg": torch.from_numpy(super_seg).int(),
            # "seg_uni_id": torch.from_numpy(new_seg_ids).int(),
            "color": torch.from_numpy(colors).float(),
            "scene": scene_id,
            "view": "0-3",
            "mesh": new_scene_mesh,
            # "seg_mesh": viz_overseg_mesh,
        }

        if self.use_sem_pred:
            sem_fn = osp.join(self.sem_pred_dir, f"{scene_id}.npy")
            sem_pred = np.load(sem_fn)
            sem_pred_dict = {}
            for sem_id in np.unique(sem_pred):
                sem_pred_dict[self.sem_20_classes[sem_id]] = sem_pred == sem_id
            ret["sem_pred"] = sem_pred_dict

        return ret

    def to_device(self, data_dict, device):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(device)
        return data_dict
