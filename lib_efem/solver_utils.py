# move the some func from the solver class here]
import os
import os.path as osp
import torch
import imageio
from torch import nn
import numpy as np
from tqdm import tqdm
import trimesh
from .viz_utils import render
from .database import Database
from .proposalpool import ProposalPoolTrajectory
from .model_utils import CatePrior
from pytorch3d.ops.knn import knn_points
import logging
from matplotlib import cm

np.set_printoptions(precision=2)


class PerModelConfig:
    def __init__(self, name, c: dict, c_d={}) -> None:
        self.name = name
        # for proposal init
        self.init_proposal_radius = self.with_default(c, c_d, ["init_proposal_radius"])
        self.compute_error_radius = self.with_default(c, c_d, ["compute_error_radius"])
        if not isinstance(self.init_proposal_radius, list):
            self.init_proposal_radius = [self.init_proposal_radius]
        if not isinstance(self.compute_error_radius, list):
            self.compute_error_radius = [self.compute_error_radius]

        init_ratio = self.with_default(c, c_d, ["init_proposal_ratio"], 1.0)
        if not isinstance(init_ratio, list):
            init_ratio = [init_ratio]
        init_ratio = np.asarray(init_ratio)
        init_ratio = init_ratio / init_ratio.sum()

        assert len(self.init_proposal_radius) == len(self.compute_error_radius)
        assert len(self.init_proposal_radius) == len(init_ratio)

        self.num_init_proposal_list = []
        num_init_proposal_list = self.with_default(c, c_d, ["num_init_proposal_list"])
        for n_init in num_init_proposal_list:
            n = (init_ratio * n_init).astype(np.int)
            self.num_init_proposal_list.append(n.tolist())

        self.init_crop_mode = self.with_default(c, c_d, ["crop_mode"])
        assert self.init_crop_mode in ["sphere", "cylinder", "zbox"]

        # for probability
        self.sdf2value_cfg = self.with_default(c, c_d, ["sdf2value"])
        self.normal2value_cfg = self.with_default(c, c_d, ["normal2value"])
        self.background_omega = self.with_default(c, c_d, ["background_omega"])

        self.precompute_confidence_coverage_th = self.with_default(
            c, c_d, ["precompute_confidence_coverage_th"], 0.4
        )

        # for duplication removel
        self.duplication_removal_cfg = self.with_default(c, c_d, ["duplication_removal"])

        # for output
        self.output_sdf_th = self.with_default(c, c_d, ["output_sdf_th"])
        self.output_normal_deg_th = self.with_default(c, c_d, ["output_normal_th"])
        # self.output_overlap_th = self.with_default(c, c_d, ["output_overlap_th"])

        # for elimination
        self.proposal_min_size = self.with_default(c, c_d, ["proposal_min_size"])
        self.stationary_score_cfg = self.with_default(c, c_d, ["stationary_score"])
        self.scale_check_cfg = self.with_default(c, c_d, ["scale_check"])
        self.orientation_check_cfg = self.with_default(c, c_d, ["orientation_check"])
        self.database_similarity_cfg = self.with_default(c, c_d, ["database_similarity"])
        self.coverage_score_cfg = self.with_default(c, c_d, ["coverage_score"])
        # self.net_mc_check_cfg = self.with_default(c, c_d, ["net_mc_check"])

        # semantic map
        self.semantic_map = self.with_default(c, c_d, ["semantic_map"], [name])

        # seg
        self.seg_cfg = self.with_default(c, c_d, ["seg_cfg"], None)

        return

    def with_default(self, cfg, cfg_default, key_list, default=None):
        try:
            d = cfg
            for k in key_list:
                d = d[k]
            return d
        except:
            try:
                d = cfg_default
                for k in key_list:
                    d = d[k]
                return d
            except:
                return default


def find_neighborhood(full_scene_pcl, center, model_cfg, radius):
    assert len(center) == len(radius)
    if model_cfg.init_crop_mode == "sphere":
        nn_d = (full_scene_pcl[None, ...] - center).norm(dim=-1)
    else:
        nn_d = (full_scene_pcl[None, :, [0, 2]] - center[..., [0, 2]]).norm(dim=-1)  # P,N
    # nn_mask = nn_d < model_cfg.compute_error_radius
    nn_mask = nn_d < radius[:, None]
    return nn_mask


def pad_segmask(PPT: ProposalPoolTrajectory, mask, pad_th=0.0):
    # modify the mask to fit the ppt's segment
    # first remove invalid points
    # for valid points, also include the ones that belongs to fg segment
    if len(mask) == 0:
        return mask
    seg = PPT.segment
    new_mask = mask * PPT.available_mask
    ret_mask = []
    for i in range(len(new_mask)):
        fg_seg = seg[new_mask[i]].unique()
        if len(fg_seg) == 0:
            logging.warning(
                "some proposal neighborhoods only has invalid points, use original mask to continue"
            )
            ret_mask.append(mask[i])
        else:
            _m = torch.zeros_like(mask[0]).bool()
            for j in fg_seg:
                seg_mask = seg == j
                ratio = (new_mask[i] * seg_mask).sum() / seg_mask.sum()
                if ratio > pad_th:
                    _m[seg_mask] = True
                else:  # use the old mask of this seg instead
                    _m[seg_mask] = new_mask[i][seg_mask]
            ret_mask.append(_m)
    ret_mask = torch.stack(ret_mask, 0)
    return ret_mask


def sample_mesh_pts(mesh_list, N, device):
    pcl_list = []
    for m in mesh_list:
        if isinstance(m, trimesh.Trimesh):
            if m.vertices.shape[0] == 0:  # dummy place holder
                samples = np.zeros((N, 3))
                normals = np.zeros((N, 3))
            else:
                samples, normals = None, None
                while samples is None or samples.shape[0] < N:
                    _samples, face_id = trimesh.sample.sample_surface_even(m, N)
                    _normals = np.asarray(m.face_normals[face_id]).copy()
                    if samples is None:
                        samples = _samples
                        normals = _normals
                    else:
                        samples = np.concatenate([samples, _samples], 0)
                        normals = np.concatenate([normals, _normals], 0)
                samples = samples[:N, :]
                normals = normals[:N, :]
                assert samples.shape[0] == N
            pcl_list.append(np.concatenate([samples, normals], -1))
        else:
            choice = np.random.choice(len(m), N, replace=True)
            pcl_list.append(m[choice])
    pcl = torch.from_numpy(np.stack(pcl_list, axis=0)).float()
    pcl = pcl.to(device)  # B,N,3
    return pcl


def compute_bbox(pcl_rot, basis):
    pcl_can_world_scale = torch.einsum("bni,bij->bnj", pcl_rot, basis)
    pred_shape_pcl_can_h = pcl_can_world_scale.max(dim=1).values
    pred_shape_pcl_can_l = pcl_can_world_scale.min(dim=1).values
    bbox_center = (pred_shape_pcl_can_h + pred_shape_pcl_can_l) / 2.0
    bbox = (pred_shape_pcl_can_h - pred_shape_pcl_can_l) / 2.0
    return bbox, bbox_center, pcl_can_world_scale


def icp_align(n_iter, pred_pcl_centered, knn_pcl_aligned, optimal_R, chunk=32):
    logging.info(f"icp refining (chunk={chunk})... ")
    knn_pcl_aligned_list, optimal_R_list = [], []
    cur = 0
    while cur < len(pred_pcl_centered):
        _pred_pcl_centered = pred_pcl_centered[cur : cur + chunk].clone()
        _knn_pcl_aligned = knn_pcl_aligned[cur : cur + chunk].clone()
        _optimal_R = optimal_R[cur : cur + chunk].clone()

        B, K, N, _ = _knn_pcl_aligned.shape
        pcl_centroid = _pred_pcl_centered.mean(dim=1, keepdim=True)  # B,1,3
        knn_pcl_centroid = _knn_pcl_aligned.mean(dim=2, keepdim=True)  # B,K,1,3
        icp_dst = _pred_pcl_centered - pcl_centroid
        icp_src = _knn_pcl_aligned - knn_pcl_centroid
        for icp_iter in range(n_iter):
            _, _, nn_in_dst = knn_points(
                icp_src.reshape(B * K, N, 3),
                icp_dst.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, N, 3),
                return_nn=True,
            )
            nn_in_dst = nn_in_dst.reshape(B, K, N, 3)
            # W = (nn_in_dst[..., None] @ icp_src.unsqueeze(3)).sum(2)  # B,K,3,3
            W = torch.einsum("bkni,bknj->bkij", nn_in_dst, icp_src)
            icp_u, _, icp_vh = torch.linalg.svd(W.double())
            icp_R = (icp_u @ icp_vh).float()
            icp_src = torch.einsum("bkij,bknj->bkni", icp_R, icp_src)
            _optimal_R = icp_R @ _optimal_R
        _knn_pcl_aligned = icp_src + pcl_centroid.unsqueeze(1)

        knn_pcl_aligned_list.append(_knn_pcl_aligned)
        optimal_R_list.append(_optimal_R)
        cur += chunk
    knn_pcl_aligned_list = torch.cat(knn_pcl_aligned_list, 0)
    optimal_R_list = torch.cat(optimal_R_list, 0)
    # assert knn_pcl_aligned_list.shape[0] == B
    # assert optimal_R_list.shape[0] == B
    return knn_pcl_aligned_list, optimal_R_list


def query_database(scale, z_so3, z_inv, database, chunk=32):
    cur = 0
    knn_pcl_aligned, optimal_R = [], []
    knn_scale, nn_dist, nn_info_bbox = [], [], []
    while cur < len(scale):
        _scale = scale[cur : cur + chunk]
        _z_so3, _z_inv = z_so3[cur : cur + chunk], z_inv[cur : cur + chunk]

        # ! WARNING: THE SOLVED BASIS/R IS IN O(3) NOT ALWAYS SO(3) because it's possible det(R) = -1
        device = _z_so3.device
        # * Query NN from database
        nn_idx, nn_info, _nn_dist = database.query(
            z_so3=_z_so3, z_inv=_z_inv, device=device, use_inv=True, use_so3=True, use_joint=True
        )  # B,K
        # * Align NN PCL to pred PCL with latent registration
        knn_pcl_centered = nn_info["pcl"] - nn_info["center"]  # B,K,N,3
        knn_z_so3 = nn_info["z_so3"]  # B,K,C,3; z_so3 [B,C,3]
        _knn_scale = nn_info["scale"]
        latent_W = (_z_so3[:, None, :, :, None] @ knn_z_so3.unsqueeze(-2)).sum(2)
        u, _, vh = torch.linalg.svd(latent_W.double())
        _optimal_R = (u @ vh).float()  # B,K,3,3  R @ knn -> obs
        knn_pcl_rotated = _optimal_R[:, :, None, :, :] @ knn_pcl_centered[..., None]
        knn_pcl_rotated = knn_pcl_rotated.squeeze(-1)  # B,K,N,3
        _knn_pcl_aligned = (
            knn_pcl_rotated / _knn_scale[..., None, None] * _scale[:, None, None, None]
        )

        knn_pcl_aligned.append(_knn_pcl_aligned)
        optimal_R.append(_optimal_R)
        knn_scale.append(_knn_scale)
        nn_dist.append(_nn_dist)
        nn_info_bbox.append(nn_info["bbox"])
        cur += chunk

    knn_pcl_aligned = torch.cat(knn_pcl_aligned, 0)
    optimal_R = torch.cat(optimal_R, 0)
    knn_scale = torch.cat(knn_scale, 0)
    nn_dist = np.concatenate(nn_dist, 0)
    nn_info_bbox = torch.cat(nn_info_bbox, 0)

    return knn_pcl_aligned, optimal_R, knn_scale, nn_dist, nn_info_bbox


def compute_recon_obs_dist(
    pid,
    ppt: ProposalPoolTrajectory,
    shape_pcl,
    center,
    # scale,
    # nn_scale,
    chunk: int,
    sdf_max: int,
    config: PerModelConfig,
):
    normal_flag = ppt.full_nrm is not None
    full_scene_pcl = ppt.full_pcl
    B, N, Dim = shape_pcl.shape
    obs2recon_D = []  # the scene pts to reconstructed shape distance
    recon2obs_D = []  # the reconstructed shape pts to nearest observed scene pts disttance
    nn_r = ppt.fetch_nn_r(pid)
    mask = find_neighborhood(full_scene_pcl, center, config, nn_r)
    invalid_mask = (ppt.unavailable_mask)[None, :].expand_as(mask)
    mask[invalid_mask] = False
    for bid in range(B):
        m = mask[bid]
        dist = torch.ones_like(full_scene_pcl[:, 0]) * sdf_max
        active_query = full_scene_pcl[m]
        cur, _dist_list = 0, []
        _angle_list = []
        while cur < len(active_query):
            _dist_sq, idx, _ = knn_points(
                active_query[None, cur : cur + chunk, :], shape_pcl[bid : bid + 1, :, :3]
            )
            _dist_list.append(torch.sqrt(torch.clamp(_dist_sq, min=0.0)).squeeze(0).squeeze(-1))
            if normal_flag:
                assert Dim > 3, "when using normal. should have normals of recon mesh"
                _obs_normal = (ppt.full_nrm[m][cur : cur + chunk, :]).clone()
                _shape_normal = shape_pcl[bid, :, 3:][idx.squeeze(0).squeeze(-1)]
                _angle_diff = torch.acos((_obs_normal * _shape_normal).sum(-1))
                _angle_list.append(_angle_diff)
            cur += chunk
        dist[m] = torch.cat(_dist_list, dim=0)
        if normal_flag:
            angle = torch.ones_like(full_scene_pcl[:, 0]) * 180
            angle[m] = torch.cat(_angle_list, dim=0) / np.pi * 180
            dist = torch.stack([dist, angle], -1)  # in degree
        obs2recon_D.append(dist)

        _dist_sq, idx, _ = knn_points(shape_pcl[bid : bid + 1, :, :3], active_query[None, ...])
        _recon2obs_D = torch.sqrt(torch.clamp(_dist_sq, min=0.0)).squeeze(0).squeeze(-1)
        idx = idx.squeeze(0).squeeze(-1)
        if normal_flag:
            # ! Nov 3rd, fixed a bug here
            # _obs_normal = (ppt.full_nrm[idx]).clone() # ! bug
            _obs_normal = (ppt.full_nrm[m][idx]).clone()
            _shape_normal = shape_pcl[bid, :, 3:]
            _angle_diff = torch.acos((_obs_normal * _shape_normal).sum(-1))
            _angle_diff = _angle_diff / np.pi * 180.0
            _recon2obs_D = torch.stack([_recon2obs_D, _angle_diff], -1)
        recon2obs_D.append(_recon2obs_D)
    obs2recon_D = torch.stack(obs2recon_D, 0)
    recon2obs_D = torch.stack(recon2obs_D, 0)
    if not normal_flag:
        obs2recon_D = obs2recon_D.unsqueeze(-1)
        recon2obs_D = recon2obs_D.unsqueeze(-1)
    return obs2recon_D, recon2obs_D, mask


################################################################################################


def compute_sim2database(
    check_config,
    pred_pcl,
    knn_pcl,
    scale,
    knn_scale,
    knn_R,
):
    B, K, N, _ = knn_pcl.shape
    p2d_d_sq, _, _ = knn_points(
        pred_pcl.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, N, 3),
        knn_pcl.reshape(B * K, N, 3),
    )
    p2d_d = torch.sqrt(torch.clamp(p2d_d_sq, min=0.0)).reshape(B, K, N)
    d2p_d_sq, _, _ = knn_points(
        knn_pcl.reshape(B * K, N, 3),
        pred_pcl.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, N, 3),
    )
    d2p_d = torch.sqrt(torch.clamp(d2p_d_sq, min=0.0)).reshape(B, K, N)
    # should measure in canonical space
    p2d_d_can = p2d_d / scale[:, None, None] * knn_scale[..., None]
    d2p_d_can = d2p_d / scale[:, None, None] * knn_scale[..., None]

    # accept based on the similarity
    inlier_th = check_config["inlier_th"]
    first_k = check_config["accept_topk"]
    all_ratio = (p2d_d_can < inlier_th).sum(dim=-1) + (d2p_d_can < inlier_th).sum(dim=-1)
    all_ratio = all_ratio / (2 * N)
    topk_nn = (all_ratio).topk(first_k, dim=-1)
    ratio = topk_nn.values.min(dim=-1).values
    best_id = topk_nn.indices[:, 0]
    selected_scale = torch.gather(knn_scale, index=best_id[:, None], dim=1).squeeze(-1)
    # fetch the sRt based on comparing the most similar database shape
    basis = torch.gather(knn_R, dim=1, index=best_id[:, None, None, None].expand(-1, -1, 3, 3))
    basis = basis.squeeze(1)  # B,3,num_basis
    # ! warning:  # B,3,num_basis
    return ratio, basis, selected_scale


def compute_coverage_score(check_config, obj2obs_DF):
    observed_mask = obj2obs_DF[..., 0] <= check_config["inlier_th"]
    if "inlier_angle_th" in check_config.keys() and obj2obs_DF.shape[-1] > 1:
        angle_mask = obj2obs_DF[..., 1] <= check_config["inlier_angle_th"]
        observed_mask = observed_mask * angle_mask
    coverage = observed_mask.sum(-1) / observed_mask.shape[1]
    return coverage


def check_scale(check_config, pred_pcl_can_world_scale, pids):
    valid_mask = torch.ones(len(pred_pcl_can_world_scale)).bool()

    for dim, rang in zip(check_config["dim"], check_config["range"]):
        if not isinstance(rang[0], list):
            rang = [rang]
        pts = pred_pcl_can_world_scale[..., dim]
        r = pts.norm(dim=-1)
        s = r.max(-1).values
        valid_mask_list = []
        for ra in rang:
            valid_mask_list.append((s >= min(ra)) * (s <= max(ra)))
        valid = torch.stack(valid_mask_list, 0).any(0)
        pass_rate = 100.0 * valid.sum() / float(len(valid))
        logging.info(
            f"[{pass_rate:.2f}%] Pass Scale-D{dim} check | {pids[~valid].cpu().numpy()} fail; iou={s.cpu().numpy()} range={rang}"
        )
        valid_mask = valid_mask * valid.cpu()
    return valid_mask


def check_orientation(check_config, basis, pids):
    valid_mask = torch.ones(len(basis)).bool()

    for dim, dir, rang in zip(
        check_config["dim"], check_config["target_dir"], check_config["range"]
    ):
        if not isinstance(rang[0], list):
            rang = [rang]
        assert isinstance(dim, int)
        target_dir = torch.Tensor(dir)[None].to(basis.device)
        target_dir = target_dir.expand_as(basis[:, :, dim])  # B,3
        inner = (target_dir * basis[:, :, dim]).sum(-1)
        angle = torch.acos(torch.clamp(inner, -1.0, 1.0)) / np.pi * 180  # y is upward
        valid_mask_list = []
        for ra in rang:
            valid_mask_list.append((angle >= min(ra)) * (angle <= max(ra)))
        valid = torch.stack(valid_mask_list, 0).any(0)
        pass_rate = 100.0 * valid.sum() / float(len(valid))
        logging.info(
            f"[{pass_rate:.2f}%] Pass Pose-Dim{dim}-Dir{dir} check | {pids[~valid].cpu().numpy()} fail; iou={angle.cpu().numpy()} range={rang}"
        )
        valid_mask = valid_mask * valid.cpu()
    return valid_mask
