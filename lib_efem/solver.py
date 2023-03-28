import sys
import os
import os.path as osp

sys.path.append(osp.abspath(osp.dirname(__file__)))
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../lib_shape_prior/")))


import torch
import imageio
from torch import nn
import numpy as np
from tqdm import tqdm
import trimesh

from .misc import setup_seed
from .viz_utils import viz_single_proposal, viz_scene_jointly, viz_database
from .viz_utils import render
from .database import Database
from .proposalpool import ProposalPoolTrajectory
from .model_utils import CatePrior
from .solver_utils import *

from .misc import cfg_with_default
from core.models.utils.occnet_utils.mesh_extractor2 import Generator3D as Generator3D_MC
from core.models.utils.ndf_utils.pcl_extractor import Generator3D as Generator3D_Grad
from pytorch3d.ops.knn import knn_points
import shutil
import logging
from matplotlib import cm
from torch_scatter import scatter_mean
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix


class Solver:
    def __init__(self, cfg, lite_print_flag=False) -> None:
        self.prior_name_list = [k for k in cfg["shape_priors"].keys()]
        self.configure_global_settings(cfg["solver_global"])
        self.configure_per_model_settings(cfg["solver_per_model"])
        self.configure_viz_settings(cfg["viz"])
        self.print_flag = lite_print_flag  # used for demo printing
        return

    def configure_global_settings(self, cfg):
        # * global settings for the solver
        # network config
        self.sdf_flag = cfg_with_default(cfg, ["use_sdf"], True)
        self.normal_flag = cfg_with_default(cfg, ["use_normal"], False)

        if not self.sdf_flag:
            assert not self.normal_flag, "UDF does not support normal"
        if self.sdf_flag:
            self.mesh_extractor = Generator3D_MC(**cfg["mesh_extractor"])
        else:
            self.mesh_extractor = Generator3D_Grad(**cfg["mesh_extractor"])
        # solver steps control
        self.phase1_max_steps = cfg["phase1_max_steps"]
        self.phase2_max_steps = cfg["phase2_max_steps"]
        self.num_episode = cfg["num_episode"]
        # chunks
        self.est_chunk = cfg["est_chunk"]
        self.query_chunk = cfg["query_chunk"]
        self.filter_chunk_size = cfg_with_default(cfg, ["filter_chunk"], 32)
        # for globally tracking proposals
        self.pid_max = 0
        self.sdf_max = 1000
        # joint optimization
        self.joint_background_omega = cfg["joint_background_omega"]
        self.joint_duplication_removal_cfg = cfg["joint_duplication_removal_cfg"]
        self.inclusion_removal_start = cfg_with_default(
            cfg, ["inclusion_removal_start"], self.phase1_max_steps
        )

        return

    def configure_per_model_settings(self, cfg):
        # * each model (trained shape prior) can have different solver configs
        # use a dict to maintain the configs
        self.model_cfg_dict = {}
        for key in self.prior_name_list:
            c = cfg[key]
            c_d = {} if "inherit" not in c.keys() else cfg[c["inherit"]]
            self.model_cfg_dict[key] = PerModelConfig(key, c, c_d)
        return

    def configure_viz_settings(self, v_cfg):
        self.viz_flag = cfg_with_default(v_cfg, ["viz_flag"], True)

        self.scene_viz_args = cfg_with_default(v_cfg, ["scene_viz_args"], {})
        self.step_viz_args = cfg_with_default(v_cfg, ["step_viz_args"], {})

        self.viz_input_flag = cfg_with_default(v_cfg, ["viz_input"], True)
        self.viz_iter_flag = cfg_with_default(v_cfg, ["viz_iter"], True)
        self.viz_iter_start = cfg_with_default(v_cfg, ["viz_iter_start"], 0)
        self.viz_detailed_flag = cfg_with_default(v_cfg, ["viz_detailed_proposal"], True)
        self.viz_results_during_filtering_flag = cfg_with_default(
            v_cfg, ["viz_results_during_filtering"], True
        )

    ###################################################################################################

    def init_new_proposals(self, PPT: ProposalPoolTrajectory, epi: int, model_cfg: PerModelConfig):
        ava_mask = PPT.available_mask
        if ava_mask.sum() > model_cfg.proposal_min_size:
            # have enough ava pts, init new proposals from it, now **uniformly**
            full_pcl = PPT.full_pcl
            anchor_set = full_pcl[~PPT.unavailable_mask]
            n_list = model_cfg.num_init_proposal_list[epi]
            for i in range(len(n_list)):  # support multi radius setting
                N = n_list[i]

                seed = np.random.choice(len(anchor_set), N)
                anchor = anchor_set[seed]

                R_init = model_cfg.init_proposal_radius[i]
                R_error = model_cfg.compute_error_radius[i]
                # the second dim is Y, the upward dim
                if model_cfg.init_crop_mode == "sphere":
                    d = (full_pcl[:, None, :] - anchor[None, :, :]).norm(dim=-1)  # P,N
                    proposal_mask = (d < R_init).transpose(1, 0)
                elif model_cfg.init_crop_mode == "cylinder":
                    d = (full_pcl[:, None, [0, 2]] - anchor[None, :, [0, 2]]).norm(dim=-1)  # P,N
                    proposal_mask = (d < R_init).transpose(1, 0)
                elif model_cfg.init_crop_mode == "zbox":
                    # ! only init, the later is using the cylinder
                    euler = torch.zeros(N, 3)
                    euler[:, 0] = torch.rand(N) * np.pi * 2
                    rand_R = euler_angles_to_matrix(euler, "YZX").to(full_pcl.device)  # N,3,3
                    d = torch.einsum("bij,ni->bnj", rand_R, full_pcl)[..., [0, 2]]
                    bound = torch.rand(N, 2) * (R_init[1] - R_init[0]) + R_init[0]
                    proposal_mask = d < bound[:, None, :].to(full_pcl.device)
                    proposal_mask = proposal_mask.all(dim=-1)
                else:
                    raise NotImplementedError()
                init_W = torch.zeros_like(proposal_mask).float()
                init_W[proposal_mask] = 1.0
                init_W[PPT.unavailable_mask[None, :].expand_as(init_W)] = 0.0
                # append to PPT
                pids = [i for i in range(self.pid_max, self.pid_max + len(init_W))]
                PPT.append_new_proposals(W=init_W, epi=epi, ids=pids, nn_r_list=[R_error] * N)
                self.pid_max += len(init_W)
        return

    def M_step_sample_pcl(self, ppt_dict: dict, model_dict: dict, joint_assign_flag=False):
        active_w, active_id, score = {}, {}, {}
        for prior_name in self.prior_name_list:
            ppt: ProposalPoolTrajectory = ppt_dict[prior_name]
            _w, _id = ppt.active_active_latest_w_id
            # check if _w is too small
            # * this check is only useful in fg-bg mode, since the remove small proposal operation is not applied there
            _w_check = (_w > 0).sum(-1)
            invalid_w_mask = _w_check == 0
            _w[
                invalid_w_mask, :
            ] += 1.0  # ! if invalid, this proposal will become a full proposal over the scene
            if len(_id) == 0:
                continue  # no active
            active_w[prior_name], active_id[prior_name] = _w, _id
            if joint_assign_flag:
                try:
                    score[prior_name] = ppt.fetch(_id, -2, ["score_stationary"])["score_stationary"]
                except:
                    logging.warning(
                        "missing stationary score in joint assignment, use 1.0 for each prop"
                    )
                    score[prior_name] = torch.ones_like(_id).float()
        sampled_xyz, sampled_idx = {}, {}

        if joint_assign_flag:
            W = torch.cat([_w for _w in active_w.values()], 0)
            S = torch.cat([_s for _s in score.values()], 0).to(W.device)
            _W = W.transpose(1, 0)
            omega = self.joint_background_omega * torch.ones_like(_W[:, :1])

            joint_W = torch.cat([omega, _W * S[None, :]], -1)
            full_assign = torch.multinomial(joint_W, num_samples=1)
            fg_assignment = (full_assign > 0).squeeze(-1)
            prop_assignment = full_assign[fg_assignment] - 1
            cluster_assignment = torch.zeros_like(W).bool()
            cluster_assignment[:, fg_assignment] = torch.stack(
                [prop_assignment == i for i in range(len(S))], 0
            ).squeeze(-1)
            # handle singular case, if one prop has never assigned points
            singular_mask = ~cluster_assignment.any(-1)
            if singular_mask.any():
                # ! this might need to be checked
                logging.warning(
                    "Singular case happens in joint assignment, one proposal has no assigned points"
                )
                for _ind, singular in enumerate(singular_mask):
                    if singular:
                        slots = ~cluster_assignment.any(0)
                        cluster_assignment[_ind][(slots.float() * W[_ind]).argmax()] = True

            cur = 0
            for prior_name in active_w.keys():
                _w = active_w[prior_name]
                _assignment = cluster_assignment[cur : cur + len(_w)].float()
                # * to better sample the assigned fg, here also use weight to do multinomial
                multinomial_w = _w * _assignment
                _idx = torch.multinomial(
                    multinomial_w,
                    num_samples=model_dict[prior_name].field_input_n,
                    replacement=True,
                )  # B,N
                _xyz = ppt_dict[prior_name].full_pcl.clone()[_idx]
                sampled_xyz[prior_name], sampled_idx[prior_name] = _xyz, _idx
                cur += len(_w)
        else:
            # * isolated samples
            for prior_name in active_w.keys():
                _w = active_w[prior_name]
                # first compute the fg prob
                C = self.model_cfg_dict[prior_name].background_omega
                fg_prob = _w / (_w + C)
                # then sample an assignment from this prob
                fg_assignment = torch.bernoulli(fg_prob)
                # finally, in the fg assignment sample N pts for network input
                # * to better sample the assigned fg, here also use weight to do multinomial
                multinomial_w = _w * fg_assignment
                _idx = torch.multinomial(
                    multinomial_w,
                    num_samples=model_dict[prior_name].field_input_n,
                    replacement=True,
                )  # B,N
                _xyz = ppt_dict[prior_name].full_pcl.clone()[_idx]
                sampled_xyz[prior_name], sampled_idx[prior_name] = _xyz, _idx
        # TODO: handle if there are no fg !!!!
        return sampled_xyz, sampled_idx, active_id

    def M_estimate(self, est_pcl: torch.Tensor, model: CatePrior):
        logging.info("estimating ...")
        typ = est_pcl.dtype
        centroid = est_pcl.mean(dim=1, keepdim=True)
        est_pcl = est_pcl - centroid
        s, z_so3, z_inv, cur = [], [], [], 0
        t = centroid
        cur = 0
        if not self.print_flag:
            bar = tqdm(total=len(est_pcl))
        while cur < len(est_pcl):
            _ret = model.encode(est_pcl[cur : cur + self.est_chunk].transpose(2, 1))
            if len(_ret) > 3:
                _t, _s, _z_so3, _z_inv = _ret
                t[cur : cur + self.est_chunk] = t[cur : cur + self.est_chunk] + _t
            else:
                _s, _z_so3, _z_inv = _ret
            s.append(_s.type(typ))
            z_so3.append(_z_so3.type(typ)), z_inv.append(_z_inv.type(typ))
            cur += self.est_chunk
            if not self.print_flag:
                bar.update(self.est_chunk)
        # bar.close()
        s = torch.cat(s, 0)
        z_so3, z_inv = torch.cat(z_so3, 0), torch.cat(z_inv, 0)
        return t, s, z_so3, z_inv

    def E_query(
        self,
        pid,
        PPT: ProposalPoolTrajectory,
        model: CatePrior,
        mode_cfg: PerModelConfig,
        center: torch.Tensor,
        scale: torch.Tensor,
        z_so3: torch.Tensor,
        z_inv: torch.Tensor,
    ):
        full_scene_pcl = PPT.full_pcl

        logging.info("approximating errors ...")
        B, N = len(center), full_scene_pcl.shape[0]
        nn_r = PPT.fetch_nn_r(pid)
        query_nn_mask = find_neighborhood(full_scene_pcl, center, mode_cfg, nn_r)

        invalid_mask = (PPT.unavailable_mask)[None, :].expand_as(query_nn_mask)
        query_nn_mask[invalid_mask] = False
        sdf_list = []
        if self.normal_flag:  # require grad
            old_grad_context = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            model.decoder.eval()
            for param in model.decoder.parameters():
                param.requires_grad = False
        _iter = range(B) if self.print_flag else tqdm(range(B))
        for i in _iter:
            sdf = torch.ones(N, 3 if self.normal_flag else 1) * self.sdf_max
            sdf = sdf.to(full_scene_pcl.device)
            roi_pcl = full_scene_pcl[query_nn_mask[i]]
            if self.normal_flag:
                roi_normal = PPT.full_nrm[query_nn_mask[i]]
            if len(roi_pcl) > 0:
                _cur, _total = 0, len(roi_pcl)
                _sdf_list = []
                embedding = {
                    "z_so3": z_so3[i : i + 1],
                    "z_inv": z_inv[i : i + 1],
                    "s": scale[i : i + 1],
                    "t": center[i : i + 1],
                }
                while _cur < _total:  # chunk
                    _input_pcl = roi_pcl[_cur : _cur + self.query_chunk][None, ...]
                    _input_pcl.requires_grad = self.normal_flag
                    _sdf = model.decoder(
                        _input_pcl,
                        None,
                        embedding,
                        True,
                    ).squeeze(0)
                    if self.normal_flag:
                        _J = _sdf.sum()
                        _J.backward()
                        _nrm_pred = (_input_pcl.grad).squeeze(0)
                        _nrm_pred = nn.functional.normalize(_nrm_pred, dim=-1)
                        _nrm_obs = roi_normal[_cur : _cur + self.query_chunk]
                        _nrm_inner = (_nrm_obs * _nrm_pred).sum(-1)
                        _nrm_degdiff = torch.acos(torch.clamp(_nrm_inner, -1.0, 1.0)) / np.pi * 180
                        _sdf = torch.stack([_sdf, _nrm_inner, _nrm_degdiff], 1)
                    else:
                        _sdf = _sdf[:, None]
                    _sdf_list.append(_sdf)
                    _cur += self.query_chunk
                _sdf = torch.cat(_sdf_list, 0)
                sdf[query_nn_mask[i]] = _sdf
            sdf_list.append(sdf)
        if self.normal_flag:
            torch.set_grad_enabled(old_grad_context)
        ret_sdf = torch.stack(sdf_list, 0)
        # * SDF will have 3 channel if use normal, the second is the inner of two normals and the third is the acos in deg
        return ret_sdf, query_nn_mask

    @staticmethod
    def E_get_floating_d2v(cfg, step: int, device):
        k_start, k_end = (cfg["k_start"], cfg["k_end"])
        th_start, th_end = (cfg["th_start"], cfg["th_end"])
        end_step = cfg["end_step"]
        clip_step = np.clip(step, 0, end_step)
        k = clip_step / float(end_step) * (k_end - k_start) + k_start
        th = clip_step / float(end_step) * (th_end - th_start) + th_start
        upper = cfg["out_th_value"]
        return k, th, upper

    def E_compute_W(self, D, D_mask, step: int, model_config: PerModelConfig):
        # TSDF TND
        # if D or normal diff is larger than a floating th, then directly set to a very large value
        # also the alpha and beta of the inside th part is changing
        k_sdf, th_sdf, u_sdf = self.E_get_floating_d2v(model_config.sdf2value_cfg, step, D.device)
        dist_diff = abs(D[..., 0])[D_mask].clone()
        dist_out_mask = dist_diff > th_sdf
        exponent = k_sdf * dist_diff
        exponent[dist_out_mask] = u_sdf

        if self.normal_flag:
            k_nrm, th_nrm, u_nrm = self.E_get_floating_d2v(
                model_config.normal2value_cfg, step, D.device
            )
            normal_diff = D[..., 2][D_mask].clone()  # in degree
            normal_out_mask = normal_diff > th_nrm
            normal_exponent = normal_diff * k_nrm
            normal_exponent[normal_out_mask] = u_nrm
            exponent = exponent + normal_exponent
        W = torch.zeros_like(D[..., 0])
        W[D_mask] = torch.exp(-exponent)

        # * also compute an inlier mask for removing too small proposals
        inlier_mask = torch.zeros_like(D[..., 0]).bool()
        inlier_mask[D_mask] = abs(D[..., 0])[D_mask] <= th_sdf
        if self.normal_flag:
            inlier_mask[D_mask] = inlier_mask[D_mask] * (D[..., 2][D_mask] <= th_nrm)

        W[torch.isinf(W)] = 0.0
        W[torch.isnan(W)] = 0.0
        assert (W <= 1.0).all()

        return W, inlier_mask

    def compute_score_stationary(self, D, sample_idx, model_config: PerModelConfig):
        # * compute the likelihood like score
        # * how much percentage is the input computed as correctly agree with the output
        dist_th = model_config.stationary_score_cfg["sdf_th"]
        est_D = abs(torch.gather(D, dim=1, index=sample_idx[..., None].expand(-1, -1, D.shape[-1])))
        est_dist = abs(est_D[..., 0])
        score = est_dist < dist_th
        if self.normal_flag:
            angle_th = model_config.stationary_score_cfg["angle_th"]
            est_normal_diff = est_D[..., 2]  # in deg
            score = score * (est_normal_diff < angle_th)
        score = score.float().mean(-1)
        return score

    def EM_step(self, ppt_dict: dict, model_dict: dict, step: int, epi: int, joint_flag=False):
        # * prepare Samples for M step
        M_pcl, M_pcl_idx, active_id = self.M_step_sample_pcl(
            ppt_dict, model_dict, joint_assign_flag=joint_flag
        )
        # * forward each prior independently
        for prior_name in self.prior_name_list:
            logging.info(f"EM epi={epi} step={step} prior={prior_name}")
            model: CatePrior = model_dict[prior_name]
            model_cfg: PerModelConfig = self.model_cfg_dict[prior_name]
            ppt: ProposalPoolTrajectory = ppt_dict[prior_name]
            if not ppt.has_active:  # make sure that this ppt, or M_pcl[prior_name] has value
                logging.info(
                    f"PPT={prior_name} has no active proposal epi={epi} step={step}, skip."
                )
                continue

            # * actual M step, estimate the thetas
            center, scale, z_so3, z_inv = self.M_estimate(M_pcl[prior_name], model)
            # * E step approximate error
            D, D_nn_mask = self.E_query(
                active_id[prior_name], ppt, model, model_cfg, center, scale, z_so3, z_inv
            )
            # * make new W (exp)
            W_new, inlier_mask = self.E_compute_W(
                D=D, D_mask=D_nn_mask, step=step, model_config=self.model_cfg_dict[prior_name]
            )

            # * also compute the likelihood
            score_stationary = self.compute_score_stationary(
                D, M_pcl_idx[prior_name], self.model_cfg_dict[prior_name]
            )
            # * also give the semantic prediction
            if model.cls_head is None:
                sem_label = torch.zeros_like(z_inv[:, 0])
            else:
                sem_score = model.cls_head(z_inv)
                sem_label = sem_score.argmax(-1)
            sem = [self.model_cfg_dict[prior_name].semantic_map[int(i)] for i in sem_label]

            # * record this step
            ppt.update(
                ids=active_id[prior_name],
                step=step,
                D=D,
                z_inv=z_inv,
                z_so3=z_so3,
                scale=scale,
                center=center,
                est_pcl=M_pcl[prior_name],
                est_sample_index=M_pcl_idx[prior_name],
                score_stationary=score_stationary,
                inlier_mask=inlier_mask,
                sem=sem,
            )
            ppt.update(ids=active_id[prior_name], step=step + 1, W=W_new)
            # remove the history to speed up, ppt will automatically decide whehter to maintain full trace
            ppt.clean(active_id[prior_name], step=step - 1)

        return ppt_dict

    ###################################################################################################

    def remove_small_proposals(self, ppt: ProposalPoolTrajectory, model_cfg: PerModelConfig):
        active_ids = ppt.active_i
        if len(active_ids) == 0:
            return ppt
        inlier_mask = ppt.fetch(active_ids, -2, keys=["inlier_mask"])["inlier_mask"]
        proposal_size_cnt = inlier_mask.sum(-1)
        valid_size_mask = proposal_size_cnt >= model_cfg.proposal_min_size
        if not valid_size_mask.all():
            ppt.log_decision(active_ids[~valid_size_mask], reason="proposal too small")
        return ppt

    def remove_duplications(
        self, ppt_dict: dict, prior_name: str, inclusion_flag=False, device=torch.device("cuda")
    ):
        # ! handle the inclusion case: now if a proposal is inside (than a threshold) another proposal (after the dup check, no equal cases), then this proposal is rejected
        # ! since we don't find the decomposition of object and tend to find a more "complete" and "simple" explaination of the scene
        # ! Now the equal, inclusion is handled here, hope the smaller overlapping case can be handled by joint assignment

        if prior_name == "joint":
            sdf_th = self.joint_duplication_removal_cfg["sdf_th"]
            iou_th = self.joint_duplication_removal_cfg["iou_th"]
            inclusion_th = self.joint_duplication_removal_cfg["inclusion_th"]
            active_ids, D, score = [], [], []
            active_flag = False
            for ppt in ppt_dict.values():
                _id = ppt.active_i
                active_ids += _id
                if len(_id) > 0:
                    active_flag = True
                    data = ppt.fetch(_id, -2, keys=["D", "score_stationary"])
                    D.append(data["D"][..., 0].to(device))
                    score.append(data["score_stationary"].to(device))
            if not active_flag:
                # if len(active_ids) == 0:
                return ppt_dict
            active_ids = torch.stack(active_ids)

            D, score = torch.cat(D, 0), torch.cat(score, 0)
        elif prior_name in self.prior_name_list:
            sdf_th = self.model_cfg_dict[prior_name].duplication_removal_cfg["sdf_th"]
            iou_th = self.model_cfg_dict[prior_name].duplication_removal_cfg["iou_th"]
            ppt: ProposalPoolTrajectory = ppt_dict[prior_name]
            active_ids = ppt.active_i
            if len(active_ids) == 0:
                return ppt_dict
            data = ppt.fetch(active_ids, -2, keys=["D", "score_stationary"])
            D, score = data["D"][..., 0].to(device), data["score_stationary"].to(device)
        else:
            raise NotImplementedError()

        N = len(D)
        active_mask = torch.ones(N).to(device)
        for i in range(N):
            if active_mask[i] == 0:
                continue  # if removed already
            # find duplication
            B = (abs(D) < sdf_th) * active_mask[:, None]
            inter = torch.logical_and(B, B[i : i + 1])
            union = torch.logical_or(B, B[i : i + 1])
            iou = inter.sum(-1).float() / (union.sum(-1).float() + 1e-6)
            duplication_mask = iou >= iou_th
            # merge
            if duplication_mask.sum() > 1:
                _score = score.clone()
                _score[~duplication_mask] = 0.0
                merge_to_i = _score.argmax()
                logging.info(
                    f"ID: {active_ids[duplication_mask]} duplicated; iou={iou[duplication_mask]}, use {active_ids[merge_to_i]} whose score={_score[merge_to_i]}"
                )
                active_mask[duplication_mask] = 0.0
                active_mask[merge_to_i] = 1.0

        if inclusion_flag:
            for i in range(N):
                if active_mask[i] == 0:
                    continue  # if removed already
                # find duplication
                B = (abs(D) < sdf_th) * active_mask[:, None]
                inter = torch.logical_and(B, B[i : i + 1])
                ratio = inter.sum(-1).float() / (B[i : i + 1].sum().float() + 1e-6)
                ratio[i] = 0.0
                inclusion_ratio = ratio.max()
                if inclusion_ratio > inclusion_th:  # reject
                    active_mask[i] = 0.0
                    logging.info(
                        f"ID: {i} included by others removed; inclusion ratio={inclusion_ratio}, th={inclusion_th}"
                    )

        active_mask = active_mask > 0
        logging.info(f"Duplicate reduce active prop from {N} -> {int(sum(active_mask))}")
        logging.info(f"current active ids: {active_ids[active_mask]}")

        # kill rejected ids
        if not active_mask.all():
            if prior_name == "joint":
                for name in ppt_dict.keys():
                    ppt_dict[name].log_decision(
                        active_ids[~active_mask], reason="duplicated or included"
                    )
            else:
                ppt_dict[prior_name].log_decision(
                    active_ids[~active_mask], reason="duplicated or included"
                )
        return ppt_dict

    ###################################################################################################

    def extract_shape(
        self,
        PPT: ProposalPoolTrajectory,
        model,
        ids,
        step,
        scale_base=1.0,  # for a reasonable marching space
        device=torch.device("cuda"),
        show_progress=True,
    ):
        # * Support MC for SDF and Gradient Dense PCL for UDF
        try:
            PPT.fetch(ids=ids, step=step, keys=["mesh"])
            logging.info("mesh has been extracted, skip duplicated MC")
            return PPT
        except:
            logging.info("extracting ...")
            # MarchingCubes the estimated codes, only do this when necessary
            data = PPT.fetch(ids=ids, step=step, keys=["z_so3", "z_inv", "scale", "center"])
            z_so3, z_inv = data["z_so3"].to(device), data["z_inv"].to(device)
            scale, center = data["scale"].to(device), data["center"].to(device)

            B = z_so3.shape[0]
            mesh_list = []
            it = tqdm(range(B)) if show_progress and not self.print_flag else range(B)
            scale_base = float(scale_base)
            for i in it:
                embedding = {
                    "z_so3": z_so3[i : i + 1],
                    "z_inv": z_inv[i : i + 1],
                    "s": torch.ones_like(scale[i : i + 1]) * scale_base,
                    "t": torch.zeros_like(center[i : i + 1]),
                }
                if self.sdf_flag:
                    _mesh = self.mesh_extractor.generate_from_latent(c=embedding, F=model.decoder)
                    _mesh.apply_scale(float(scale[i : i + 1].detach().cpu()) / scale_base)
                    _mesh.apply_translation(center[i].squeeze(0).detach().cpu())
                    mesh_list.append(_mesh)
                else:
                    _dense_pcl = self.mesh_extractor.generate_from_latent(
                        c=embedding, F=model.decoder
                    )
                    _dense_pcl = _dense_pcl * float(scale[i : i + 1].detach().cpu()) / scale_base
                    _dense_pcl = _dense_pcl + center[i].detach().cpu().numpy()
                    mesh_list.append(_dense_pcl)

            PPT.update(ids, step, mesh=mesh_list)
            return PPT

    def reject_proposals(
        self,
        ppt: ProposalPoolTrajectory,
        model: CatePrior,
        database: Database,
        model_config: PerModelConfig,
    ):
        # * Correct object pattern theta reach local optimal, but not all local optimal is desired
        # * simply use some handcrafted geometric rules to filter the proposal
        # reject activate proposals that does not satisfy the output criteria

        # prepare predictions and temporarily make the assignment (isolated, no talk)
        active_ids = ppt.active_i  # for any active prop, filter them
        if len(active_ids) == 0:
            logging.info("no active, no need rejection")
            return ppt
        logging.info(
            f"PPT {ppt.name} before prop rejection has {len(active_ids)} active {active_ids}"
        )
        device = ppt.full_pcl.device
        ppt = self.extract_shape(ppt, model, active_ids, -2, database.mean_scale)
        data = ppt.fetch(
            active_ids,
            step=-2,
            keys=["mesh", "W", "D", "z_so3", "z_inv", "center", "scale", "est_sample_index"],
        )
        shape_mesh_list = data["mesh"]
        shape_pcl_list = sample_mesh_pts(shape_mesh_list, N=database.saved_pcl_n, device=device)
        z_so3, z_inv = data["z_so3"].to(device), data["z_inv"].to(device)
        center, scale = data["center"].to(device), data["scale"].to(device)
        knn_pcl_aligned, knn_R, knn_scale, _, knn_bbox = query_database(
            scale=scale, z_so3=z_so3, z_inv=z_inv, database=database, chunk=self.filter_chunk_size
        )
        pred_pcl_centered = shape_pcl_list[..., :3] - center
        n_icp_step = model_config.database_similarity_cfg["icp_refine_iter"]
        if n_icp_step > 0:  # do icp if set
            knn_pcl_aligned, knn_R = icp_align(
                n_icp_step, pred_pcl_centered, knn_pcl_aligned, knn_R, chunk=self.filter_chunk_size
            )

        D_from_net = data["D"].to(device)
        D_to_mesh_worldscale, D_to_obs_worldscale, _ = compute_recon_obs_dist(
            active_ids,
            ppt,
            shape_pcl=shape_pcl_list,
            center=center,
            chunk=self.query_chunk,
            sdf_max=self.sdf_max,
            config=model_config,
        )

        # make assignment and record
        assignment = abs(D_from_net[..., 0]) < model_config.output_sdf_th
        if self.normal_flag:
            assignment = assignment * (D_from_net[..., 2] < model_config.output_normal_deg_th)

        ppt.update(
            active_ids,
            -2,
            tmp_assignment=assignment.bool(),
            knn_pcl_aligned=knn_pcl_aligned,
            knn_R=knn_R,
            pred_pcl_centered=pred_pcl_centered,
        )

        # * Check the similarity to a known instance, also get the best fitted pose size
        SCORE_db_sim, basis, selected_scale = compute_sim2database(
            check_config=model_config.database_similarity_cfg,
            pred_pcl=pred_pcl_centered,
            knn_pcl=knn_pcl_aligned,
            scale=scale,
            knn_scale=knn_scale,
            knn_R=knn_R,
        )
        bbox, bbox_c, pred_pcl_can_world_scale = compute_bbox(pred_pcl_centered, basis)
        ppt.update(
            active_ids,
            -2,
            basis=basis,
            bbox=bbox,
            bbox_c=bbox_c,
            pred_pcl_can_world_scale=pred_pcl_can_world_scale,
        )
        ppt.update(
            active_ids,
            -2,
            score_db_sim=SCORE_db_sim,
        )

        # * Obs Coverage Check (the object should be observed more than X%
        # first convert to the canonical scale
        D_to_mesh = D_to_mesh_worldscale
        D_to_obs = D_to_obs_worldscale
        D_to_mesh[..., 0] = D_to_mesh[..., 0] / scale[:, None] * selected_scale[:, None]
        D_to_obs[..., 0] = D_to_obs[..., 0] / scale[:, None] * selected_scale[:, None]
        ppt.update(active_ids, -2, D_tomesh=D_to_mesh, DF_to_obs=D_to_obs)
        SCORE_coverage = compute_coverage_score(model_config.coverage_score_cfg, D_to_obs)
        ppt.update(
            active_ids,
            -2,
            score_coverage=SCORE_coverage,
        )

        # * Scale Check (the object should lie in a reasonable scale range)
        valid_scale_mask = check_scale(
            model_config.scale_check_cfg, pred_pcl_can_world_scale, active_ids
        )
        ppt.update(active_ids, -2, valid_scale_mask=valid_scale_mask)

        # * Orientation Check (sometimes the object may always towards one direction)
        valid_pose_mask = check_orientation(model_config.orientation_check_cfg, basis, active_ids)
        ppt.update(active_ids, -2, valid_pose_mask=valid_pose_mask)

        # * reject some trivially unreasonable proposals
        ppt.log_decision(active_ids[~valid_scale_mask], reason="scale-fail")
        ppt.log_decision(active_ids[~valid_pose_mask], reason="pose-fail")
        active_ids = ppt.active_i
        logging.info(
            f"PPT {ppt.name} after prop rejection has {len(active_ids)} active {active_ids}"
        )
        return ppt

    def accept_active(self, ppt_dict):
        # ! # sync the assigned part across models
        assignment_list = []
        for name in ppt_dict.keys():
            ppt: ProposalPoolTrajectory = ppt_dict[name]
            active_id = ppt.active_i
            if len(active_id) == 0:
                continue
            score_dict = ppt.fetch(
                active_id,
                -2,
                ["score_db_sim", "score_coverage", "score_stationary"],
            )
            # ! also pre-compute a confidence score, in order to keep track of the hparam, easy to mess things up when save this during converting after finished running
            model_cfg: PerModelConfig = self.model_cfg_dict[name]
            confidence = score_dict["score_stationary"]
            confidence = confidence * (
                torch.clamp(
                    score_dict["score_coverage"],
                    0.0,
                    model_cfg.precompute_confidence_coverage_th,
                )
                / model_cfg.precompute_confidence_coverage_th
            )
            score_dict["precompute_confidence"] = confidence
            assignment = ppt.fetch(active_id, -2, ["tmp_assignment"])["tmp_assignment"]
            assignment_list.append(assignment)
            ppt_dict[name].log_decision(
                active_id,
                reason="survive",
                mode="accept",
                assignment=assignment,
                score_dict=score_dict,
            )
        if len(assignment_list) == 0:
            logging.info("no active at all, no need for acceptance")
            return ppt_dict
        assignment_list = torch.cat(assignment_list, 0)
        assigned_mask = assignment_list.any(dim=0)
        for name in ppt_dict.keys():
            ppt_dict[name].update_assignment(assigned_mask)
        return ppt_dict

    def has_active(self, RET):
        for v in RET.values():
            if len(v.active_i) > 0:
                return True
        return False

    ###################################################################################################

    @torch.no_grad()
    def solve(self, model_dict, data_dict, database_dict, viz_dir, viz_prefix="", seed=12345):
        # init
        torch.cuda.empty_cache()
        setup_seed(seed)
        assert model_dict.keys() == database_dict.keys()
        assert set(database_dict.keys()) == set(self.prior_name_list)

        # prepare and viz input
        if self.viz_input_flag and self.viz_flag:
            self.viz_input(data_dict, viz_dir, viz_prefix)
        scene_pcl = data_dict["pointcloud"]
        if self.normal_flag:
            scene_nrm = data_dict["normals"]
        else:
            scene_nrm = None

        # init trajectory record
        RET = {}
        active_prop_counts = {}
        for prior_name in self.prior_name_list:
            RET[prior_name] = ProposalPoolTrajectory(
                full_pcl=scene_pcl,
                full_nrm=scene_nrm,
                name=prior_name,  # the category name
            )

        for epi in range(self.num_episode):
            # * init new proposals
            for prior_name in self.prior_name_list:
                self.init_new_proposals(RET[prior_name], epi, self.model_cfg_dict[prior_name])
                active_prop_counts[prior_name] = [len(RET[prior_name].active_i)]

            # * Phase 1: Isolated EM steps
            if self.print_flag:
                print("phase-1")
                _iter = tqdm(range(self.phase1_max_steps))
            else:
                _iter = range(self.phase1_max_steps)
            for ip1 in _iter:
                step = ip1
                logging.info("*" * (shutil.get_terminal_size()[0] - 100))
                logging.info(f"Solver Epi={epi} Iter={step} [Phase-1]")
                if not self.has_active(RET):
                    logging.info("No active exists, stop early")
                    break

                RET = self.EM_step(RET, model_dict, step, epi)
                # viz
                RET = self.viz_em_step(
                    RET, step, epi, model_dict, database_dict, viz_dir, viz_prefix
                )
                # merge inside each prior and remove small proposal
                for prior_name in self.prior_name_list:
                    RET[prior_name] = self.remove_small_proposals(
                        RET[prior_name], self.model_cfg_dict[prior_name]
                    )
                    RET = self.remove_duplications(RET, prior_name)
                    active_prop_counts[prior_name].append(len(RET[prior_name].active_i))

            # * Reject some bad prop inside each category independently
            if self.phase1_max_steps > 0:  # has phase 1
                for prior_name in self.prior_name_list:
                    RET[prior_name] = self.reject_proposals(
                        RET[prior_name],
                        model_dict[prior_name],
                        database_dict[prior_name],
                        self.model_cfg_dict[prior_name],
                    )
                if self.viz_results_during_filtering_flag:
                    # viz current active and accepted and active props
                    self.viz_results(
                        [p for p in RET.values()],
                        osp.join(
                            viz_dir,
                            "joint_viz",
                            f"{viz_prefix}_epi_{epi}_step_{step}_phase1_active+accept",
                        ),
                    )

            # * Phase 2: Joint EM steps
            if self.print_flag:
                print("phase-2")
                _iter = tqdm(range(self.phase2_max_steps))
            else:
                _iter = range(self.phase2_max_steps)
            for ip2 in _iter:
                step = ip2 + self.phase1_max_steps
                logging.info("*" * (shutil.get_terminal_size()[0] - 100))
                logging.info(f"Solver Epi={epi} Iter={step} [Phase-2]")
                if not self.has_active(RET):
                    logging.info("No active exists, stop early")
                    break

                RET = self.EM_step(RET, model_dict, step, epi, joint_flag=True)
                # viz
                RET = self.viz_em_step(
                    RET, step, epi, model_dict, database_dict, viz_dir, viz_prefix
                )
                # merge inside each prior and remove small proposal
                for prior_name in self.prior_name_list:
                    RET[prior_name] = self.remove_small_proposals(
                        RET[prior_name], self.model_cfg_dict[prior_name]
                    )
                    active_prop_counts[prior_name].append(len(RET[prior_name].active_i))
                RET = self.remove_duplications(
                    RET, "joint", inclusion_flag=step >= self.inclusion_removal_start
                )

            if self.phase2_max_steps > 0:  # has the second phase
                for prior_name in self.prior_name_list:
                    RET[prior_name] = self.reject_proposals(
                        RET[prior_name],
                        model_dict[prior_name],
                        database_dict[prior_name],
                        self.model_cfg_dict[prior_name],
                    )
                if self.viz_results_during_filtering_flag:
                    self.viz_results(
                        [p for p in RET.values()],
                        osp.join(
                            viz_dir,
                            "joint_viz",
                            f"{viz_prefix}_epi_{epi}_step_{step}_phase2_active+accept",
                        ),
                    )

            # * Final decision
            RET = self.accept_active(RET)

        self.viz_results([p for p in RET.values()], osp.join(viz_dir, f"{viz_prefix}_output"))
        return RET, active_prop_counts

    ################################################################################################################################

    def viz_input(self, data_dict, viz_dir="", viz_prefix="", force_viz=False, save_flag=True):
        viz_list = []
        if not self.viz_flag and not force_viz:
            return
        try:
            # render input pcl
            save_fn = osp.join(viz_dir, viz_prefix + "_full.png")
            if not osp.exists(save_fn):
                full_scene_pcl = data_dict["pointcloud"]
                viz_full_scene = render(
                    pcl_list=[full_scene_pcl.detach().cpu()],
                    pcl_radius_list=[0.005]
                    if "pts_r" not in self.scene_viz_args
                    else [self.scene_viz_args["pts_r"]],
                    pcl_color_list=[[0.8, 0.8, 0.8, 1.0]],
                    **self.scene_viz_args,
                )
                if save_flag:
                    imageio.imsave(save_fn, viz_full_scene)
                viz_list.append(viz_full_scene)

            # try to render a colored pcl
            save_fn = osp.join(viz_dir, viz_prefix + "_full_colored.png")
            if not osp.exists(save_fn):
                try:
                    mesh = data_dict["mesh"]
                    viz_full_scene = render(
                        mesh_list=[mesh],
                        **self.scene_viz_args,
                    )
                    if save_flag:
                        imageio.imsave(save_fn, viz_full_scene)
                    viz_list.append(viz_full_scene)
                    try:
                        mesh = data_dict["seg_mesh"]
                        viz_full_scene = render(
                            mesh_list=[mesh],
                            **self.scene_viz_args,
                        )
                        if save_flag:
                            imageio.imsave(
                                osp.join(viz_dir, viz_prefix + "_full_seg_colored.png"),
                                viz_full_scene,
                            )
                        viz_list.append(viz_full_scene)
                    except:
                        pass
                except:
                    try:
                        color = data_dict["color"]
                        if isinstance(color, torch.Tensor):
                            color = color.detach().cpu()
                        viz_full_scene = render(
                            pcl_list=[full_scene_pcl.detach().cpu()],
                            pcl_radius_list=[-1.0],
                            pcl_color_list=[color],
                            **self.scene_viz_args,
                        )
                        if save_flag:
                            imageio.imsave(save_fn, viz_full_scene)
                        viz_list.append(viz_full_scene)
                    except:
                        pass

            # try to render any predicted mask
            if "sem_pred" in data_dict.keys():
                save_fn = osp.join(viz_dir, viz_prefix + "scene_sem_input.png")
                if not osp.exists(save_fn):
                    sem_pred_dict = data_dict["sem_pred"]
                    color = cm.hsv(np.linspace(0.0, 1.0, len(sem_pred_dict) + 2)[1:-1])
                    full_scene_pcl = data_dict["pointcloud"]
                    viz_pcl_list = [full_scene_pcl[m].cpu().numpy() for m in sem_pred_dict.values()]
                    viz_sem = render(
                        pcl_list=viz_pcl_list,
                        pcl_radius_list=[0.005] * len(viz_pcl_list),
                        pcl_color_list=[c.tolist() for c in color],
                        **self.scene_viz_args,
                    )
                    if save_flag:
                        imageio.imsave(save_fn, viz_sem)
                    viz_list.append(viz_full_scene)
        except:
            logging.warning("viz input fail, pass")
        return viz_list

    def viz_em_step(
        self, ppt_dict, step: int, epi: int, model_dict, database_dict, viz_dir, viz_prefix
    ):
        if not (self.viz_iter_flag and self.viz_flag and step >= self.viz_iter_start):
            return ppt_dict

        for prior_name in self.prior_name_list:
            ppt: ProposalPoolTrajectory = ppt_dict[prior_name]
            model: CatePrior = model_dict[prior_name]
            database: Database = database_dict[prior_name]

            active_ids = ppt.active_i
            if len(active_ids) == 0:
                continue

            if self.viz_detailed_flag:
                viz_scale_base = database.mean_scale
                ppt = self.extract_shape(ppt, model, active_ids, step, viz_scale_base)
            try:
                data = ppt.fetch(active_ids, step, keys=["mesh", "W", "center", "est_pcl"])
                shape_mesh_list = data["mesh"]
            except:
                data = ppt.fetch(active_ids, step, keys=["W", "center", "est_pcl"])
                shape_mesh_list = None

            W, center, est_pcl = data["W"], data["center"], data["est_pcl"]
            viz_single_proposal(
                W=W,
                center=center,
                full_pcl=ppt.full_pcl,
                est_shape_pcl=est_pcl,
                mesh_list=shape_mesh_list,
                id_list=active_ids,
                viz_dir=osp.join(viz_dir, "step_viz"),
                step=step,
                fn_prefix=f"{viz_prefix}_{prior_name}",
                fn_postfix=f"epi_{epi}_step_{step}",
                scene_viz_cfg=self.scene_viz_args,
                viz_detailed_recon=self.viz_detailed_flag,
                **self.step_viz_args,
            )

        return ppt_dict

    def viz_results(self, ppt_list, output_path, accept_flag=True, active_flag=True):
        assert active_flag or accept_flag
        if not self.viz_flag:
            return
        accept_object_dict, active_object_dict = {}, {}
        cnt = 0
        for ppt in ppt_list:
            if accept_flag:
                l = ppt.fetch_output(include_bg=False, mode="accept")
                cnt += len(l)
                for (
                    it
                ) in (
                    l
                ):  # the returned list has mixing different cate objects (if one propr support multi semantic labels), so here decouple them to different group
                    sem = it["cate"]
                    if sem not in accept_object_dict.keys():
                        accept_object_dict[sem] = [it]
                    else:
                        accept_object_dict[sem].append(it)
            if active_flag:
                l = ppt.fetch_output(include_bg=False, mode="active")
                cnt += len(l)
                for it in l:
                    sem = it["cate"]
                    if sem not in active_object_dict.keys():
                        active_object_dict[sem] = [it]
                    else:
                        active_object_dict[sem].append(it)
        active_object_list = [v for v in active_object_dict.values()]
        accept_object_list = [v for v in accept_object_dict.values()]
        if cnt == 0:
            logging.warning("no prop for viz_results, return")
            return

        for mode in self.scene_viz_args["viz_modes"]:
            rgb_pcl, rgb_msh = viz_scene_jointly(
                background_pcl=ppt_list[0].full_pcl.detach().cpu().numpy(),
                group1_object_list=accept_object_list,
                group2_object_list=active_object_list,
                mode=mode,
                **self.scene_viz_args,
            )
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            output_fn1 = output_path + f"_{mode}_pcl.png"
            output_fn2 = output_path + f"_{mode}_msh.png"
            imageio.imsave(output_fn1, rgb_pcl)
            imageio.imsave(output_fn2, rgb_msh)
        return
