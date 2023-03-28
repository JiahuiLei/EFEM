from genericpath import exists
import torch
import logging
import numpy as np
import os


def make_bbox_8pts(basis, bbox):
    bbox8pts_weight = torch.Tensor(
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
    device = bbox.device
    bbox_pts_coordi = bbox8pts_weight.to(device)[None, ...] * bbox[:, None, :]  # B,8,3
    bbox_pts = (bbox_pts_coordi[..., None] * basis[:, None, ...]).sum(2)
    return bbox_pts  # B,8,3


class ProposalPoolTrajectory:
    def __init__(
        self,
        name,
        full_pcl,
        full_nrm=None,
        hard_bg_mask=None,
        verbose=True,
        full_trace_flag=False,
        segment=None,
    ) -> None:
        self.name = name
        self.full_pcl = full_pcl
        self.full_nrm = full_nrm
        # set hard background mask
        if hard_bg_mask is None:  # if is bg, = 1
            self.hard_bg_mask = torch.zeros_like(full_pcl[:, 0]).bool()
        else:
            self.hard_bg_mask = torch.as_tensor(hard_bg_mask).bool().to(full_pcl.device)
        # set occupied mask
        self.assigned_mask = torch.zeros_like(full_pcl[:, 0]).bool()
        self.proposal_traj_list, self.active_list, self.accept_list = [], [], []
        self.verbose = verbose
        self.full_trace_flag = full_trace_flag
        
        self.segment = segment
        if self.segment is not None:
            self.segment = self.segment.to(self.full_pcl.device)

    def __getitem__(self, pid):
        # get item by pid
        for it in self.proposal_traj_list:
            if it["id"] == int(pid):
                return it
        raise IndexError(f"Pid={int(pid)} not found in PPT={self.name}")

    def __index_pid(self, pid):
        for ind, it in enumerate(self.proposal_traj_list):
            if it["id"] == int(pid):
                return ind
        return None  # not found

    @property
    def unavailable_mask(self):  # either hard bg or assigned
        return torch.logical_or(self.hard_bg_mask, self.assigned_mask)

    @property
    def available_mask(self):  # either hard bg or assigned
        return ~self.unavailable_mask

    def append_new_proposals(self, W: torch.Tensor, epi: int, ids: list, nn_r_list: list):
        assert isinstance(W, torch.Tensor)
        for w, id, nn_r in zip(W, ids, nn_r_list):
            self.proposal_traj_list.append({"id": id, "epi": epi, "nn_r": nn_r, "traj": [{"W": w}]})
            self.active_list.append(True)
            self.accept_list.append(False)
        if self.verbose:
            logging.info(f"{self.name} PPT: append {len(W)} new proposals")
        return

    def update(self, ids, step: int, **kwargs):
        # at each call, either all appending new step or all modifying existing record
        for ind in range(len(ids)):
            pid = int(ids[ind])
            len_traj = len(self[pid]["traj"])
            if step < len_traj:  # update existing record
                assert "W" not in kwargs.keys(), "W should append to new record, can't modify"
                for k, v in kwargs.items():
                    vi = v[ind]
                    if isinstance(vi, torch.Tensor):
                        vi = vi.detach().cpu()
                    self[pid]["traj"][step][k] = vi
            elif step == len_traj:  # append a new record
                assert "W" in kwargs.keys(), "When appending new record, must have W"
                self[pid]["traj"][-1]["W"] = (
                    self[pid]["traj"][-1]["W"].detach().cpu()
                )  # move latest W to cpu, let the newly added W to stay on GPU
                self[pid]["traj"].append(dict())
                for k, v in kwargs.items():
                    vi = v[ind]
                    if isinstance(vi, torch.Tensor) and k != "W":
                        vi = vi.detach().cpu()  # the latest W should not be detached
                    self[pid]["traj"][-1][k] = vi
            else:
                raise IndexError("step output index")
        return

    def clean(self, ids, step: int):
        # if not full traj, clean the buffer before the step
        if self.full_trace_flag or step < 0:
            return
        for pid in ids:
            for i in range(step):
                old_content = self[pid]["traj"][i]
                self[pid]["traj"][i] = None
                del old_content

    def delete(self, pid):
        for ind, it in enumerate(self.proposal_traj_list):
            if it["id"] == int(pid):
                try:
                    del self.proposal_traj_list[ind]["traj"][-1]["W"]
                except:
                    logging.warning("PPT internal warning: the last W may not be removed properly")
                self.proposal_traj_list[ind]["traj"] = []
                return
        raise IndexError(f"Pid={int(pid)} not found in PPT={self.name}")

    def update_assignment(self, mask):
        # for sync invalid mask across ppt
        self.assigned_mask = torch.logical_or(
            self.assigned_mask, mask.to(self.assigned_mask.device)
        )
        return

    def log_decision(self, ids, reason=None, mode="reject", assignment=None, score_dict=None):
        assert mode in ["accept", "reject"]
        if mode == "accept":
            assignment = assignment.to(self.assigned_mask.device)
            assert assignment is not None, "when logging acceptance, need the assignment"
            # assert (assignment.sum(0) <= 1).all()
            # ! warning, there is no gaurantee now that two proposals are not overlapped
            assert (assignment.any(0) + self.assigned_mask <= 1).all()
            self.assigned_mask = torch.logical_or(self.assigned_mask, assignment.any(0))
        for ind, pid in enumerate(ids):
            if self.__index_pid(pid) is None:
                continue  # not in this proposal
            self.active_list[self.__index_pid(pid)] = False
            if reason is not None:
                if "reason" not in self[pid].keys():
                    self[pid]["reason"] = ""
                if isinstance(reason, list):
                    self[pid]["reason"] += f"_{reason[ind]}"
                else:
                    self[pid]["reason"] += f"_{reason}"
            if mode == "accept":
                self.accept_list[self.__index_pid(pid)] = True
                self[pid]["output"] = assignment[ind].detach().cpu()
                if score_dict is not None:
                    self[pid]["score"] = {k: v[ind] for k, v in score_dict.items()}
                self[pid]["sem"] = self[pid]["traj"][-2]["sem"]
            if not self.full_trace_flag and mode == "reject":
                # clean the buffer
                self.delete(pid)
        if self.verbose:
            logging.info(f"{self.name} PPT: {mode} {ids} proposals because {reason}")
        return

    # * Retrieval Methods
    def fetch(self, ids, step: int, keys: list):
        ret = {}
        for k in keys:
            buffer = []
            for pid in ids:
                item = self[pid]["traj"][step][k]
                buffer.append(item)
            if isinstance(buffer[0], torch.Tensor):
                buffer = torch.stack(buffer, 0)
            ret[k] = buffer
        return ret

    def fetch_nn_r(self, ids):
        nn_r_list = [self[pid]["nn_r"] for pid in ids]        
        nn_r = torch.Tensor(nn_r_list).to(self.full_pcl.device)
        return nn_r
    
    @property
    def active_active_latest_w_id(self):
        return self._fetch_active_latest()

    @property
    def active_i(self):
        return self._fetch_active_latest(False)

    @property
    def has_active(self):
        return any(self.active_list)

    def _fetch_active_latest(self, get_w=True):
        W, id = [], []
        for p, active in zip(self.proposal_traj_list, self.active_list):
            if active:
                if get_w:
                    W.append(p["traj"][-1]["W"])  # fetch the latest weight
                id.append(p["id"])
        id = torch.LongTensor(id)
        if get_w:
            if len(id) == 0:
                W = torch.zeros(0, len(self.full_pcl)).to(self.full_pcl.device)
                return W, id
            else:
                W = torch.stack(W, 0)
                return W, id
        else:
            return id

    def fetch_output(self, include_bg=True, mode="accept"):
        assert mode in ["accept", "active"]
        # return a list of dict including the background
        if include_bg:
            ret = [
                {
                    "instance": -1,
                    "cate": "background",
                    "pcl": self.full_pcl[~self.assigned_mask].detach().cpu().numpy(),
                    "mask": (~self.assigned_mask).detach().cpu().numpy(),
                }
            ]
        else:
            ret = []
        for ind in range(len(self.proposal_traj_list)):
            mode_list = self.accept_list if mode == "accept" else self.active_list
            if mode_list[ind]:

                record = self.proposal_traj_list[ind]["traj"][-2]
                basis = record["basis"]
                center = record["center"]
                bbox_c = record["bbox_c"]
                bbox = record["bbox"]
                bbox_8pts = (
                    make_bbox_8pts(basis.transpose(1, 0)[None, ...], bbox[None, ...]) + center
                )
                if mode == "acccept":
                    mask = self.proposal_traj_list[ind]["output"]
                else:
                    mask = record["tmp_assignment"]
                object_pcl = self.full_pcl[mask]
                try:
                    score_dict = self.proposal_traj_list[ind]["score"]
                except:
                    score_dict = {}
                ret.append(
                    {
                        "instance": self.proposal_traj_list[ind]["id"],
                        # "cate": self.name,
                        # ! note here not use the sem outside, use the one in the traj -2
                        "cate": self.proposal_traj_list[ind]["traj"][-2]["sem"],
                        "B": basis.transpose(-2, -1).detach().cpu().numpy(),
                        "t": (center + bbox_c[None, :]).detach().cpu().numpy(),
                        "bbox": bbox.detach().cpu().numpy(),
                        "bbox_8pts": bbox_8pts[0].detach().cpu().numpy(),
                        "pcl": object_pcl.detach().cpu().numpy(),
                        "mesh_world": record["mesh"],
                        "mask": mask,
                        "score": score_dict,
                    }
                )
        return ret

    # def export_results(self, fn):
    #     raise DeprecationWarning("Not use this")
    #     # only save the results
    #     os.makedirs(os.path.dirname(fn), exist_ok=True)
    #     output = self.fetch_output()
    #     np.savez_compressed(fn, output=output)
    #     masks = np.stack([o["mask"] for o in output], 0)  # The first is always the
    #     return output, masks[1:], masks[:1]

    # def fetch_output_masks(self):
    #     output = self.fetch_output()
    #     masks = np.stack([o["mask"] for o in output], 0)  # The first is always the
    #     return masks[1:], masks[:1]

    # def print():
    #     return
