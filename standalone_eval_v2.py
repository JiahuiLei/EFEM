# Adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py  # noqa E501
# Modified by Thang Vu
# Collected by Jiahui Lei 2022, Sep
# ! warning, different to scannet official metric, here we don't ignore the false positive that are matched to invalid labels like floor and walls
# ! version 2

from copyreg import pickle
import multiprocessing as mp
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import torch
import json
import pickle
import sys
from matplotlib import pyplot as plt


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"] = self.label_id
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id = int(data["instance_id"])
        self.label_id = int(data["label_id"])
        self.vert_count = int(data["vert_count"])
        if "med_dist" in data:
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return "(" + str(self.instance_id) + ")"


def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances


def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = " ".join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    counts = rle["counts"]
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


class ScanNetEval(object):
    def __init__(self, class_labels, min_npoint=None, iou_type=None, use_label=True):
        self.valid_class_labels = class_labels
        self.valid_class_ids = (
            np.arange(len(class_labels)) + 1
        )  # ! waring, the bg is 0, fg cate id are starting from 1
        self.id2label = {}
        self.label2id = {}
        for i in range(len(self.valid_class_ids)):
            self.label2id[self.valid_class_labels[i]] = self.valid_class_ids[i]
            self.id2label[self.valid_class_ids[i]] = self.valid_class_labels[i]

        self.ious = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        if min_npoint:
            # ! warning, note here is a parameter
            self.min_region_sizes = np.array([min_npoint])
        else:
            self.min_region_sizes = np.array([100])
            # ! Ours always produce enough points mask, but using 100 will help other baselines to get better performance since they might output les than 100 pts masks!
            # ! turn this to small numbers like 10 will decrease baseline performance, but ours stays the same, so we leave it as 100
        print("MIN_REGION_SIZE", self.min_region_sizes)
        self.distance_threshes = np.array([float("inf")])
        self.distance_confs = np.array([-float("inf")])

        self.iou_type = iou_type
        self.use_label = use_label
        if self.use_label:
            self.eval_class_labels = self.valid_class_labels
        else:
            # ! TODO: check this!!
            assert NotImplementedError("JH need to check this!")
            self.eval_class_labels = ["class_agnostic"]

    def evaluate_matches(self, matches, prcurv_save_dir, scannet_flag=False):
        ious = self.ious
        min_region_sizes = [self.min_region_sizes[0]]
        dist_threshes = [self.distance_threshes[0]]
        dist_confs = [self.distance_confs[0]]

        # results: class x iou
        ap = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), np.float)
        rc = np.zeros((len(dist_threshes), len(self.eval_class_labels), len(ious)), np.float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, iou_th in enumerate(ious):
                pred_visited = {}
                for m in matches:
                    for p in matches[m]["pred"]:
                        for label_name in self.eval_class_labels:
                            for p in matches[m]["pred"][label_name]:
                                if "filename" in p:
                                    pred_visited[p["filename"]] = False
                for li, label_name in enumerate(self.eval_class_labels):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]["pred"][label_name]
                        gt_instances = matches[m]["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["instance_id"] >= 1000
                            and gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["filename"]]:
                                    continue
                                # TODO change to use compact iou
                                iou = pred["iou"]
                                if iou > iou_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is
                                    # automatically a FP
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["filename"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match == True]  # noqa E712
                        cur_score = cur_score[cur_match == True]  # noqa E712

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                iou = gt["iou"]
                                if iou > iou_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                if (
                                    scannet_flag
                                ):  # ! here is the original scannet script, ignoring invalid classes false positive
                                    num_ignore = pred["void_intersection"]
                                    for gt in pred["matched_gt"]:
                                        # group?
                                        if gt["instance_id"] < 1000:
                                            num_ignore += gt["intersection"]
                                        # small ground truth instances
                                        if (
                                            gt["vert_count"] < min_region_size
                                            or gt["med_dist"] > distance_thresh
                                            or gt["dist_conf"] < distance_conf
                                        ):
                                            num_ignore += gt["intersection"]
                                    proportion_ignore = float(num_ignore) / pred["vert_count"]
                                    # if not ignored append false positive
                                    if proportion_ignore <= iou_th:
                                        cur_true = np.append(cur_true, 0)
                                        confidence = pred["confidence"]
                                        cur_score = np.append(cur_score, confidence)
                                else:
                                    # ! always append as FP, no ignore if this pred is matched to invalid mask, e.g. the floor and wall
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        num_true_examples = y_true_sorted_cumsum[-1]
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # recall is the first point on recall curve
                        rc_current = recall[0]

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # plot and save
                        fig = plt.figure(figsize=(15, 5))
                        plt.subplot(1, 3, 1)
                        plt.plot(recall, precision)
                        plt.plot(recall, precision, "r*")
                        plt.grid()
                        plt.xlabel("Recall")
                        plt.xlim((0.0, 1.0))
                        plt.ylabel("Precision")
                        plt.ylim((0.0, 1.0))
                        plt.title(f"PR di={di} iou={iou_th:.3f} {label_name}")

                        plt.subplot(1, 3, 2)
                        plt.plot(thresholds, precision[:-1])
                        plt.plot(thresholds, precision[:-1], "r*")
                        plt.grid()
                        plt.xlabel("conf TH")
                        plt.xlim((0.0, 1.0))
                        plt.ylabel("Precision")
                        plt.ylim((0.0, 1.0))
                        plt.title(f"P-TH di={di} iou={iou_th:.3f} {label_name}")

                        plt.subplot(1, 3, 3)
                        plt.plot(thresholds, recall[:-1])
                        plt.plot(thresholds, recall[:-1], "r*")
                        plt.grid()
                        plt.xlabel("conf TH")
                        plt.xlim((0.0, 1.0))
                        plt.ylabel("Recall")
                        plt.ylim((0.0, 1.0))
                        plt.title(f"R-TH di={di} iou={iou_th:.3f} {label_name}")

                        plt.savefig(
                            osp.join(prcurv_save_dir, f"{di}_iou={iou_th:.3f}_{label_name}.png")
                        )
                        np.savez_compressed(
                            osp.join(prcurv_save_dir, f"{di}_iou={iou_th:.3f}_{label_name}.npz"),
                            precision=precision,
                            recall=recall,
                            thresholds=thresholds,
                        )
                        plt.close()

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                        rc_current = 0.0
                    else:
                        ap_current = float("nan")
                        rc_current = float("nan")
                    ap[di, li, oi] = ap_current
                    rc[di, li, oi] = rc_current
        return ap, rc

    def compute_averages(self, aps, rcs):
        d_inf = 0
        o50 = np.where(np.isclose(self.ious, 0.5))
        o25 = np.where(np.isclose(self.ious, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.ious, 0.25)))
        avg_dict = {}
        # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
        avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
        avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
        avg_dict["all_rc"] = np.nanmean(rcs[d_inf, :, oAllBut25])
        avg_dict["all_rc_50%"] = np.nanmean(rcs[d_inf, :, o50])
        avg_dict["all_rc_25%"] = np.nanmean(rcs[d_inf, :, o25])
        avg_dict["classes"] = {}
        for li, label_name in enumerate(self.eval_class_labels):
            avg_dict["classes"][label_name] = {}
            avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
            avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
            avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
            avg_dict["classes"][label_name]["rc"] = np.average(rcs[d_inf, li, oAllBut25])
            avg_dict["classes"][label_name]["rc50%"] = np.average(rcs[d_inf, li, o50])
            avg_dict["classes"][label_name]["rc25%"] = np.average(rcs[d_inf, li, o25])
        return avg_dict

    def assign_instances_for_scan(self, preds, gts):
        """get gt instances, only consider the valid class labels even in class
        agnostic setting."""
        gt_instances = get_instances(
            gts, self.valid_class_ids, self.valid_class_labels, self.id2label
        )
        # associate
        if self.use_label:
            gt2pred = deepcopy(gt_instances)
            for label in gt2pred:
                for gt in gt2pred[label]:
                    gt["matched_pred"] = []

        else:
            gt2pred = {}
            agnostic_instances = []
            # concat all the instances label to agnostic label
            for _, instances in gt_instances.items():
                agnostic_instances += deepcopy(instances)
            for gt in agnostic_instances:
                gt["matched_pred"] = []
            gt2pred[self.eval_class_labels[0]] = agnostic_instances

        pred2gt = {}
        for label in self.eval_class_labels:
            pred2gt[label] = []
        num_pred_instances = 0
        # mask of void labels in the groundtruth
        bool_void = np.logical_not(np.in1d(gts // 1000, self.valid_class_ids))
        # go thru all prediction masks
        for pred in preds:
            if self.use_label:
                label_id = pred["label_id"]
                if label_id not in self.id2label:
                    continue
                label_name = self.id2label[label_id]
            else:
                label_name = self.eval_class_labels[0]  # class agnostic label
            conf = pred["conf"]
            pred_mask = pred["pred_mask"]
            # pred_mask can be np.array or rle dict
            if isinstance(pred_mask, dict):
                pred_mask = rle_decode(pred_mask)
            assert (
                pred_mask.shape[0] == gts.shape[0]
            ), f"pred={pred_mask.shape[0]} but gt={gts.shape[0]}"

            # convert to binary
            pred_mask = np.not_equal(pred_mask, 0)
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue  # skip if empty

            pred_instance = {}
            pred_instance["filename"] = "{}_{}".format(pred["scan_id"], num_pred_instances)  # dummy
            pred_instance["pred_id"] = num_pred_instances
            pred_instance["label_id"] = label_id if self.use_label else None
            pred_instance["vert_count"] = num
            pred_instance["confidence"] = conf
            pred_instance["void_intersection"] = np.count_nonzero(
                np.logical_and(bool_void, pred_mask)
            )

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for gt_num, gt_inst in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(
                    np.logical_and(gts == gt_inst["instance_id"], pred_mask)
                )
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy["intersection"] = intersection
                    pred_copy["intersection"] = intersection
                    iou = float(intersection) / (
                        gt_copy["vert_count"] + pred_copy["vert_count"] - intersection
                    )
                    gt_copy["iou"] = iou
                    pred_copy["iou"] = iou
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
            pred_instance["matched_gt"] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    def print_results(self, avgs):
        sep = ""
        col1 = ":"
        lineLen = 64

        print()
        print("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>8}".format("AP") + sep
        line += "{:>8}".format("AP_50%") + sep
        line += "{:>8}".format("AP_25%") + sep
        line += "{:>8}".format("AR") + sep
        line += "{:>8}".format("RC_50%") + sep
        line += "{:>8}".format("RC_25%") + sep

        print(line)
        print("#" * lineLen)

        for li, label_name in enumerate(self.eval_class_labels):
            ap_avg = avgs["classes"][label_name]["ap"]
            ap_50o = avgs["classes"][label_name]["ap50%"]
            ap_25o = avgs["classes"][label_name]["ap25%"]
            rc_avg = avgs["classes"][label_name]["rc"]
            rc_50o = avgs["classes"][label_name]["rc50%"]
            rc_25o = avgs["classes"][label_name]["rc25%"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>8.3f}".format(ap_avg) + sep
            line += sep + "{:>8.3f}".format(ap_50o) + sep
            line += sep + "{:>8.3f}".format(ap_25o) + sep
            line += sep + "{:>8.3f}".format(rc_avg) + sep
            line += sep + "{:>8.3f}".format(rc_50o) + sep
            line += sep + "{:>8.3f}".format(rc_25o) + sep
            print(line)

        all_ap_avg = avgs["all_ap"]
        all_ap_50o = avgs["all_ap_50%"]
        all_ap_25o = avgs["all_ap_25%"]
        all_rc_avg = avgs["all_rc"]
        all_rc_50o = avgs["all_rc_50%"]
        all_rc_25o = avgs["all_rc_25%"]

        print("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>8.3f}".format(all_ap_avg) + sep
        line += "{:>8.3f}".format(all_ap_50o) + sep
        line += "{:>8.3f}".format(all_ap_25o) + sep
        line += "{:>8.3f}".format(all_rc_avg) + sep
        line += "{:>8.3f}".format(all_rc_50o) + sep
        line += "{:>8.3f}".format(all_rc_25o) + sep
        print(line)
        print("#" * lineLen)
        print()

    def print_fb_results(self, REPORT):
        print("=" * 64)
        for label in self.valid_class_labels:
            print(f"OBJ-IOU {label} mean={REPORT['obj_iou'][label]:.3f}")
        print("- " * 32)
        for label in self.valid_class_labels:
            print(f"F-B-ACC {label} mean={REPORT['fb_acc'][label]:.3f}")
        print("=" * 64)
        print()
        return

    def write_result_file(self, avgs, filename):
        _SPLITTER = ","
        with open(filename, "w") as f:
            f.write(_SPLITTER.join(["class", "class id", "ap", "ap50", "ap25"]) + "\n")
            for class_name in self.eval_class_labels:
                ap = avgs["classes"][class_name]["ap"]
                ap50 = avgs["classes"][class_name]["ap50%"]
                ap25 = avgs["classes"][class_name]["ap25%"]
                f.write(_SPLITTER.join([str(x) for x in [class_name, ap, ap50, ap25]]) + "\n")

    def evaluate(self, pred_list, gt_list, prcurv_save_dir, scannet_flag=False):
        """
        Args:
            pred_list:
                for each scan:
                    for each instance
                        instance = dict(scan_id, label_id, mask, conf)
            gt_list:
                for each scan:
                    for each point:
                        gt_id = class_id * 1000 + instance_id
        """

        pool = mp.Pool()
        results = pool.starmap(self.assign_instances_for_scan, zip(pred_list, gt_list))
        pool.close()
        pool.join()

        matches = {}
        for i, (gt2pred, pred2gt) in enumerate(results):
            matches_key = f"gt_{i}"
            matches[matches_key] = {}
            matches[matches_key]["gt"] = gt2pred
            matches[matches_key]["pred"] = pred2gt
        ap_scores, rc_scores = self.evaluate_matches(
            matches, prcurv_save_dir=prcurv_save_dir, scannet_flag=scannet_flag
        )
        avgs = self.compute_averages(ap_scores, rc_scores)

        # print
        self.print_results(avgs)
        return avgs


def load_scannet_format_results(load_dir, semantic_fn=""):
    # https://kaldir.vc.in.tum.de/scannet_benchmark/documentation#format-instance3d
    if len(semantic_fn) == 0:
        semantic_fn = osp.join(load_dir, "semantic.npz")
    semantic_data = np.load(semantic_fn, allow_pickle=True)
    class_name = semantic_data["class_names"]
    print(f"load sem from semantic.npz")
    print(class_name)
    # ! in the semantic npz file, "class_names" is required, "nyu_id" is optional
    if "nyu_id" in semantic_data.files:
        nyu_id = semantic_data["nyu_id"].tolist()  # scannet use nyu id as label for eval
    else:
        nyu_id = None

    mask_results_dir = osp.join(load_dir, "scannet_format")
    recovered_pred_insts = []
    scan_id_list = [f[:-4] for f in os.listdir(mask_results_dir) if f.endswith(".txt")]
    scan_id_list.sort()
    args = []
    for scan_id in tqdm(scan_id_list):
        meta_fn = osp.join(mask_results_dir, f"{scan_id}.txt")
        args.append((scan_id, meta_fn, nyu_id))

    print("loading sem")
    try:
        sem_pred = []
        for scan_id in tqdm(scan_id_list):
            fn = osp.join(load_dir, "semantic_pred", f"{scan_id}.npy")
            data = np.load(fn)
            sem_pred.append(data)
    except:
        print("Warning, can't load sem pred, skip sem loading")
        sem_pred = None

    # pool = mp.Pool()
    # ret = pool.map(__load_single_scannet_format, args)
    # pool.close()
    # # sort
    # ret_scan_id = [r[0]["scan_id"] for r in ret]
    # for scan_id in scan_id_list:
    #     recovered_pred_insts.append(ret[ret_scan_id.index(scan_id)])

    for arg in tqdm(args):
        recovered_pred_insts.append(__load_single_scannet_format(arg))

    return scan_id_list, recovered_pred_insts, class_name, nyu_id, sem_pred
    # !the recover_pred_insts lable are form 1-K, not in nyu format


def __load_single_scannet_format(args):
    scan_id, meta_fn, nyu_id = args

    with open(meta_fn, "r") as f:
        meta_info = f.readlines()
    load_dir = osp.dirname(meta_fn)
    results = []
    for line in meta_info:
        mask_fn, label, score = line.split("\n")[0].split(" ")
        label, score = int(label), float(score)
        if nyu_id is not None:
            # * convert from nyu id back to compact 1-K id
            label = nyu_id.index(label) + 1
        mask_fn = osp.join(load_dir, mask_fn)
        if osp.exists(mask_fn + ".npy"):
            mask = np.load(mask_fn + ".npy")
        else:
            with open(mask_fn, "r") as f:
                data = f.readlines()
            mask = [int(l[0]) for l in data]
            mask = np.asarray(mask, dtype=np.uint8)
        # mask = np.loadtxt(mask_fn, dtype=np.uint8)
        results.append({"scan_id": scan_id, "label_id": label, "conf": score, "pred_mask": mask})
    return results


def load_gt_from_scannet_preprocessed(
    gt_dir,
    semantic_classes=20,  # mugs: 2
    instance_classes=18,  # mugs: 1
    postfix="_inst_nostuff.pth",  # mugs: .pth
):
    # ! the data format follow the scannet preprocessing of softgroup and dknet etc
    scan_id_list = [f[: -len(postfix)] for f in os.listdir(gt_dir) if f.endswith(postfix)]
    scan_id_list.sort()
    ins_ret = []
    sem_ret = []
    for scan_id in tqdm(scan_id_list):
        data_fn = osp.join(gt_dir, f"{scan_id}{postfix}")
        _, _, sem, ins = torch.load(data_fn)
        gt = get_gt_instances(
            sem, ins, semantic_classes=semantic_classes, instance_classes=instance_classes
        ).astype(np.int32)
        ins_ret.append(gt)
        sem_ret.append(sem.astype(np.int8))
    return scan_id_list, ins_ret, sem_ret


def get_gt_instances(semantic_labels, instance_labels, semantic_classes=20, instance_classes=18):
    # convert to evaluation format 0: ignore, 1->N: valid
    label_shift = semantic_classes - instance_classes
    semantic_labels = semantic_labels - label_shift + 1
    semantic_labels[semantic_labels < 0] = 0
    instance_labels += 1
    ignore_inds = instance_labels < 0
    # scannet encoding rule
    gt_ins = semantic_labels * 1000 + instance_labels
    gt_ins[ignore_inds] = 0
    return gt_ins


def evaluate_semantic_acc(pred_list, gt_list, ignore_label=-100):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    assert gt.shape == pred.shape
    correct = (gt[gt != ignore_label] == pred[gt != ignore_label]).sum()
    whole = (gt != ignore_label).sum()
    acc = correct.astype(float) / whole * 100
    print(f"Total Acc: {acc:.1f}")
    return acc


def evaluate_semantic_miou(pred_list, gt_list, class_names, ignore_label=-100):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    pos_inds = gt != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    assert gt.shape == pred.shape
    iou_list = []
    for _index in np.unique(gt):
        if _index != ignore_label:
            intersection = ((gt == _index) & (pred == _index)).sum()
            union = ((gt == _index) | (pred == _index)).sum()
            iou = intersection.astype(float) / union * 100
            iou_list.append(iou)
    miou = np.mean(iou_list)
    for _iou, name in zip(iou_list, class_names):
        print(f"{name} mIoU: {_iou:.1f}")
    print(f"mIoU: {miou:.1f}")
    return miou, iou_list


def eval_main(
    gt_dir,
    results_dir,
    eval_min_npoint=None,
    semantic_classes=2,
    instance_classes=1,
    postfix=".pth",
    strict=True,
    semantic_fn="",
    scannet_flag=False,
):
    # eval_min_npoint defaulat is 100 in ScanNetEval
    if scannet_flag:
        print("Ignoring the invalid background, following the original SCANNET eval script")

    # * load gt
    loaded_gt_scan_id_list, gt_insts, sem_labels = load_gt_from_scannet_preprocessed(
        gt_dir,
        semantic_classes=semantic_classes,
        instance_classes=instance_classes,
        postfix=postfix,
    )
    # * load the prediction from scannet format
    (
        loaded_results_scan_id_list,
        pred_insts,
        CLASSES_NAME,
        NYU_ID,
        sem_preds,
    ) = load_scannet_format_results(load_dir=results_dir, semantic_fn=semantic_fn)
    if strict:
        assert loaded_gt_scan_id_list == loaded_results_scan_id_list
    else:
        print(
            f"WARNING, ONLY EVALUATE PARTIAL DATASET! {len(loaded_results_scan_id_list)}/{len(loaded_gt_scan_id_list)}"
        )
        _gt_insts = []
        for scan_id in loaded_results_scan_id_list:
            _gt_insts.append(gt_insts[loaded_gt_scan_id_list.index(scan_id)])
        gt_insts = _gt_insts
        loaded_gt_scan_id_list = loaded_results_scan_id_list

    # * eval
    # * semantic

    sem_class_name = [
        f"label_{i}" for i in range(semantic_classes - instance_classes)
    ] + CLASSES_NAME.tolist()
    if sem_preds is not None:
        miou, sem_iou_list = evaluate_semantic_miou(
            sem_preds, sem_labels, sem_class_name, ignore_label=-100
        )
        acc = evaluate_semantic_acc(sem_preds, sem_labels, ignore_label=-100)
    else:
        miou, acc = np.nan, np.nan

    # * instance
    scannet_eval = ScanNetEval(CLASSES_NAME, eval_min_npoint)

    prcurv_save_dir = osp.join(results_dir, "pr_curve")
    os.makedirs(prcurv_save_dir, exist_ok=True)
    avgs = scannet_eval.evaluate(
        pred_insts, gt_insts, prcurv_save_dir=prcurv_save_dir, scannet_flag=scannet_flag
    )

    # * save detailed report to results dir

    out_fn = osp.join(results_dir, "results.txt")
    if osp.exists(out_fn):
        os.rename(out_fn, out_fn + ".bck")
    with open(out_fn, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("-" * 20 + "instnace" + "-" * 20)
        scannet_eval.print_results(avgs)

        if sem_preds is not None:
            print("-" * 20 + "semantic" + "-" * 20)
            for _iou, name in zip(sem_iou_list, sem_class_name):
                print(f"{name} mIoU: {_iou:.1f}")
            print(f"mIoU: {miou:.1f}")

        sys.stdout = original_stdout

    # * save a xls report
    import pandas as pd

    report = {
        "name": ["ave"],
        "sem_iou": [miou],
        "ap": [avgs["all_ap"]],
        "ap50%": [avgs["all_ap_50%"]],
        "ap25%": [avgs["all_ap_25%"]],
        "rc": [avgs["all_rc"]],
        "rc50%": [avgs["all_rc_50%"]],
        "rc25%": [avgs["all_rc_25%"]],
        "sem_acc": [acc],
    }

    per_class_ins = avgs["classes"]
    for _id, k in enumerate(sem_class_name):
        report["name"].append(k)
        if sem_preds is not None:
            report["sem_iou"].append(sem_iou_list[_id])
        else:
            report["sem_iou"].append(None)
        report["sem_acc"].append(None)
        if k in per_class_ins.keys():
            for metric_name, value in per_class_ins[k].items():
                report[metric_name].append(value)
        else:
            n = len(report["name"])
            for k, v in report.items():
                if len(v) < n:
                    assert len(v) + 1 == n, f"{k} {len(v)}+1 != {n}"
                    report[k].append(None)

    # for k, v in report.items():
    #     print(f"{k} len={len(v)}")
    out_fn = osp.join(results_dir, "results.xlsx")
    df = pd.DataFrame(report)
    df.to_excel(out_fn)

    return


if __name__ == "__main__":
    import argparse

    """
    IMPORTANT

    The saved prediction's semantic label either
        a.) start from 1, which corresponds to the semantic.npz ["class_names"] file list [0]
        b.) corresponds to NYU id for scannet, need to translate back to compact 1-K id for eval
    The predicted_masks can either contain standard scannet format txt (which is quite slow) or npy files,
    whose name is just xxxx.txt.npy, NOTE, the name in the scene_id.txt description stays the scannet format,
    i.e. the file path is .txt/
    """
    parser = argparse.ArgumentParser("SoftGroup")
    parser.add_argument(
        "--results_dir",
        required=True,
        type=str,
        help="This dir must contain a dir named scannet_format and a file called semantic.npz",
    )
    parser.add_argument(
        "--gt_dir",
        required=True,
        type=str,
        help="The (SoftGroup/DKNet) preprocessed dataset dir",
    )
    parser.add_argument(
        "--n_sem",
        required=True,
        type=int,
        help="number of sem classes including bg like floor or walls, Scannet=20, Sapien=2 or larger",
    )
    parser.add_argument(
        "--n_ins",
        required=True,
        type=int,
        help="number of sem classes of fg instances, Scannet=18, Sapien=1 (mugs) or larger",
    )
    parser.add_argument(
        "--postfix",
        default=".pth",
        type=str,
        help="the post fix of the dataset, scannet=_inst_nostuff.pth, default sapien is .pth",
    )
    parser.add_argument(
        "--non_strict",
        default=False,
        action="store_true",
        help="if set, only eval the existed results, not require the whole dataset are finished",
    )
    parser.add_argument(
        "--semantic_fn",
        default="",
        type=str,
        help="a path to .npz file that has a file called classes",
    )
    parser.add_argument(
        "--scannet_flag",
        default=False,
        action="store_true",
        help="Whether to ignore invalid background, if on scannet set true, otherwise leave it false",
    )
    args = parser.parse_args()

    eval_main(
        results_dir=args.results_dir,
        gt_dir=args.gt_dir,
        semantic_classes=args.n_sem,
        instance_classes=args.n_ins,
        postfix=args.postfix,
        strict=not args.non_strict,
        semantic_fn=args.semantic_fn,
        scannet_flag=args.scannet_flag,
    )

    # debug
    # eval_main(
    #     results_dir="/home/ray/projects/SoftGroup/work_dirs/debug/official_scannet/eval_ins_seg_results",
    #     gt_dir="/home/ray/datasets/ScanNet/DKNet/val",
    #     semantic_classes=20,
    #     instance_classes=18,
    #     postfix="_inst_nostuff.pth",
    # )
