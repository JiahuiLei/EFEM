# source activate efem
which python

# python efem.py --config ./configs/real_mugs_tree.yaml
# python standalone_eval_v2.py --results_dir ./log/real_mugs_tree/results_eval --gt_dir ./data/chairs_and_mugs/real_mugs_tree_normal_pth/test --n_sem 2 --n_ins 1 --postfix .pth

declare -a CFGArray=("real_chairs_z" "real_chairs_pile" "real_chairs_so3" "real_mugs_pile" "real_mugs_so3" "real_mugs_z" "real_mugs_wild" "real_mugs_others" "real_mugs_tree")
for i in "${!CFGArray[@]}"; do
    cfg=${CFGArray[i]}
    echo "cfg" $cfg
    python efem.py --config ./configs/$cfg.yaml
    python standalone_eval_v2.py --results_dir ./log/$cfg/results_eval --gt_dir ./data/chairs_and_mugs/"$cfg"_normal_pth/test --n_sem 2 --n_ins 1 --postfix .pth
done

declare -a CFGArray=("kit4cates_novel_pile" "chairs_novel_pile" "mugs_novel_tree"  "kit4cates_uniform_so3" "kit4cates_uniform_z" "chairs_uniform_z" "chairs_uniform_so3"  "mugs_novel_pile" "mugs_novel_box" "mugs_novel_shelf" "mugs_uniform_so3" "mugs_uniform_z")
# declare -a CFGArray=("mugs_uniform_z")
for i in "${!CFGArray[@]}"; do
    cfg=${CFGArray[i]}
    echo "cfg" $cfg
    python efem.py --config ./configs/sapien_"$cfg".yaml
    python standalone_eval_v2.py --results_dir ./log/sapien_"$cfg"/results_eval --gt_dir ./data/sapien/"$cfg"_normal/test --n_sem 2 --n_ins 1 --postfix .pth
done

# python standalone_eval_v2.py --results_dir ./log/scannet_val/results_eval \
#     --gt_dir ./data/scannet/val --n_sem 20 --n_ins 18 --postfix _inst_nostuff.pth --scannet_flag