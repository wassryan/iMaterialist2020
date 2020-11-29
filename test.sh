# !/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0,1 python3 test.py --mode=test  --width=512 --height=512

# save='/data/experiments/fashion/torch_MaskRCNN2/save/'
# out='/data/experiments/fashion/torch_MaskRCNN2/out/'
# submit='/data/experiments/fashion/torch_MaskRCNN2/'

echo $#
if [ $# -eq 3 ];
then
    ckpt_dir=$1
    mask_t=$2
    ckpt_idx=$3
else
    echo "usage: cmd ckpt_dir mask_thresh ckpt_idx"
    exit
fi

echo $ckpt_dir
echo $mask_t
echo $ckpt_idx

submit='/data/experiments/fashion/'$ckpt_dir'/'
save=$submit'save/'
out=$submit'out/'
echo $submit

set -x

python3 test.py \
    --mode test \
    --mask_thresh $mask_t \
    --rep_intv 50 \
    --width 512 \
    --height 512 \
    --ann_path /data/imaterialist2020/train.csv \
    --test_dir /data/imaterialist2020/test/ \
    --sample_path /data/imaterialist2020/sample_submission.csv \
    --submit_path $submit \
    --save_dir $save \
    --out_dir $out \
    --checkpoint $ckpt_idx
