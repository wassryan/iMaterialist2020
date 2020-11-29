# !/usr/bin/env bash

echo $#
if [ $# -eq 4 ];
then
    ckpt_dir=$1
    mask_t=$2
    ckpt_idx=$3
    attr_score_thresh=$4
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

python3 test_attr.py \
    --mode test \
    --mask_thresh $mask_t \
    --attr_score_thresh $attr_score_thresh \
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
