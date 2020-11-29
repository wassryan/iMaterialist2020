# !/usr/bin/env bash

echo $#
if [ $# -eq 2 ];
then
    ckpt_dir=$1
    ckpt_idx=$2
else
    echo "usage: cmd ckpt_dir ckpt_idx"
    exit
fi

echo $ckpt_dir
echo $ckpt_idx

submit='/data/experiments/fashion/'$ckpt_dir'/'
save=$submit'save/'
out=$submit'out/'
echo $submit

set -x

python3 val.py \
    --mode test \
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
