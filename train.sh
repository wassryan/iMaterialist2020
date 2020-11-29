

echo $#
if [ $# -eq 2 ];
then
    schedule=$1
    lr=$2
else
    echo "usage: cmd (lr schedule) (lr)"
    exit
fi

root='/data/experiments/fashion/woattr_'$1'schedule_'$2'lr'
log=$root'/logs/'
save=$root'/save/'
out=$root'/out/'

echo $root

set -x

python3 train.py \
    --mode='train' \
    --batch_size 8 \
    --lr 0.01 \
    --scheduler $schedule \
    --num_epochs 40 \
    --rep_intv 1000 \
    --log_dir  $log \
    --save_dir $save \
    --out_dir $out

# batch_size=64 OOM