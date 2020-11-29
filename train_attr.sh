
# please specify pos_weight, loss_weight, lr in shell cmd

echo $#
if [ $# -eq 3 ];
then
    pos_weight=$1
    loss_weight=$2
    lr=$3
else
    echo "usage: cmd (attribute-wise weight) (attribute-loss weight) (lr)"
    exit
fi

root='/data/experiments/fashion/attr_'$1'posweight_'$2'weight_'$3'lr'
log=$root'/logs/'
save=$root'/save/'
out=$root'/out/'

echo $root

set -x

python3 train_attr.py \
    --mode='train' \
    --batch_size 8 \
    --lr 0.001 \
    --num_epochs 30 \
    --rep_intv 1000 \
    --log_dir  $log \
    --save_dir $save \
    --out_dir $out \
    --pos_weight 1000. \
    --loss_aweight 3.

# batch_size=64 OOM