# iMaterialist-2020
This is a segmentation framework modified from *MaskRCNN*.

## requirements
- Python (>= 3.6)
- PyTorch
`pip3 install -r requirements.txt `

## Demo
we provide a clothing parsing system using our trained model as segmentation model.
![image](https://github.com/wassryan/iMaterialist2020/blob/master/assets/demo.png)

## Train
This code support two types of model: `without attribute` and `with attribute`
```
./train.sh (lr schedule) (lr) # train without attr
#./train_attr.sh (attribute-wise weight) (attribute-loss weight) (lr) # train with attr
```

## Val
*predict on validation data (output mIOU and mF1)*
```
./val.sh (experiment_name) (checkpoint_idx)
```

## Test
*test without attribute*
```
# ./test.sh experiment_name mask_thresh checkpoint_idx
./test.sh torch_MaskRCNN_40e_lr0.1 0.5 21
```

*test with attribute*
```
# ./test_attr.sh experiment_name mask_thresh checkpoint_idx attr_score_thresh
./test_attr.sh torch_MaskRCNN_20e_lr0.01_attr_1000weight_3aweight 0.5 0 0.7
```

## code log
`train.py` # train with train/val.csv, not support for online validation (validate while training)。[something wrong in code, seems to be the cuda device incorrespondence]
`train_attr.py` # train with attribute data，using `binary cross entropy` with specified `pos weight` and `loss weight`.