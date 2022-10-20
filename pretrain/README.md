### pretrain_vg : Model pretraining for Visual Geo-localization

This repository is part of the framework proposed in our work "Deep Visual Geo-localization". This module can be used to pretrain the backbones on:

* [Google Landmark v2 (GLDv2)](https://github.com/cvdfoundation/google-landmark)
* [Places365](http://places2.csail.mit.edu)

---

### Usage

#### Training on GLDv2
For GLDv2, the backbone used in our work were trained for 30 epochs on the full GLDv2 training set.
To launch the training procedure use : 
 
  ```
 python3 train_gldv2.py [-h] [--device {cuda,cpu}] [--exp_name EXP_NAME]
                      [--gldv2_csv GLDV2_CSV]
                      [--data_path DATA_PATH]
                      [--train_batch_size TRAIN_BATCH_SIZE]
                      [--seed SEED]
                      [--num_workers NUM_WORKERS]
                      [--resize_shape RESIZE_SHAPE RESIZE_SHAPE]
                      [--loss_module {arcface,}] [--arcface_s ARCFACE_S]
                      [--arcface_margin ARCFACE_MARGIN]
                      [--arcface_ls_eps ARCFACE_LS_EPS]
                      [--epochs_num EPOCHS_NUM] [--patience PATIENCE]
                      [--lr LR] [--optim {adam,sgd}]
                      [--arch {vgg16,r18,r50,r101}] [--resume RESUME]
  ```
  For more details use
  ``` python3 train_gldv2.py -h ```


#### Training on Places365
For Places365, the backbone used in our work were trained until convergence using early stopping with patience equals to `5`.  
   
```
python3 train_places.py [-h] [--device {cuda,cpu}] [--exp_name EXP_NAME]
                       [--data_path DATA_PATH]
                       [--train_batch_size TRAIN_BATCH_SIZE]
                       [--eval_batch_size EVAL_BATCH_SIZE] [--seed SEED]
                       [--num_workers NUM_WORKERS]
                       [--resize_shape RESIZE_SHAPE RESIZE_SHAPE]
                       [--epochs_num EPOCHS_NUM] [--patience PATIENCE]
                       [--lr LR] [--optim {adam,sgd}]
                       [--arch {vgg16,r18,r50,r101}] [--resume RESUME]  
  ```                     
  For more details use
  ``` python3 train_places.py -h ```
