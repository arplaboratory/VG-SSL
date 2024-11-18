# VG-SSL

This is the official repository for [VG-SSL: Benchmarking Self-supervised Representation Learning Approaches for Visual Geo-localization](). VG-SSL is designed to facilitate research and development in geo-localization by providing a robust framework for evaluating self-supervised learning approaches.

## Prerequisites

Before you begin, ensure you have Anaconda installed on your system as it is required for setting up the environment.

## Environment setup
To set up your environment for running VG-SSL, follow these steps:

Install Anaconda, if not already installed.

Use the following command to create and activate the VG-SSL environment:
```
conda env create -f env.yml
```

## Datasets
VG-SSL utilizes four public datasets: Pitts30k, MSLS, Tokyo 24/7, and Nordland. Each dataset offers unique challenges and scenarios for visual geo-localization.

**Download and Preparation**: Follow the instructions at VPR datasets downloader to download and prepare these datasets. Note that for MSLS, we use the raw format to leverage the official API for performance evaluation.

**Directory Structure**: Organize the datasets in your local environment as follows:
```
VG_SSL/datasets
├── tokyo247
│   └── images
│       └── test
├── nordland
│   └── images
│       └── test    
├── msls
│   ├── test
│   └── train_val
└── pitts30k
    └── images
        ├── train
        ├── val
        └── test  
```

## Training
Our training scripts are used with singularity and slurm system. If you are not using slurm, do **step 1-2**. If you are not using singularity, do **step 3**. Otherwise, just do **step 4**.

### 1. Export the environment variables

You need to export the environment variables. For example, in the training shell script ```train_msls_byol_final_sbatch.sh```, when you see 

```sbatch --export=ALL,SSL=byol,LR=1e-5,BATCH=64,PROJ=4096,LAY=2,NEG=0,FC=1024,NEGQ=0 ...```,

you need to run

```export SSL=byol LR=1e-5 BATCH=64 PROJ=4096 LAY=2 NEG=0 FC=1024 NEGQ=0```.

### 2. Remove sbatch and run training shell script

Remove the sbatch in training shell script. For example, change

```sbatch --export=ALL,SSL=byol,LR=1e-5,BATCH=64,PROJ=2048,LAY=2,NEG=0,FC=1024,NEGQ=0 ./script/train_msls_ssl_long_byol_neg_random_pair.sbatch```

to

```./script/train_msls_ssl_long_byol_neg_random_pair.sbatch```.

### 3. Remove singularity part

To remove the dependency on Singularity, extract and run the core command from the script. For example, Change

```
singularity exec --nv \
                 --overlay /vast/jx1190/mapillary_sls.sqf:ro \
                 /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
                 /bin/bash -c "source ~/.bashrc; conda activate VG_SSL; python3 -u train_ssl.py --dataset_name msls --backbone resnet50conv5 --aggregation gem --mining partial --ssl_method $SSL --method pair --datasets_folder ./datasets --save_dir global_retrieval --lr $LR --fc_output_dim $FC --train_batch_size $BATCH --infer_batch_size 256 --num_workers 12 --epochs_num 200 --patience 10 --negs_num_per_query $NEGQ --queries_per_epoch 10000 --unfreeze --n_layers $LAY --projection_size $PROJ --neg_samples_num $NEG --pair_negative --random_resized_crop 0.25 --horizontal_flip"
```

to

```
source ~/.bashrc; conda activate VG_SSL; python3 -u train_ssl.py --dataset_name msls --backbone resnet50conv5 --aggregation gem --mining partial --ssl_method $SSL --method pair --datasets_folder ./datasets --save_dir global_retrieval --lr $LR --fc_output_dim $FC --train_batch_size $BATCH --infer_batch_size 256 --num_workers 12 --epochs_num 200 --patience 10 --negs_num_per_query $NEGQ --queries_per_epoch 10000 --unfreeze --n_layers $LAY --projection_size $PROJ --neg_samples_num $NEG --pair_negative --random_resized_crop 0.25 --horizontal_flip
```

### 4. Run the experiments

* First stage: Training global retrieval part:

To run simclr and mocov2 experiments, run ```train_msls_simclr_final_sbatch.sh``` 

To run byol and simsiam experiments, run ```train_msls_byol_final_sbatch.sh``` 

To run barlow twins and vicreg experiments, run ```train_msls_vicreg_final_sbatch.sh``` 
 
After training, you can find the **model_folder_name** in the **logs** folders and the name conversion is as follows:

```
$Training_dataset-$datetime-$uuid
```

* Second stage: Training the reranking part and finetuning:

Use ```train.sh``` to train the reranking part and finetune.

Remember to Change the ```--resume``` argument in the sbatch scripts to load the global retrieval models you trained in the first stage

## Evaluation

For evaluation, use the script located in ```./script/eval_ssl_singularity.sbatch```. This script is designed to evaluate the recall performance of your trained models using standard metrics.

## Acknowledgement

VG-SSL builds upon several existing frameworks and repositories:

https://github.com/gmberton/deep-visual-geo-localization-benchmark  
https://github.com/bytedance/R2Former  
https://github.com/google-research/simclr  
https://github.com/facebookresearch/moco  
https://github.com/lucidrains/byol-pytorch  
https://github.com/facebookresearch/barlowtwins  
https://github.com/facebookresearch/vicreg  
