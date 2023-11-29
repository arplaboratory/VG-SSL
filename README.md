# VG-SSL

This is the official repository for VG-SSL: Benchmarking Self-supervised Representation Learning Approaches for Visual Geo-localization.

## Environment setup
The repo requires Anaconda to install the running environment. After installing Anaconda, run the following command to install environment:
```
conda env create -f env.yml
```

## Datasets
We are using four public datasets: Pitts30k, MSLS, tokyo 24/7, and Nordland. Please refer to https://github.com/gmberton/VPR-datasets-downloader to download and prepare the dataset. For MSLS, we use the raw format of the dataset so that we can use the official API to evaluate the performance.
The datasets should be organized like this:
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
├── pitts30k
    └── images
        ├── train
        ├── val
        └── test  
```

## Training
Our training scripts are used with singularity and slurm system. If you are using slurm, skip to Step 3. To run the scripts locally, you need to: 
### 1. Export the environment variables
You need to export the environment variables. For example, in the training shell script ```train_msls_byol_final_sbatch.sh```, when you see 

```sbatch --export=ALL,SSL=byol,LR=1e-5,BATCH=64,PROJ=4096,LAY=2,NEG=0,FC=1024,NEGQ=0 ...```,

you need to run

```export SSL=byol LR=1e-5 BATCH=64 PROJ=4096 LAY=2 NEG=0 FC=1024 NEGQ=0```.

### 2. Remove sbatch and run training shell script

Remove the sbatch in training shell script. For example, in the training shell script ```train_msls_byol_final_sbatch.sh```, when you see 

```sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=1.0 ./script/train_msls_ssl_long_byol_neg.sbatch```,

you need to change it to:

```./script/train_msls_ssl_long_byol_neg.sbatch```.

### 3. Run the experiments

To run simclr and mocov2 experiments, run ```train_msls_simclr_final_sbatch.sh``` 

To run byol and simsiam experiments, run ```train_msls_byol_final_sbatch.sh``` 

To run barlow twins and vicreg experiments, run ```train_msls_vicreg_final_sbatch.sh``` 
 
After training, you can find the **model_folder_name** in the **logs** folders and the name conversion is as follows:

```
$Training_dataset-$datetime-$uuid
```

## Evaluation
Evaluation script is in ```./script/eval_ssl_singularity.sbatch```.

## Acknowledgement
The VG global retrieval framework is implemented based on https://github.com/gmberton/deep-visual-geo-localization-benchmark.

The R2Former reranking part refers to https://github.com/bytedance/R2Former.

The implementation of SSL methods refers to the following repos:

https://github.com/google-research/simclr

https://github.com/facebookresearch/moco

https://github.com/lucidrains/byol-pytorch

https://github.com/facebookresearch/barlowtwins

https://github.com/facebookresearch/vicreg

