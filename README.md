# VG_SSL

This repo is to investigate Visual Geo-localization (VG) with Self-Supervised representation Learning (SSL) methods.

## Environment setup
The repo requires Anaconda to install the running environment. After installing Anaconda, run the following command to install environment:
```
conda env create -f env.yml
```

## Datasets
We are using five public datasets: Pitts30k, MSLS, tokyo 24/7, Eynsham, St. Lucia. Please refer to https://github.com/gmberton/VPR-datasets-downloader to download and prepare the dataset.
Additionaly, we use h5df files to merge the dataset images into one h5 files per dataset. Run the following command to prepare h5 files:
```
transform_dataset.sh
```
The datasets should be organized like this:
```
VPR_SSL/datasets
├── eynsham
│   ├── test_database.h5
│   └── test_queries.h5
├── msls
│   ├── test_database.h5 -> val_database.h5
│   ├── test_queries.h5 -> val_queries.h5
│   ├── train_database.h5
│   ├── train_queries.h5
│   ├── val_database.h5
│   └── val_queries.h5
├── pitts30k
│   ├── test_database.h5
│   ├── test_queries.h5
│   ├── train_database.h5
│   ├── train_queries.h5
│   ├── val_database.h5
│   └── val_queries.h5
├── st_lucia
│   ├── test_database.h5
│   └── test_queries.h5
└── tokyo247
    ├── test_database.h5
    └── test_queries.h5
```

## Training
Our training scripts are used to submit to slurm system. If you are using slurm, skip to Step 4. To run the scripts locally, you need to: 
### 1. Remove srun
Remove the ```srun``` from the training script. For example, if you are using ```train_msls_ssl_long_byol_neg1.sbatch```, then change 

``` srun python3 train_ssl.py --dataset_name=msls --mining random ...``` 

to 

``` python3 train_ssl.py --dataset_name=msls --mining random ...``` 

### 2. Export the environment variables
You need to export the environment variables. For example, in the training shell script ```train_msls_byol_final_sbatch.sh```, when you see 

```sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=1.0 ...```,

you need to run

```export SSL=byol SEED=0 LR=1e-4 BATCH=64 PROJ=2048 LAY=2 NEG=1.0```.

### 3. Remove sbatch and run training shell script

Remove the sbatch in training shell script. For example, in the training shell script ```train_msls_byol_final_sbatch.sh```, when you see 

```sbatch --export=ALL,SSL=byol,SEED=0,LR=1e-4,BATCH=64,PROJ=2048,LAY=2,NEG=1.0 ./script/train_msls_ssl_long_byol_neg.sbatch```,

you need to change it to:

```./script/train_msls_ssl_long_byol_neg.sbatch```.

### 4. Run the experiments

To run baseline experiment, run ```train_msls_partial_sbatch.sh``` 

To run simclr and mocov2 experiments, run ```train_msls_simclr_final_sbatch.sh``` 

To run byol and simsiam experiments, run ```train_msls_byol_final_sbatch.sh``` 

To run barlow twins and vicreg experiments, run ```train_msls_vicreg_final_sbatch.sh``` 
 
After training, you can find the **model_folder_name** in the **logs** folders and the name conversion is as follows:

```
$Training_dataset-$datetime-$uuid
```

## Evaluation
Similar to training, if you want to evaluate locally, you need to change the sbatch scripts to bash scripts following Step 1, 2, and 3 in **Training** for eval scripts. Additionally, you need to run the following command to indicate the datasets you want to evaluate:

``` export SLURM_ARRAY_TASK_ID=X ```

The available datasets are

```
SLURM_ARRAY_TASK_ID=1 # st_lucia
SLURM_ARRAY_TASK_ID=2 # pitts30k
SLURM_ARRAY_TASK_ID=3 # msls
SLURM_ARRAY_TASK_ID=4 # tokyo24/7 
SLURM_ARRAY_TASK_ID=7 # Eynsham
```

Copy your **model_folder_name** to ```eval_sbatch_all.sh, eval_sbatch_pca_all.sh, eval_ssl_sbatch_proj_all.sh```, for example:

```
./eval_sbatch.sh msls-2023-04-23_22-41-52-97371a43-c35e-4ccd-ab42-3483b184fc56 resnet50conv4
```

To evaluate baseline experiment, run ```eval_sbatch_all.sh``` 

To evaluate baseline experiments with PCA, run ```eval_sbatch_pca_all.sh``` (Remember to change pca dim as the last argument)

To evaluate SSL experiments, run ```eval_ssl_sbatch_proj_all.sh```  (Remember to change the 3rd argument as the projection dim, and the 4th argument as the number of projection layers)

After evaluation, you can find the **model_folder_name** in the **test/default** folders. All results of one **model_folder_name** are grouped in that folder.

## Acknowledgement
The VG framework is inspired by https://github.com/gmberton/deep-visual-geo-localization-benchmark.
The implementation of SSL methods refers to the following repos:
https://github.com/google-research/simclr
https://github.com/facebookresearch/moco
https://github.com/lucidrains/byol-pytorch
https://github.com/facebookresearch/barlowtwins
https://github.com/facebookresearch/vicreg

