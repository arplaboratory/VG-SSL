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

### Training
Our training scripts are used to submit to slurm system. To run the scripts locally, you need to change the suffix from "sbatch" to 
To run baseline experiment, run the following command:


## Acknowledgement
The VG framework is inspired by https://github.com/gmberton/deep-visual-geo-localization-benchmark.
The implementation of SSL methods refers to the following repos:
https://github.com/google-research/simclr
https://github.com/facebookresearch/moco
https://github.com/lucidrains/byol-pytorch
https://github.com/facebookresearch/barlowtwins
https://github.com/facebookresearch/vicreg

