#!/bin/bash

sbatch script/train_msls_partial_dbaug.sbatch
sbatch script/train_msls_partial_highlr.sbatch
sbatch script/train_msls_partial.sbatch
sbatch script/train_msls_res18conv5_partial_highlr.sbatch
sbatch script/train_msls_res18conv5_partial.sbatch
sbatch script/train_msls_res50_partial_dbaug.sbatch
sbatch script/train_msls_res50_partial_highlr.sbatch
sbatch script/train_msls_res50_partial.sbatch
sbatch script/train_msls_res50conv5_partial_highlr.sbatch
sbatch script/train_msls_res50conv5_partial.sbatch
sbatch script/train_pitts30k_full_dbaug.sbatch
sbatch script/train_pitts30k_full_highlr.sbatch
sbatch script/train_pitts30k_full.sbatch
sbatch script/train_pitts30k_partial.sbatch
sbatch script/train_pitts30k_res18conv5_full_highlr.sbatch
sbatch script/train_pitts30k_res18conv5_full.sbatch
sbatch script/train_pitts30k_res50_full_dbaug.sbatch
sbatch script/train_pitts30k_res50_full_highlr.sbatch
sbatch script/train_pitts30k_res50_full.sbatch
sbatch script/train_pitts30k_res50_partial.sbatch
sbatch script/train_pitts30k_res50conv5_full_highlr.sbatch
sbatch script/train_pitts30k_res50conv5_full.sbatch