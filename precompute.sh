 # ResNet50conv5 + GEM
# sbatch script/precompute_msls.sbatch resnet50conv5 logs/global_retrieval/msls-2023-08-07_22-21-01-555d99eb-81e2-4109-a023-135a5e764966
 # ResNet101conv5 + GEM
# sbatch script/precompute_msls.sbatch resnet101conv5 logs/global_retrieval/msls-2023-08-07_22-21-02-0b246b35-24e7-4975-bd17-28c4680e2a47
 # Deitbase + GEM
# sbatch script/train_msls_partial_deit_b.sbatch
 # Deit + GEM
sbatch script/precompute_msls.sbatch deit logs/global_retrieval/msls-2023-08-07_23-03-01-f729b894-bb51-4874-8731-ba2d0af09505
