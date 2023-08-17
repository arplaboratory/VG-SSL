# Precompute
 # ResNet50conv5 + GEM
sbatch script/precompute_msls.sbatch resnet50conv5 logs/global_retrieval/msls-2023-08-10_11-47-49-6b0fdeef-1ebc-4af6-8332-d8145cc9acc5
 # ResNet101conv5 + GEM
sbatch script/precompute_msls.sbatch resnet101conv5 logs/global_retrieval/msls-2023-08-09_20-35-49-592ba884-45dd-4c06-8ed7-a6d6c20fd822
 # Deitbase + GEM
sbatch script/precompute_msls.sbatch deitBase logs/global_retrieval/msls-2023-08-09_19-25-11-5382da1d-d113-4d00-bde6-d2238d05a339
 # Deit + GEM
sbatch script/precompute_msls.sbatch deit logs/global_retrieval/msls-2023-08-10_15-17-59-d389d9e3-71aa-461d-bc47-59ea7bf3c259
