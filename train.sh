# Global retrieval
 # ResNet50conv5 + GEM
# sbatch script/train_msls_partial_50.sbatch
 # Deit + GEM
# sbatch script/train_msls_partial_deit.sbatch


# Reranking
 # ResNet50conv5 + GEM + unfreeze
# sbatch script/train_msls_rerank_50_unfreeze.sbatch
 # Deit + GEM
# sbatch script/train_msls_rerank_deit.sbatch

# Finetuning
 # ResNet50conv5 + GEM + unfreeze
# sbatch script/train_msls_finetune_50_unfreeze.sbatch
 # Deit + GEM
# sbatch script/train_msls_finetune_deit.sbatch
