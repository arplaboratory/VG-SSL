# Global retrieval
 # ResNet50conv5 + GEM
# sbatch script/train_msls_partial_50.sbatch
 # ResNet101conv5 + GEM
# sbatch script/train_msls_partial_101.sbatch
 # ResNet50conv5 + GEM + unfreeze
# sbatch script/train_msls_partial_50_unfreeze.sbatch
 # ResNet101conv5 + GEM + unfreeze
# sbatch script/train_msls_partial_101_unfreeze.sbatch
 # Deitbase + GEM
# sbatch script/train_msls_partial_deit_b.sbatch
 # Deit + GEM
# sbatch script/train_msls_partial_deit.sbatch


# Reranking
 # ResNet50conv5 + GEM + unfreeze
sbatch script/train_msls_rerank_50_unfreeze.sbatch
 # ResNet101conv5 + GEM + unfreeze
sbatch script/train_msls_rerank_101_unfreeze.sbatch
 # Deitbase + GEM
sbatch script/train_msls_rerank_deit_b.sbatch
 # Deit + GEM
sbatch script/train_msls_rerank_deit.sbatch