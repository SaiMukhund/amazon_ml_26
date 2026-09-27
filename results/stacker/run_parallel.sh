#!/bin/bash
# Train the five stage-2 models in parallel (CPU + 4 GPUs), then build the ensembles.
cd /home/saimukhundm/amazon_ml
P=/home/saimukhundm/miniconda3/envs/general/bin/python
L=experiments/stacker
# (features already built)
[ -f $L/features.parquet ] || { echo "features.parquet missing" > $L/parallel.log; exit 1; }
LGBM_THREADS=30 CUDA_VISIBLE_DEVICES=""  $P -u code/stack_models.py --models lgbm   --no-ensemble > $L/m_lgbm.log 2>&1 &
CUDA_VISIBLE_DEVICES=0                   $P -u code/stack_models.py --models xgb    --no-ensemble > $L/m_xgb.log 2>&1 &
CUDA_VISIBLE_DEVICES=1                   $P -u code/stack_models.py --models cat    --no-ensemble > $L/m_cat.log 2>&1 &
CUDA_VISIBLE_DEVICES=2                   $P -u code/stack_models.py --models mlp    --no-ensemble > $L/m_mlp.log 2>&1 &
CUDA_VISIBLE_DEVICES=3                   $P -u code/stack_models.py --models logreg --no-ensemble > $L/m_logreg.log 2>&1 &
wait
$P -u code/stack_models.py > $L/ensemble.log 2>&1              # reuses the cached OOF predictions
echo done > $L/parallel.done
