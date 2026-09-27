# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source-1 (S1) business record, find all Source-2 / Source-3 records that describe the same
business. The metric is **macro F0.5 per S1 entity**. Singletons count: an S1 with no match scores 1
only if nothing is predicted for it. The data covers three countries: India and the US are in train,
and France appears only in test.

Constraints of the challenge: open models (MIT / Apache) with at most 8B parameters, and no external data or lookups.

## Final pipeline

```
 raw TSVs ──► preprocess.py ──► dataset_processed/        (normalisation, abbreviation expansion,
                                                            Indic transliteration, house / legal parsing)
             │
             ▼
 e5-large fine-tuned (contrastive, hard negatives)  ──► FAISS top-20 per S1, within country
             │                                          (pair recall@20 = 99.9 %)
             ▼
 mDeBERTa-v3 cross-encoder reranker  ("name | address" pairs, yes / no)
             │
             ▼
 stage-2 ensemble  LightGBM + XGBoost + CatBoost on 50 features
 (scores, rank / gap in the S1 list, competition between S1s for a record, house / legal / name / address)
             │
             ▼
 decision: threshold + each S2/S3 record to at most one S1  (+ France rules, see below)
             │
             ▼
 matching_results.tsv  +  candidate_pairs.tsv  ──► official validator: PASS
```

## Results

**Validation.** An 80/20 split of the train ground truth by S1. The 20% part (441,200 S1) was never
used to train the retriever or the reranker.

| Stage | Metric | Value |
|---|---|---|
| Retrieval, base e5-large (`embed_text`) | pair recall @10 / @100 | 96.7 / 98.2 |
| Retrieval, **e5-large fine-tuned** | pair recall @10 / @20 / @50 | **99.8 / 99.9 / 100.0** |
| Retrieval, names only (best base model) | pair recall @10 / @100 | 70.3 / 89.7 |
| Cosine threshold + assignment | macro F0.5 | 0.9583 |
| Reranker + assignment | macro F0.5 | 0.9831 |
| Reranker + assignment + abstain rule (**v1**) | macro F0.5 | 0.9914 |
| Stage-2 LightGBM / XGBoost, 59 features (5-fold out-of-fold) | macro F0.5 | 0.9942 |
| Stage-2 ensemble, 50 deployable features (**v2**, hold-out fold) | macro F0.5 | 0.9939 |

**Test submissions.** All five pass the official validator. Leaderboard reference points from other
systems: 0.986149 and 0.987337.

| Submission | Pairs | France matches per S1 | Notes |
|---|---|---|---|
| `submissions/v1` | 5,779,371 | 3.21 | retrieval + reranker + abstain |
| `submissions/v2` | 5,861,979 | 3.35 | + stage-2 ensemble |
| `submissions/v2_fix` | 5,842,356 | 3.27 | + cautious France threshold |
| `submissions/merge` | 5,858,264 | 3.31 | agreement ensemble with another team's submission |
| `submissions/v3_france` | 5,818,106 | 3.18 | **recommended**: v2 + France rules from manual inspection |

Full write-up of every experiment, the error analysis and the France investigation: [reports/RESULTS.md](reports/RESULTS.md).

## Repository layout

```
code/
  eda.py                    exploratory data analysis  -> reports/EDA.pdf
  preprocess.py             normalisation -> student_resource/dataset_processed/
  retrieval_experiments.py  embedding models, FAISS search, recall evaluation, 80/20 split
  finetune_e5.py            hard-negative mining + contrastive fine-tuning of e5-large (DDP)
  train_reranker.py         mDeBERTa cross-encoder: candidates, training, prediction, evaluation
  error_analysis.py         false-positive / false-negative patterns of the reranker
  stack_lgbm.py             stage-2 features + fast macro-F0.5 decision search
  stack_models.py           LightGBM / XGBoost / CatBoost / MLP / logreg + ensembles (5-fold out-of-fold)
  full_pipeline_v1.py       test set: retrieval + reranker -> submission
  full_pipeline_v2.py       test set: + stage-2 ensemble (train on validation split, predict on test)
  postprocess_v2.py         fix / merge / france variants of the v2 submission
  validate_submission.py    official validator (from the challenge kit)
  analysis/                 report builders (Preprocessing.pdf), submission comparison and
                            France inspection scripts (scratch scripts, paths hard-coded)
reports/                    EDA.pdf, Preprocessing.pdf, RESULTS.md, eda/ figures, challenge statement
results/                    metrics (json / md) and training / inference logs of every stage
models/                     stage-2 ensemble weights (in git) + MANIFEST.md for the large models
submissions/                final matching_results.tsv.gz of every variant
```

## Reproducing

```bash
pip install -r requirements.txt           # CUDA 12 build of torch / faiss-gpu; tested on 8x V100 32 GB
# place the challenge kit in the repo root (gitignored): student_resource/dataset/{train,test}/...
# large models: see models/MANIFEST.md (put them in models/)

python code/preprocess.py                                  # ~30 min, 40 CPUs
python code/retrieval_experiments.py                       # base-model retrieval experiments
python code/finetune_e5.py mine && python code/finetune_e5.py train && python code/finetune_e5.py eval
python code/train_reranker.py cands_ft && python code/train_reranker.py train \
    && python code/train_reranker.py predict && python code/train_reranker.py eval
python code/stack_lgbm.py features && python code/stack_models.py
python code/full_pipeline_v1.py --gpus 0,1,2,3,4,5,6,7     # ~60 min -> output/
python code/full_pipeline_v2.py all                        # ~20 min -> output/v2/
python code/postprocess_v2.py france                       # ~5 min  -> output/v3_france/
```

Large intermediate files (embeddings, candidate lists, logits and features: about 80 GB) are
written to `experiments/` and `/tmp/amazon_ml_cache/`. They aren't kept in git, and every stage
regenerates and caches them.
