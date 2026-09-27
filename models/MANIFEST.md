# Models

## In this repository
| Path | What | Size |
|---|---|---|
| `stacker_v2/lgbm.txt` | LightGBM, 274 trees | |
| `stacker_v2/xgb.json` | XGBoost, 405 trees | |
| `stacker_v2/cat.cbm` | CatBoost, 791 trees | |
| `stacker_v2/mlp.pt` | MLP 512-256-128, with its normalisation statistics (trained, not in the final ensemble) | |
| `stacker_v2/config.json` | feature list, ensemble weights, threshold (0.75), abstain flag and hold-out scores | |
| | total | 12 MB |

## Not in git (too large): weights on the training machine
The configs, tokenizer configs and training logs of both models are in their folders here. Copy the
weights to the same paths to run the pipelines.

| Model | Base (licence) | Params | File | Bytes | sha256 |
|---|---|---|---|---|---|
| `e5-large-ft` retriever | `intfloat/multilingual-e5-large` (MIT) | 560M | `model.safetensors` | 2,235,408,584 | `a4585f192a8e113d438ff341ccc8fd5b56fb78edee0fbb3fa9206036abb2cde7` |
| `mdeberta-reranker` cross-encoder | `microsoft/mdeberta-v3-base` (MIT) | 279M | `model.safetensors` | 1,115,265,124 | `fbbee26bf8d41031dbb6c608ca405824c1b0f1883d61a6b59797e8e14140939c` |

These live on the training machine at `/home/saimukhundm/amazon_ml/models/{e5-large-ft,mdeberta-reranker}/`.
`tokenizer.json` comes from each base model on the Hugging Face Hub.

Base models used only in experiments, downloaded from the Hub: `intfloat/multilingual-e5-base` (MIT),
`intfloat/multilingual-e5-large` (MIT) and `BAAI/bge-m3` (MIT; converted from `.bin` to safetensors
because transformers refuses `.bin` on torch < 2.6).

## Training summary
- **e5-large-ft:** symmetric InfoNCE loss, gathering in-batch negatives across all 8 GPUs, plus mined hard negatives. Mined 1.67M anchors and 6.1M positives, then trained 1 epoch: 1,627 steps, batch 128 per GPU, lr 2e-5, temperature 0.02, max length 128, with records of the validation split excluded.
- **mdeberta-reranker:** BCE loss on 8.5M pairs (37.9% positive) from the fine-tuned retriever's top-20. Trained 8,000 steps, batch 64 per GPU, lr 3e-5, max length 160. Input is `name_clean | addr_clean` for each side. Must be loaded with `dtype=torch.float32` to train.
- **stacker_v2:** fitted on folds 1–4 of the validation split's top-20 pairs (7.1M pairs); threshold and ensemble chosen on fold 0.
