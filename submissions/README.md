# Submissions

Each folder has the `matching_results.tsv` in gzip form: all 1,732,544 test S1 rows, with columns
`source1_entity_id`, `matched_entity_ids`. Unzip with `gunzip -k`.

`candidate_pairs.tsv` is about 450–540 MB per variant, so it isn't kept here. It's identical for
v1, v2, v2_fix and v3_france (the fine-tuned retriever's top-20), and `full_pipeline_v1.py`
regenerates it. All variants passed `code/validate_submission.py` together with their candidate file.

| Variant | Pairs | Empty S1 | India / US / France matches per S1 | How it's made |
|---|---|---|---|---|
| v1 | 5,779,371 | 5.9% | 3.35 / 3.37 / 3.21 | `full_pipeline_v1.py`: reranker p > 0.7, one S1 per record, abstain |
| v2 | 5,861,979 | 5.7% | 3.39 / 3.39 / 3.35 | `full_pipeline_v2.py`: mean of LightGBM + XGBoost + CatBoost, p > 0.75 |
| v2_fix | 5,842,356 | 5.7% | 3.39 / 3.39 / 3.27 | `postprocess_v2.py fix`: France thr 0.9, risky pairs need p > 0.98 |
| merge | 5,858,264 | 5.7% | 3.40 / 3.39 / 3.31 | `postprocess_v2.py merge`: agreement ensemble with another team's submission |
| **v3_france** | 5,818,106 | 5.8% | 3.39 / 3.39 / 3.18 | `postprocess_v2.py france`: France descriptor-swap and house-number rules |

Estimated macro F0.5 (validation): v1 0.9914, v2 0.9939. France can't be validated because it has no
training data. See `reports/RESULTS.md` §8–9.
