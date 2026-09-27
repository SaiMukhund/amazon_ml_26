# Results and findings

All validation numbers are on the 20% hold-out of the train ground truth (seed 42): 441,200 S1
queries, against a corpus of 2,061,938 S2/S3 records (every true match plus 20% of the distractors).
The retriever and the reranker never saw these S1s.

## 1. Data (see EDA.pdf)
- **Train:** 2.21M S1, 5.03M S2, 5.29M S3.
- **Test:** 1.73M S1 (India 810k, US 663k, **France 259k, a country absent from train**), plus 9.97M S2+S3.
- **Record ownership:** every S2/S3 record belongs to at most one S1. 5.6% of S1 are singletons, and country agrees in 100% of true pairs, so blocking by country is exact.
- **Hard negatives** are generated: the house number is off by 1–5, 7, 9 or 11, and/or the legal suffix is changed.
- **Indic script:** 24% (S2) and 13% (S3) of India names are written in Indic script. The Indic vocabulary is closed (about 170 words per script), and all of it is seen in train.

## 2. Preprocessing (see Preprocessing.pdf)
- Unicode and case normalisation.
- Abbreviation expansion: rd→road, st→street/saint by position, and so on.
- Indic→Latin word map learned from train pairs (1,347 words).
- State and region codes (US, India; French départements→régions).
- Legal-form and honorific stripping.
- House, unit and number parsing.
- `embed_text = name_core | addr_clean` for the embedders.

## 3. Retrieval (pair recall %, within country)
| Model | Input | @10 | @20 | @50 | @100 |
|---|---|---|---|---|---|
| e5-base | name | 70.0 | 76.7 | 86.2 | 89.4 |
| e5-large | name | 70.3 | 77.1 | 86.5 | 89.7 |
| bge-m3 | name | 69.4 | 76.1 | 85.6 | 89.0 |
| e5-base | name + address | 96.2 | 96.9 | 97.5 | 97.9 |
| e5-large | name + address | 96.7 | 97.3 | 97.9 | 98.2 |
| bge-m3 | name + address | 96.4 | 97.0 | 97.5 | 97.8 |
| **e5-large fine-tuned** | name + address | **99.8** | **99.9** | **100.0** | **100.0** |

The address matters much more than the name. Contrastive fine-tuning with mined hard negatives
closes almost all of the remaining gap, so top-20 is enough (candidate recall 0.9992).

## 4. Reranker (mDeBERTa-v3-base cross-encoder)
| Scorer | Decision | Macro F0.5 |
|---|---|---|
| cosine (fine-tuned e5) | threshold | 0.9564 |
| cosine | threshold + one S1 per record | 0.9583 |
| reranker | threshold | 0.8468 |
| reranker | threshold + one S1 per record | 0.9831 |
| reranker | threshold 0.7 + one S1 per record + **abstain** | **0.9914** |

The pair AUC of the reranker (0.9949) is below that of the cosine (0.9971), yet it wins by a wide
margin after assignment. It is very good at ranking the right S1 first, but it gives high scores to
namesakes, so the "one S1 per record" rule is essential.

**Abstain rule:** if two or more S1s pass the threshold for the same record, give it to none of them.

## 5. Error analysis (reranker + assignment, thr 0.5)
- **Where F0.5 is lost:**
  - 52.7%: non-singletons with false merges only.
  - 26.8%: misses only.
  - 15.7%: singletons with a false merge.
  - 4.8%: both.
- **Totals:** 22,977 false positives and 24,038 missed pairs. Only 1,146 missed pairs were not retrieved at all.
- **Candidates with an empty address** cause **91% of false positives and 82% of misses**. A name-only record can't be told apart from namesakes. If these were handled perfectly, the ceiling would be 0.9979.

## 6. Stage-2 decision models (59 features, 5-fold grouped by S1, out-of-fold)
| Model | Macro F0.5 | Precision | Recall | Threshold |
|---|---|---|---|---|
| reranker + abstain (reference) | 0.9914 | 0.9992 | 0.9741 | 0.7 |
| logistic regression | 0.9866 | 0.9946 | 0.9706 | 0.7 |
| MLP | 0.9940 | 0.9988 | 0.9837 | 0.7 |
| CatBoost | 0.9941 | 0.9992 | 0.9831 | 0.8 |
| **LightGBM** | **0.9942** | 0.9988 | 0.9843 | 0.7 |
| **XGBoost** | **0.9942** | 0.9992 | 0.9833 | 0.8 |
| mean / weighted / tree ensembles | 0.9942 | 0.9989 | 0.9842 | 0.7 |

- **Top LightGBM features by gain:**
  - the reranker's margin over the best *other* S1 claiming the same record (63%)
  - the fine-tuned cosine (22%)
  - the raw reranker logit (11%)
- **The gain over the abstain rule is all recall.** The model learns when a contested record is still safe to assign.
- **Ensembling adds nothing** beyond the best single tree model.

**Deployed feature set (v2):** 50 features. The cosines of the 6 base embedders were dropped (they
would need 6 more passes over 11.7M test records), as were `cos_mean` and the two name-frequency
features, which scale with corpus size. This costs about 0.0003: on the same hold-out fold, the
59-feature models score 0.9942 and the 50-feature models 0.9939.

## 7. Test-set pipelines
- **v1:** embedding 1.73M S1 and 9.97M S2/S3, retrieving 34.65M candidate pairs, and reranking them on 8×V100 took 61 min.
- **v2:** features, ensemble and decision took 16 min on top of v1's caches.
- **Empty S1 rows:** 5.7–5.9%, in line with the 5.6% singleton rate in train.

## 8. Comparison with two external submissions (leaderboard 0.986149 and 0.987337)
- **Overall agreement:** pair-level Jaccard of about 0.98 between all systems. India and US are nearly identical across systems (about 3.39 matches per S1 each).
- **France is where the systems diverge:**

  | Matches per S1 in France | |
  |---|---|
  | 0.987337 submission | 3.16 |
  | 0.986149 submission | 3.24 |
  | our v1 | 3.21 |
  | our v2 | 3.35 |

  The better-scoring external file matches *less* in France.
- **v2's extra pairs** (compared with the 0.986 submission) have a house-number fingerprint 10× as often as agreed pairs (15.1% vs 1.5%).
- **On validation (India/US),** stricter house rules only hurt, because the model already separates those pairs there. So the problem is specific to France.

## 9. France: manual inspection of disagreements
The France pairs were split into groups by which systems accept them, and samples of each group were
read by hand. The table shows each pattern's rate per group:

| France pair group | Descriptor swap | House number differs | Names share no word | Candidate address empty |
|---|---|---|---|---|
| F. all three agree (782k) | 1.2% | 0.9% | 4.1% | 1.7% |
| A. only v2 says yes (31.8k) | 25.8% | 29.3% | 17.8% | 13.6% |
| B. v2 and 0.9873 file yes (19.6k) | 4.1% | 7.6% | 59.8% | 15.4% |
| C. only the 0.9873 file yes (12.0k) | 3.4% | 16.5% | 40.2% | 34.5% |
| D. v2 and 0.9861 file yes, 0.9873 file no (35.1k) | 49.5% | 1.2% | 6.2% | 1.1% |
| E. only the 0.9861 file yes (17.2k) | 88.3% | 2.3% | 2.2% | 1.4% |

**Findings:**
1. **Descriptor swap: the French hard negative.**
   - French names follow a template: core name, then a generic word (club, comite, lycee, amis, union, federation…), then the legal form.
   - The false candidates keep the core name and the address and swap that one word. For example, "House Lycée SAS" vs "House Section SAS", both at 41 Chaussée Denis Papin.
   - Swaps are 1.2% of the pairs everyone agrees on, but 50–88% of the pairs the better-scoring submission rejects.
2. **v2 accepts differing house numbers in France** (18 vs 20, 33 vs 36, 50 vs 54). Parsing isn't the cause: `N°36`, `Nº 20`, `#3`, `NO 47` and `0131` all parse correctly.
3. **Completely different names at the same address are usually real in France:** brand names (WEXORBI), initials ("Raid Compagnie SAS" ↔ "RC"), and domains (`developpementclubeurl.com`). These pairs should be kept.
4. **The remaining recall gap is empty-address namesakes and initials:** for example "Collège de la Daction" ↔ the same name with no address, or "LC" for "Lille Compagnie", where the stacker overrules a confident reranker.

**The v3_france rule** (`postprocess_v2.py france`, France only):
- Reject descriptor swaps. The word list is the 114 most frequent France S1 name words. Synonyms such as cie/compagnie, ets/etablissements and centre/center don't count as swaps.
- When both house numbers are present and differ, require probability above 0.98.
- **Effect:** it removes 37.1k swap pairs and 19.0k differing-house pairs, bringing France to 3.18 matches per S1.
- **Agreement:** its France Jaccard with the 0.987337 submission is 0.929, the highest of all files (v2 0.906, the 0.986149 submission 0.904).

## 10. Possible next steps
- For empty-address candidates, use name-only retrieval and a namesake-aware decision; this is the largest remaining error source.
- Build France-specific hard negatives (descriptor swaps, house-number shifts) and fine-tune the reranker on them.
- Take a majority vote across three independent systems.
