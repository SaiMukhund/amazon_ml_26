| Scorer | Macro F0.5 | Precision | Recall | Thr | Abstain | F0.5 without abstain |
|---|---|---|---|---|---|---|
| reranker v1 only | 0.9914 | 0.9992 | 0.9741 | 0.7 | True | 0.9831 |
| retriever cosine only | 0.9584 | 0.9871 | 0.9106 | 0.8 | False | 0.9584 |
| lgbm | 0.9942 | 0.9988 | 0.9843 | 0.7 | False | 0.9942 |
| xgb | 0.9942 | 0.9992 | 0.9833 | 0.8 | False | 0.9942 |
| cat | 0.9941 | 0.9992 | 0.9831 | 0.8 | True | 0.9941 |
| mlp | 0.9940 | 0.9988 | 0.9837 | 0.7 | True | 0.9940 |
| logreg | 0.9866 | 0.9946 | 0.9706 | 0.7 | True | 0.9862 |
| mean(lgbm+xgb+cat+mlp+logreg) | 0.9942 | 0.9989 | 0.9840 | 0.7 | True | 0.9942 |
| rank-mean(all) | 0.9874 | 0.9919 | 0.9808 | 0.8 | True | 0.9821 |
| mean(lgbm+xgb+cat) | 0.9942 | 0.9989 | 0.9842 | 0.7 | True | 0.9942 |
| weighted(all) | 0.9942 | 0.9989 | 0.9842 | 0.7 | False | 0.9942 |
