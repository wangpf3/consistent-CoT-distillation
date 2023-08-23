# SCOTT

This is a Pytorch implementation for our ACL 2023 outstanding paper "SCOTT: Self-Consistent Chain-of-Thought Distillation" [[arxiv](https://arxiv.org/abs/2305.01879)].

## Contrastive Decoding

### 1. Prepare the data

Download the dataset: [StrategyQA](https://allenai.org/data/strategyqa)/[CommonsenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa)/[CREAK](https://github.com/yasumasaonoe/creak)/[QASC](https://allenai.org/data/qasc).

Split the dataset into `train/dev/test.jsonl` subsets. Also build a `train.counterfactual.jsonl` subset by perturbing the answers in the `train.jsonl` subset. Organize all the subsets in the folder `data/DATASET`.

### 2. Obtain the rationales

```bash
./scripts/run_contrastive_decoding.sh
```
The generated rationales would be stored at `outputs/DATASET`.

## Counterfactual Training

```bash
./scripts/run_counterfactual_training.sh
```
After training, the checkpoints and the evaluation result will be stored at `checkpoints/DATASET`.

## Citation

```
@inproceedings{wang-etal-2023-scott,
    title = "{SCOTT}: Self-Consistent Chain-of-Thought Distillation",
    author = "Wang, Peifeng  and
      Wang, Zhengyang  and
      Li, Zheng  and
      Gao, Yifan  and
      Yin, Bing  and
      Ren, Xiang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.304",
    doi = "10.18653/v1/2023.acl-long.304",
    pages = "5546--5558",
}
```
