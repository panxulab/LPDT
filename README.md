# Pre-trained Language Models Improve the Few-shot Prompt Ability of Decision Transformer
### <p align="center">[TMLR 2025]</p>
<p align="center">
  <a href="https://yangyu0879.github.io/">Yu Yang</a> ·
  <a href="https://panxulab.github.io/">Pan Xu</a>
</p>
<p align="center">
  Duke University
</p>

---

## Overview
Official implementation of the paper "Pre-trained Language Models Improve the Few-shot Prompt Ability of Decision Transformer".

## Installation
Creating the environment through the following steps:
```bash
conda create --name lpdt python=3.8.5
conda activate lpdt
pip install -r requirements.txt
./install_envs.sh
```


## Experiments
First, download the [Dataset](https://drive.google.com/drive/folders/1bTDFZ_NuJvI_9XhPeXqfuwCT-tL86pa3?usp=sharing) and place them in `./dataset`.


Fine-tune the pretrained language model with classifier regularization using the Decision Transformer.
```bash
python experiment.py \
    --env ant_dir \
    --model_type dt \
    --dataset_mode expert \
    --test_dataset_mode expert \
    --seed 0 \
    --K 20 \
    -lr 1e-4 \
    -lmlr 1e-5 \
    --warmup_steps 10000 \
    --pretrained_lm gpt2 \
    --model_type dt \
    --adapt_mode \
    --adapt_embed \
    --lora \
    --mlp_embedding \
    --outdir test/ \
    --dropout 0.1 \
    --description "test_ratio_1.0" \
    --batch_size 6 \
    -w \
    --load_path "" \
    --ratio 1.0 \
    --classifier \
    --classifier_lambda 0.1 \
    --num_class 50
```



Fine-tune the pretrained language model with classifier regularization using the Reinformer.
```bash
python experiment.py \
    --env ant_dir \
    --model_type dt \
    --dataset_mode expert \
    --test_dataset_mode expert \
    --seed 0 \
    --K 20 \
    -lr 1e-4 \
    -lmlr 1e-5 \
    --warmup_steps 10000 \
    --pretrained_lm gpt2 \
    --model_type reinformer \
    --adapt_mode \
    --adapt_embed \
    --lora \
    --mlp_embedding \
    --outdir test/ \
    --dropout 0.1 \
    --description "test_ratio_1.0" \
    --batch_size 6 \
    -w \
    --load_path "" \
    --ratio 1.0 \
    --classifier \
    --classifier_lambda 0.1 \
    --num_class 50
```

## Citation
```
@article{
yang2025pretrained,
title={Pre-trained Language Models Improve the Few-shot  Prompt Ability of Decision Transformer},
author={Yu Yang and Pan Xu},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=k520i3XEMK},
note={}
}
```



