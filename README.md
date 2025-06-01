# Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO

<p align="center">
⬇️ <a href="https://huggingface.co/WaltonFuture/Qwen2.5-VL-7B-MM-UPT-MMR1" target="_blank">Model</a> | 📃 <a href="https://arxiv.org/pdf/2505.22453" target="_blank">Paper</a> <br>
</p>


This project is built based on [EasyR1](https://github.com/hiyouga/EasyR1) project to support unsupervised GRPO for multi-modal LLMs. We thank the authors of EasyR1 for providing such a high-performance RL training framework.


## Introduction

In this work, we are the first to investigate the use of GRPO, a stable and scalable online RL algorithm, for enabling MLLM's continual self-improvement without any external supervision. We propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs. MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that MM-UPT significantly improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3\%→72.9\% on MathVista, 62.9\%→68.7\% on We-Math), using standard dataset without ground truth labels. MM-UPT also outperforms prior unsupervised baselines and even approaches the results of supervised GRPO. Furthermore, we show that incorporating synthetic questions, generated solely by MLLM itself, can boost performance as well, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for continual, autonomous enhancement of MLLMs in the absence of external supervision.

<div align=center>
<img src="assets/mm-upt.png"  width = "80%" alt="mm-upt" align=center/>
</div>



## Example: Train Qwen2.5-VL-7B using MM-UPT on [MMR1](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0) Dataset without Labels


```bash
git clone https://github.com/waltonfuture/MM-UPT.git
cd MM-UPT
pip install -e .
```

### Unsupervised GRPO Training

```bash
bash examples/qwen2_5_vl_7b_mmr1.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/mm-upt/qwen2_5_vl_7b_mmr1/global_step_80/actor
```

Our model trained based on this script is available [here](https://huggingface.co/WaltonFuture/Qwen2.5-VL-7B-MM-UPT-MMR1).

For other standard datasets used in our paper, please refer to:

- [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k)

- [GeoQA](https://huggingface.co/datasets/WaltonFuture/GEOQA_R1V_Train_8K)


## Using Synthetic Datasets

Please refer to these synthetic datasets built on different seed datasets using two synthetic methods.

| Methods | Geometry3K | GeoQA | MMR1 |
|------------------|------------|-------|------|
| In-Context Synthesizing         |    [Geometry3K-1](https://huggingface.co/datasets/WaltonFuture/geometry3k-in-context-synthesizing)        |   [GeoQA-1](https://huggingface.co/datasets/WaltonFuture/GeoQA-8K-in-context-synthesizing)    |  [MMR1-1](https://huggingface.co/datasets/WaltonFuture/MMR1-in-context-synthesizing)    |
| Direct Synthesizing         |    [Geometry3K-2](https://huggingface.co/datasets/WaltonFuture/geometry3k-direct-synthesizing)         | [GeoQA-2](https://huggingface.co/datasets/WaltonFuture/GeoQA-8K-direct-synthesizing)      | [MMR1-2](https://huggingface.co/datasets/WaltonFuture/MMR1-direct-synthesizing)     |



## Acknowledgment

Our models are built upon the amazing [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) family.
We thank [EasyR1](https://github.com/hiyouga/EasyR1)  for the training codes.
We also appreciate that several concurrent works have explored similar ideas. [TTRL](https://github.com/PRIME-RL/TTRL) demonstrates strong performance in test-time training with this approach.
[SRT](https://github.com/tajwarfahim/srt) adopts this paradigm for the unsupervised self-training of LLMs.

## Contact

Please contact Lai Wei (waltonfuture@sjtu.edu.cn) if needed.

## Citation
```
@article{wei2025unsupervised,
  title={Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO},
  author={Wei, Lai and Li, Yuting and Wang, Chen and Wang, Yue and Kong, Linghe and Huang, Weiran and Sun, Lichao},
  journal={arXiv preprint arXiv:2505.22453},
  year={2025}
}
```
