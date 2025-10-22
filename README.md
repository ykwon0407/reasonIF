# ReasonIF: Large Reasoning Models Fail to Follow Instructions During Reasoning

<p align="center">
  <img src="figures/reasonIF_main.png" width="500">
  <br>
  <em>State-of-the-art large reasoning models demonstrate remarkable problem-solving capabilities, <br>but often fail to follow very simple instructions during reasoning.</em>
</p>

**TL;DR:** It’s critical that LLMs follow user instructions. While prior studies assess instruction adherence in the model’s main responses, we argue that it is also important for large reasoning models (LRMs) to follow user instructions throughout their reasoning process. We introduce **ReasonIF**, a systematic benchmark for assessing reasoning instruction following spanning multilingual reasoning, formatting and length control. **We find frontier LRMs, including GPT-OSS-120B, Qwen3-235B, and DeepSeek-R1, fail to follow reasoning instructions more than 75% of time.** Notably, as task difficulty increases, reasoning instruction following degrades further. We also explore two strategies to enhance reasoning instruction fidelity: (1) multi-turn reasoning and (2) Reasoning Instruction Finetuning (RIF) using synthetic data. RIF improves the IFS of GPT-OSS-20B from 0.11 to 0.27, indicating measurable progress but leaving ample room for improvement.

<p align="center">
  <a href="https://huggingface.co/datasets/ykwon-hf/reasonIF">
    <img alt="Hugging Face" src="https://img.shields.io/badge/Dataset-Hugging%20Face%20🤗-yellow">
  </a>
  <a href="https://arxiv.org/pdf/2510.15211.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arXiv-2510.15211-blue">
  </a>
  <a href="https://www.together.ai/blog/large-reasoning-models-fail-to-follow-instructions-during-reasoning-a-benchmark-study">
    <img alt="Paper URL" src="https://img.shields.io/badge/Blog-Together%20AI-red">
  </a>
</p>


## Key results

<p align="center">
  <img src="figures/overall_comparison.png" width="500">
  <br>
  <em>Figure 1. Instruction-following score of state-of-the-art LRMs when the instruction’s constraint target is the reasoning trace versus the main response. We evaluate six state-of-the-art LRMs with the same set of questions and instructions for all models, differing only in the constraint target. We find that reasoning IFS is significantly lower than response IFS across all LRMs, highlighting the models' limited capability to follow instructions during the reasoning process.</em>
</p>

## Datasets

`data/reasonIF_dataset.json` – The core benchmark dataset. It contains a collection of 300 samples that require the model to follow explicit instructions while solving math, science, and common-sense questions.

`data/number_of_words_reference.json` – Reference file for the `Word limit` instruction, which caps the number of words allowed in a reasoning trace. Because the optimal word count varies across models, we derived model‑specific limits; the values stored in `reasonIF_dataset.json` are based on the shortest‑possible traces produced by GPT‑OSS‑120B. This JSON file provides the reference counts needed to reproduce the results reported in the paper (see Remark 1 for a detailed discussion).

## Setup

```bash
# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment for inference / evaluation
source .venv/bin/activate
```

## Usage

**Step 1: Run model inference**
```bash
python -m src.main --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # model_name should be compatible with vLLM.
```

**Step 2: Evaluate results**
```bash
python -m src.eval_core --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # model_name should be compatible with vLLM.
```

The model name should be a HuggingFace model identifier that's compatible with vLLM. See [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models) for full model support.

## Output

Results are saved to `outputs/[model-name]/`:
- `model_outputs_reasonIF.json`: Raw model outputs with reasoning content
- `eval_results.json`: Evaluation metrics including instruction following accuracy

We also include a simple result in the `outputs` folder. They are generatd from `DeepSeek-R1-Distill-Qwen-14B`.

## Citation

```
@article{kwon2025reasonif,
  title        = {ReasonIF: Large Reasoning Models Fail to Follow Instructions During Reasoning},
  author       = {Yongchan Kwon and Shang Zhu and Federico Bianchi and Kaitlyn Zhou and James Zou},
  year         = {2025},
  journal      = {arXiv preprint arXiv:2510.15211},
  archivePrefix= {arXiv},
  eprint       = {2510.15211},
  primaryClass = {cs.LG},
  note         = {Preprint — submitted October 17, 2025}
}
```


