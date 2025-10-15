# ReasonIF

ReasonIF – a systematic benchmark for assessing large reasoning models' reasoning instruction following capability. In the paper, we find substantial failures in reasoning instruction adherence: the highest instruction following score (IFS) remains below 0.25, meaning that fewer than 25\% of reasoning traces comply with the given instructions. Notably, as task difficulty increases, reasoning instruction following degrades further. We also explore two strategies to enhance reasoning instruction fidelity: (1) multi-turn reasoning and (2) Reasoning Instruction Finetuning (RIF) using synthetic data. RIF improves the IFS of GPT-OSS-20B from 0.11 to 0.27, indicating measurable progress but leaving ample room for improvement.


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

## Output

Results are saved to `outputs/[model-name]/`:
- `model_outputs_reasonIF.json`: Raw model outputs with reasoning content
- `eval_results.json`: Evaluation metrics including instruction following accuracy

We also include a simple result in the `outputs` folder. They are generatd from `DeepSeek-R1-Distill-Qwen-14B`.

