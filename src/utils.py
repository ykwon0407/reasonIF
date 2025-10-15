"""
Data utilities for loading and processing datasets.
"""
import json, os, random
import pandas as pd
import copy
from .instructions.instruction_util import count_words

def prepare_message_list(model_key, input_path = f"data/reasonIF_dataset.json"):
    """
    Prepare message list from dataset, updating word limits if needed.
    
    Args:
        dataset: List of dataset items
        model_key: Model identifier key for word limit lookup
        number_of_words_dict: Dictionary containing word limits per model
        
    Returns:
        tuple: (messages_list, updated_dataset)
    """
    # Load word limit reference if it exists
    number_of_words_dict = {}
    try:
        with open("data/number_of_words_reference.json", 'r') as f:
            number_of_words_dict = json.load(f)
    except FileNotFoundError:
        print("No word limit reference file found, using default limits")

    print(f"Loading dataset from: {input_path}")
    dataset = load_json(input_path) 
    print(f"Loaded {len(dataset)} examples")

    tmp_dataset = copy.deepcopy(dataset)
    messages_list = []
    
    for item in tmp_dataset:
        if (item["constraint_name"][0] == "length_constraint_checkers:number_words" and 
            model_key in number_of_words_dict):
            # Update the num_words value
            new_num_words = int(number_of_words_dict[model_key][item["source"]])
            item["constraint_args"][0]["num_words"] = new_num_words
            item["prompt"] = replace_word_limit(item["prompt"], new_num_words)
        messages_list.append([{"role": "user", "content": item["prompt"]}])
    
    return messages_list, tmp_dataset

class SimpleDataset:
    """Simple dataset wrapper for JSONL files."""
    
    def __init__(self, data):
        self.data = data
        self.column_names = list(data[0].keys()) if data else []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_json(file_path):
    """Load JSON file and return SimpleDataset."""
    with open(file_path, 'r') as f:
        data=json.load(f)
    return SimpleDataset(data)    

def create_prompt_from_data(example):
    """Convert dataset to messages format for LLM inference."""
    prompt_template_reasonIF="""Think step-by-step, and place only your final answer inside the tags `<answer>` and `</answer>`. Format your reasoning according to the following rule: **{constraint_text}**

Here is the question:

{question_statement}"""
        
    question_statement=example['question']
    constraint_text = "".join(example["constraint_desc"])
    full_content = prompt_template_reasonIF.format(constraint_text=constraint_text, question_statement=question_statement)
        
    return full_content    

import re

def replace_word_limit(text: str, new_limit: int) -> str:
    """
    Replace the numeric word‑limit in the phrase
    "When reasoning, respond with less than <number> words"
    with ``new_limit``.
    """
    pattern = r'(?<=less than )\d+(?= words)'
    return re.sub(pattern, str(new_limit), text)
    
def create_output_path(model_key):
    """Create output file path based on arguments."""
    output_path = "outputs/{}/model_outputs_reasonIF.json".format(model_key)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path

def save_results(output_path, inputs, outputs, dataset, n_sample=1):
    """Save inference results to file as a single JSON array."""
    assert len(inputs) == len(outputs), "Check length"
    questions=[data['question'] for data in dataset]
    answers=[data['answer'] for data in dataset]
    sources=[data['source'] for data in dataset]
    hf_ids=[data['hf_id'] for data in dataset]
    constraint_names=[data['constraint_name'] for data in dataset]
    constraint_argss=[data['constraint_args'] for data in dataset]
    
    results = []
    for i, output in enumerate(outputs):
        results.append({
            "id": i,
            "hf_id": hf_ids[i],
            "question": questions[i],
            "answer": answers[i],
            "source": sources[i],
            "constraint_name": constraint_names[i],
            "constraint_args": constraint_argss[i],
            "input": inputs[i],
            "reasoning_content": [output.outputs[j].reasoning_content for j in range(n_sample)],
            "content": [output.outputs[j].content for j in range(n_sample)]
        })
    
    # Write as a single JSON array
    with open(output_path, 'w', encoding='utf8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
