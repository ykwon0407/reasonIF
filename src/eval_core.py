import argparse
import os, json
import pandas as pd
from .eval_utils import evaluate_instruction_following

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name for evaluation')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize result containers
    task_list = []   # list of parameters (e.g., language)
    source_list = []   # list of sources (e.g., aime)
    instruction_following_list = []  # instruction following in thinking
    
    model_key=args.model_name.split("/")[-1]
    json_path = f"outputs/{model_key}/model_outputs_reasonIF.json"

    # Load model outputs (JSON array) and ground truth data (JSON)
    with open(json_path, 'r') as f:
        model_outputs = json.load(f)

    # Process each item
    for model_data in model_outputs:
        task_list.append(model_data["constraint_name"][0]) 
        source_list.append(model_data["source"])
        reasoning_content = model_data["reasoning_content"][0]
        
        # Evaluate instruction following in thinking content
        response = reasoning_content
        if response.strip() != "":
            is_follow_list_think = evaluate_instruction_following(
                instruction_id_list=model_data["constraint_name"],
                parameters=model_data["constraint_args"],
                prompt=model_data["question"],
                response=response,
            )
            instruction_following_list.append(all(is_follow_list_think))
        else:
            instruction_following_list.append(False)

    # Save detailed results
    results = {
        "instruction_following_list": instruction_following_list,
        "source_list": source_list,
        "task_list": task_list,
    }

    # Save results to file
    results_file = f"outputs/{model_key}/eval_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to: {results_file}")

    # Print results
    print("\n" + "="*50)
    print("ReasonIF EVALUATION RESULTS ")
    print("MODEL: ", model_key)
    print("="*50)

    # Simple analysis
    df = pd.DataFrame(results)
    print("="*50)
    print("Model Accuracy:")
    print("="*50)
    print("Instruction Following Accuracy:")
    print(f"Overall IF Accuracy: {df['instruction_following_list'].mean():.3f}")
    print("="*50)
    print("Instruction Following Accuracy per task:")
    task_if_results = df.groupby(["task_list"])["instruction_following_list"].mean()
    for task, accuracy in task_if_results.items():
        print(f"  {task}: {accuracy:.3f}")
    print("="*50)
    print("Instruction Following Accuracy per dataset:")
    source_if_results = df.groupby(["source_list"])["instruction_following_list"].mean()
    for source, accuracy in source_if_results.items():
        print(f"  {source}: {accuracy:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()
