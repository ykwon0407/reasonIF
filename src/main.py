"""
Main inference script for reasonIF - Simple litellm-based inference.
"""
import argparse
import json
from .utils import prepare_message_list, create_output_path, save_results
from .inference_utils import run_inference

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM inference for reasoning IF evaluation")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=16384)
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    model_key = args.model_name.split("/")[-1]

    # Prepare messages list with word limit updates
    messages_list, updated_dataset = prepare_message_list(model_key)  

    # Run inference
    outputs = run_inference(
        messages_list=messages_list,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # Save results
    output_path = create_output_path(model_key)
    inputs=[message[0]["content"] for message in messages_list]
    save_results(output_path, inputs, outputs, updated_dataset)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
