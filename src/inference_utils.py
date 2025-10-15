"""
Simple inference utilities using vLLM for local GPU inference.
"""
import os
import logging
from typing import Optional, List, Dict

from vllm import LLM, SamplingParams

def extract_reasoning_and_content(full_response: str) -> tuple[str, str]:
    """Extract reasoning and content from model response."""
    start_delim, end_delim = "<think>", "</think>"
    
    if start_delim in full_response and end_delim in full_response:
        start_idx = full_response.find(start_delim)
        end_idx = full_response.find(end_delim)
        reasoning_content = full_response[start_idx + len(start_delim):end_idx].strip()
        content = full_response[end_idx + len(end_delim):].strip()
    else:
        # No thinking tags found, assume entire response is reasoning
        reasoning_content = full_response
        content = ""
    
    return reasoning_content, content


class MockOutputText:
    """Mock output text object for compatibility."""
    def __init__(self, reasoning_content: str, content: str):
        self.reasoning_content = reasoning_content
        self.content = content


class MockOutput:
    """Mock output object for compatibility."""
    def __init__(self, request_id: int, outputs: List[MockOutputText]):
        self.request_id = request_id
        self.outputs = outputs


def run_inference(
    messages_list: List[List[Dict]], 
    model_name: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    **kwargs
) -> List[MockOutput]:
    """
    Run inference on a list of messages using vLLM for local GPU inference.
    
    Args:
        messages_list: List of message lists for inference
        model_name: Model identifier (HuggingFace model path or name)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        **kwargs: Additional inference parameters
    
    Returns:
        List of MockOutput objects
    """
    print(f'>>>>>> Starting vLLM inference for {len(messages_list)} prompts')
    print(f"Using model: {model_name}")
    
    # Initialize vLLM model
    try:
        llm = LLM(model=model_name,tensor_parallel_size=8, **kwargs)
        print(f">>>>>> Model loaded successfully")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        # Return empty outputs as fallback
        return [MockOutput(i, [MockOutputText("", "")]) for i in range(len(messages_list))]
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Convert messages to prompts (assuming chat format for now)
    prompts = []
    for messages in messages_list:
        if len(messages) == 1 and messages[0]["role"] == "user":
            prompts.append(messages[0]["content"])
        else:
            # For multi-turn conversations, concatenate them
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
                elif msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}")
            prompts.append("\n".join(prompt_parts))
    
    print(f">>>>>> Running inference on {len(prompts)} prompts...")
    
    # Run batch inference
    try:
        vllm_outputs = llm.generate(prompts, sampling_params)
        print(f'>>>>>> Completed inference for all prompts')
    except Exception as e:
        print(f"Error during vLLM inference: {e}")
        # Return empty outputs as fallback
        return [MockOutput(i, [MockOutputText("", "")]) for i in range(len(messages_list))]
    
    # Convert vLLM outputs to MockOutput format
    outputs = []
    for i, vllm_output in enumerate(vllm_outputs):
        try:
            full_response = vllm_output.outputs[0].text
            reasoning_content, content = extract_reasoning_and_content(full_response)
            
            mock_outputs = [MockOutputText(reasoning_content, content)]
            outputs.append(MockOutput(i, mock_outputs))
            
        except Exception as e:
            print(f"Error processing output for prompt {i}: {e}")
            # Create empty output as fallback
            mock_outputs = [MockOutputText("", "")]
            outputs.append(MockOutput(i, mock_outputs))
    
    print('>>>>>> Generation done')
    return outputs
