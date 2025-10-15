import re
from .instructions.instruction_registry import INSTRUCTION_DICT

def extract_final_answer(remaining_response, source):
    """
    Extract the final answer from response (after </think> tags)
    """
    # Look for answer markers - try XML tags first, then fallback to text patterns
    if '<answer>' in remaining_response and '</answer>' in remaining_response:
        remaining_response = remaining_response.split('<answer>')[1].split('</answer>')[0]
    else:
        # Look for "ANSWER:", "Answer:", or "answer:" patterns
        answer_patterns = [r'ANSWER:\s*(.*)', r'Answer:\s*(.*)', r'answer:\s*(.*)']
        for pattern in answer_patterns:
            match = re.search(pattern, remaining_response, re.IGNORECASE | re.DOTALL)
            if match:
                remaining_response = match.group(1)
                break

    
    remaining_response = remaining_response.strip()

    if source in ["gsm8k", "amc", "aime"]:
        matches = re.findall(r'([+-]?\d*\.?\d+)', remaining_response)
        if matches:
            answer = matches[-1].strip()
            try:
                return str(int(answer))
            except:
                return str(answer)
        else:
            return remaining_response
        
    elif source in ["arc", "gpqa"]:
        # If no clear choice found, look for any single capital letter A-D
        single_letters = re.findall(r'[ABCD]', remaining_response)
        if single_letters:
            return single_letters[0]
        else:    
            return remaining_response.strip()
    else:
        assert False, f"Unsupported source: {source}. Supported sources are 'gsm8k' and 'arc'"
        

def evaluate_instruction_following(
    instruction_id_list,
    parameters,
    prompt,
    response,
):
    """Tests response to see if instructions are followed."""
    is_following_list = []
    for index, instruction_id in enumerate(instruction_id_list):
        try:
            instruction_cls = INSTRUCTION_DICT[instruction_id]
        except:
            print(f"Warning: Unknown instruction ID {instruction_id}")
            is_following_list.append(False)
            continue
            
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.  
        if parameters[index]:
            kwargs = {n: p for n, p in parameters[index].items() if p}
        else:
            kwargs = {}
            
        instruction.build_description(**kwargs)
        args_dict = instruction.get_constraint_args()
        if args_dict and "prompt" in args_dict:
            instruction.build_description(prompt=prompt)
        
        try:
            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)
        except Exception as e:
            print(f"Error checking instruction {instruction_id}: {e}")
            is_following_list.append(False)

    return is_following_list
