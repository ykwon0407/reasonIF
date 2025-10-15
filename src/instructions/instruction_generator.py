import random
from .instruction_registry import INSTRUCTION_DICT

default_number_of_words={
    "aime": 860,
    "amc": 181,
    "arc": 38,
    "gpqa": 392,
    "gsm8k": 52
}

def generate_instruction(example):
    inst_id = random.choice(list(INSTRUCTION_DICT.keys()))
    cls = INSTRUCTION_DICT[inst_id]
    inst_class = cls()
    if inst_id ==  "length_constraint_checkers:number_words":
        num_words = default_number_of_words[example["source"]]
        inst_desc = inst_class.build_description(num_words=num_words)
    else:
        inst_desc = inst_class.build_description()
    inst_args = inst_class.get_constraint_args()
    
    return inst_id, inst_args, inst_desc