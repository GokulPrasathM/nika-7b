import os
import sys
import time
from typing import Optional, Sequence
from glob import glob
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser

from data.utils.io_utils import question_hash, jdump, jload

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@dataclass
class DataModuleConfigs:
    model_name: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", metadata={'help': 'Model name'})
    shard_index: int = field(default=0, metadata={'help': 'Shard index'})

def shard_question(chunk_size: int = 10_000):
    questions = load_dataset("BeastGokul/s2")['train']['question']
    for i in tqdm(range(0, len(questions), chunk_size), desc="Sharding questions"):
        shard = questions[i:i + chunk_size]
        jdump(shard, f"results/difficulty_classification/r1_inference/shard_{i // chunk_size}_input.json")

def _qwen_forward(prompts: Sequence[str], model_name: str, tokenizer_path: str, max_length: int = 32768, temperature: float = 0.05) -> Optional[Sequence[str]]:
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_length)
    tensor_parallel_size = 1 if "7B" in model_name else 2
    
    model = None
    while model is None:
        try:
            model = LLM(model=model_name, tokenizer=tokenizer_path, tensor_parallel_size=tensor_parallel_size)
        except Exception as e:
            print(f"Error loading model: {e}")
            time.sleep(10)
    
    outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]

def difficulty_classification(shard_index: int, model_name: str):
    pretty_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    questions = jload(f"results/difficulty_classification/{pretty_name}/shard_{shard_index}_input.json")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [f"{tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False)}<|im_start|>assistant\n" for question in tqdm(questions, desc="Tokenizing questions")]
    
    results = _qwen_forward(prompts, model_name, model_name)
    result_dict = {question_hash(q): r for q, r in zip(questions, results)}
    
    jdump(result_dict, f"results/difficulty_classification/{pretty_name}/shard_{shard_index}_output.json")

def assemble_output(model_name: str, upload: bool = False):
    pretty_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    output = {}
    
    for shard_index in tqdm(range(7), desc="Loading shard outputs"):
        output.update(jload(f"results/difficulty_classification/{pretty_name}/shard_{shard_index}_output.json"))
    
    dataset = load_dataset("BeastGokul/s2")['train']
    key_map_dataset = {question_hash(ex['question']): ex for ex in tqdm(dataset, desc="Mapping dataset to hash")}
    
    result = []
    for qhash, attempt in tqdm(output.items(), desc="Creating output JSON"):
        if qhash in key_map_dataset:
            example = {"question": key_map_dataset[qhash]['question'], "solution": key_map_dataset[qhash]['solution'], "attempt": attempt}
            jdump(example, f"results/difficulty_classification/{pretty_name}/grading_input/{qhash}.json")
            result.append(example)
    
    if upload:
        new_dataset = [{**ex, pretty_name: output[question_hash(ex['question'])]} for ex in tqdm(dataset, desc="Uploading dataset")]
        Dataset.from_list(new_dataset).push_to_hub(f"qfq/train_{pretty_name}_inference")
    
    jdump(result, f"results/difficulty_classification/{pretty_name}/inference_output.json")

if __name__ == "__main__":
    shard_question()
    parser = HfArgumentParser(DataModuleConfigs)
    difficulty_classification(**asdict(parser.parse_args_into_dataclasses()[0]))
    
    assemble_output("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    assemble_output("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
