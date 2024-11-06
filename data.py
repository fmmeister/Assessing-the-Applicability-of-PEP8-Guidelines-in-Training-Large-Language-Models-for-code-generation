import json
import random

import torch
from datasets import Dataset, interleave_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

def prepare_data(generator: AutoModelForCausalLM, 
                 ground_dataset: Dataset, num_samples: int, 
                 gen_args: dict) -> Dataset:

    gen_args['num_return_sequences'] = num_samples
    gen_dataset = create_gen_dataset(generator, gen_args)

    ground_samples = sample_from_dataset(ground_dataset, num_samples)

    dataset = interleave_datasets([gen_dataset, ground_samples],
                                  stopping_strategy="all_exhausted")

    return dataset


def create_gen_dataset(generator: AutoModelForCausalLM, gen_args: dict) -> Dataset:

    with torch.no_grad():
        samples = generator.generate(**gen_args)

    sample_dicts = [{"input_ids": sample,
                     'labels': 0} for sample in samples]

    return Dataset.from_list(sample_dicts)


def sample_from_dataset(dataset: Dataset, num_samples: int) -> Dataset:

    sample_ids = random.sample(range(len(dataset)), num_samples)

    samples = dataset.select(sample_ids)
    samples = samples.remove_columns('labels')
    samples = samples.add_column(name='labels', column=[1] * num_samples)
    return samples


def load_gen_data(file_path: str, tokenizer: AutoTokenizer, block_size: int=128) -> Dataset:
    with open(file_path, "r") as jfile:
        data = [json.loads(line) for line in jfile]

    data_dicts = [{"input_ids": tokenizer.encode(entry["code"]),
                   "labels": tokenizer.encode(entry["code"])} for entry in data]

    dataset = Dataset.from_list(data_dicts)
    return dataset.map(lambda batch: _group_texts(batch, block_size), batched=True)


def _group_texts(samples: dict, block_size: int) -> dict:
    # Concatenate all texts.
    concatenated_samples = {k: sum(samples[k], []) for k in samples.keys()}
    total_length = len(concatenated_samples[list(samples.keys())[0]])

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_samples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result
