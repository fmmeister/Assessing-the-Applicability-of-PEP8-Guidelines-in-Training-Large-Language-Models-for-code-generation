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


def load_gen_data(file_path: str, tokenizer: AutoTokenizer) -> Dataset:
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    data_dicts = {"input_ids": [tokenizer.encode(entry["text"]) for entry in data],
                  "labels": [tokenizer.encode(entry["code"]) for entry in data]}

    dataset = Dataset.from_dict(data_dicts)
    return dataset


def load_test_list(file_path: str, tokenizer: AutoTokenizer) -> Dataset:
    with open(file_path, "r") as jfile:
        data = [json.loads(line) for line in jfile]

    test_list = {"test_list": [tokenizer.encode(str(entry["test_list"])) for entry in data]}

    test_list_dataset = Dataset.from_dict(test_list)
    return test_list_dataset
