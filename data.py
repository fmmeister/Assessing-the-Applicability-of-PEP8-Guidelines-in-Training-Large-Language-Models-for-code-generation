import json
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead


class GeneratorDataset(Dataset):
    """
    Custom Dataset class (inherits from torch.utils.data) for the generator.
    Consists of 'prompts', 'code' and 'test_list' of the mbpp dataset.
    """
    def __init__(self, data: dict, tokenizer, padding_length):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.prompts = torch.Tensor(tokenizer(data['prompts'], padding='max_length', max_length=padding_length['prompts'])['input_ids'])
        self.code = torch.Tensor(tokenizer(data['code'], padding='max_length', max_length=padding_length['code'])['input_ids'])
        self.test_list = torch.Tensor(tokenizer(data['test_list'], padding='max_length', max_length=padding_length['test_list'])['input_ids'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompts': self.prompts[idx],
                'code': self.code[idx],
                'test_list': self.test_list[idx]}

    def select_for_discriminator(self, indices, tokenizer, padding_length, ground_truth):
        if isinstance(self.code, torch.Tensor):
            # handling bug, when prepare_data() was called two times,
            # but each time there was a different structure of self.code
            # no idea what the fuck is happening
            self.code = list(torch.unbind(Tensor.int(self.code), dim=0))
            self.code = tokenizer.batch_decode(self.code, skip_special_tokens=True)
        elif isinstance(self.code, list):
            pass
        else:
            print("Error with the structure of the code.")
        return DiscriminatorDataset({"code": [self.code[i] for i in indices],
                                     "ground_truth": [ground_truth for _ in indices]},
                                    tokenizer=tokenizer, padding_length=padding_length)


class DiscriminatorDataset(Dataset):
    """
    Custom Dataset class (inherits from torch.utils.data) for the discriminator.
    Consists of 'code' and a 'ground_truth' (1 for real data, 0 for generated data).
    """
    def __init__(self, data: dict, tokenizer, padding_length):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.code = torch.Tensor(tokenizer(data['code'], padding='max_length',
                                           max_length=padding_length['code'])['input_ids'])
        self.ground_truth = torch.Tensor(tokenizer(data['ground_truth'], padding='max_length',
                                                   max_length=padding_length['code'])['input_ids'])

    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        return {'code': self.code[idx],
                'ground_truth': self.ground_truth[idx]}

    def __add__(self, other):
        self.code = list(torch.unbind(Tensor.int(self.code), dim=0))
        self.code = self.tokenizer.batch_decode(self.code)

        other.code = list(torch.unbind(Tensor.int(other.code), dim=0))
        other.code = other.tokenizer.batch_decode(other.code)

        self.ground_truth = list(torch.unbind(Tensor.int(self.ground_truth), dim=0))
        self.ground_truth = self.tokenizer.batch_decode(self.ground_truth)

        other.ground_truth = list(torch.unbind(Tensor.int(other.ground_truth), dim=0))
        other.ground_truth = other.tokenizer.batch_decode(other.ground_truth)

        return DiscriminatorDataset(
            {"code": self.code + other.code, "ground_truth": self.ground_truth + other.ground_truth},
            self.tokenizer, self.padding_length)

    @staticmethod
    def interleave(dataset_generated, dataset_groundtruth, num_samples, tokenizer, padding_length):
        if len(dataset_generated) == len(dataset_groundtruth):
            sample_ids = random.sample(range(len(dataset_groundtruth) + len(dataset_generated)), num_samples)
            interleave_dataset = dataset_generated + dataset_groundtruth

            interleave_dataset.code = list(torch.unbind(Tensor.int(interleave_dataset.code), dim=0))
            interleave_dataset.code = tokenizer.batch_decode(interleave_dataset.code)

            interleave_dataset.ground_truth = list(torch.unbind(Tensor.int(interleave_dataset.ground_truth), dim=0))
            interleave_dataset.ground_truth = tokenizer.batch_decode(interleave_dataset.ground_truth)

            return DiscriminatorDataset({"code": [interleave_dataset.code[i] for i in sample_ids],
                                         "ground_truth": [interleave_dataset.ground_truth[i] for i in sample_ids]},
                                        tokenizer=tokenizer, padding_length=padding_length)
        else:
            return Exception


def get_index_length_datasets(train: bool, code_parrot: bool) -> dict:
    """
    Needed for max_lenth parameter for tokenization; specific to mbpp dataset.
    """
    if code_parrot:
        if train:
            return {'prompts': 49, 'code': 252, 'test_list': 302}  # for codeparrot/codeparrot-small tokenizer
        else:
            return {'prompts': 51, 'code': 402, 'test_list': 2248}
    else:
        if train:
            return {'prompts': 50, 'code': 289, 'test_list': 350}  # for Qwen/Qwen2.5-0.5B tokenizer
        else:
            return {'prompts': 47, 'code': 432, 'test_list': 3670}


def prepare_data(generator: AutoModelForCausalLMWithValueHead,
                 ground_dataset: GeneratorDataset, num_samples: int,
                 gen_args: dict, tokenizer: AutoTokenizer, train: bool,
                 code_parrot: bool) -> DiscriminatorDataset:
    """
    Called each epoch during training of the discriminator; builds new DiscriminatorDataset,
    which consists of samples from the dataset and generated samples with the current generator.
    """
    gen_args['num_return_sequences'] = num_samples

    padding_length = get_index_length_datasets(train, code_parrot)

    generated_samples = sample_from_generation(generator, gen_args, tokenizer, padding_length)
    ground_truth_samples = sample_from_dataset(ground_dataset, num_samples, tokenizer, padding_length)
    dataset = DiscriminatorDataset.interleave(generated_samples, ground_truth_samples, num_samples * 2, tokenizer, padding_length)
    return dataset


def sample_from_generation(generator: AutoModelForCausalLMWithValueHead, gen_args: dict, tokenizer: AutoTokenizer, padding_length: dict) -> DiscriminatorDataset:
    """
    Generates num_samples * samples (ground_truth = 0) with the current generator, which has been saved in the last epoch.
    """
    print(gen_args)
    with torch.no_grad():
        samples = generator.generate(**gen_args)

    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    sample_dicts = {"code": samples, "ground_truth": ["0" for _ in samples]}

    return DiscriminatorDataset(sample_dicts, tokenizer=tokenizer, padding_length=padding_length)


def sample_from_dataset(dataset: GeneratorDataset, num_samples: int, tokenizer: AutoTokenizer, padding_length: dict) -> DiscriminatorDataset:
    """
    Returns num_samples * samples (ground_truth = 1) from the dataset.
    """
    sample_ids = random.sample(range(len(dataset)), num_samples)
    samples = dataset.select_for_discriminator(sample_ids, tokenizer, padding_length, ground_truth="1")

    return samples


def load_gen_data(file_path: str, tokenizer: AutoTokenizer, train: bool, code_parrot: bool) -> GeneratorDataset:
    """
    Process dataset of the following structure to GeneratorDataset.

    Structure of dataset in json file:
    {"task_id": int, "text": str, "code": str, "test_list": List[str], "test_setup_code": str,
    "challenge_test_list": List}
    {"task_id": int, "text": str, "code": str, "test_list": List[str], "test_setup_code": str,
    "challenge_test_list": List}
    ...

    """
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    data_dicts = {"prompts": [entry["text"] for entry in data],
                  "code": [entry["code"] for entry in data],
                  "test_list": [str(entry["test_list"]) for entry in data]}

    padding_length = get_index_length_datasets(train, code_parrot=code_parrot)

    dataset = GeneratorDataset(data_dicts, tokenizer=tokenizer, padding_length=padding_length)
    return dataset
