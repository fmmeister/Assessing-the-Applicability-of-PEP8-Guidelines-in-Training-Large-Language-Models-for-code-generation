import ast
import importlib.util
import os
from typing import List

import torch
from pycodestyle import Checker
from transformers import AutoTokenizer
from trl import PPOTrainer

from data import GeneratorDataset
from reward_function import Capturing


def loading_bar(current, total, width=50):
    """
    Custom progress bar for generating test files.
    """
    progress = (current / total)
    progress_bar = "[" + "=" * int(progress * width) + " " * (width - int(progress * width)) + "]"
    percent = progress * 100
    print(f"\r{progress_bar} {percent:.1f}%", end="", flush=True)


def mbpp_sample_generieren(idx: int, device: str, dataset_test: GeneratorDataset,
                           generator: PPOTrainer, tokenizer: AutoTokenizer,
                           generation_kwargs: dict, destination_dir_path: str):
    """
    Generate code files from mbpp dataset.
    """
    prompts_tensor = dataset_test.__getitem__(idx)['prompts'].to(torch.int64).to(device)
    prompts_txt = tokenizer.decode(prompts_tensor, skip_special_tokens=True)
    generation_tensor = generator.generate(query_tensor=prompts_tensor, return_prompt=False, **generation_kwargs)
    generated_txts = tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
    header_prompt = "\n# " + prompts_txt + "\n\n"
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)
    with open(f"{destination_dir_path}/sample_{idx}", "w", encoding='utf-8') as file:
        # file.write(header_prompt)
        file.write(generated_txts[0])


def human_eval_generieren(idx: int, generator: PPOTrainer, tokenizer: AutoTokenizer,
                          generation_kwargs: dict, device: str, destination_dir_path: str, dataset_test: torch.Tensor):
    """
    Generate code files from human_eval dataset.
    """
    prompts_tensor = dataset_test[idx].to(torch.int64).to(device)
    prompts_txt = tokenizer.decode(prompts_tensor, skip_special_tokens=True)
    generation_tensor = generator.generate(query_tensor=prompts_tensor, return_prompt=False, **generation_kwargs)
    generated_txts = tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
    header_prompt = "\n# " + prompts_txt + "\n\n"
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)
    with open(f"{destination_dir_path}/sample_{idx}", "w", encoding='utf-8') as file:
        # file.write(header_prompt)
        file.write(generated_txts[0])


def eval_pep8(directory: str):
    """
    Check all files in directory for pep8 errors.
    """
    if not os.path.isdir(directory):
        print(f"Der Pfad {directory} ist kein gültiger Ordner.")
        return

    files = os.listdir(directory)

    count = 0
    num_error_list = []
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        checker = Checker(file_path, show_source=False, show_pep8=True)
        with Capturing() as output:
            try:
                num_errors = checker.check_all()
                num_error_list.append(num_errors)
            except Exception as e:
                print(f"An error occurred while checking the code: {e}")
        count += 1
    print("Average pep8 errors found: ", sum(num_error_list) / count)


def pass_at_1(directory: str, tests: List[List[str]]):
    """
    Calculate the pass@1 rate of generated code files in a directory.
    For easier usage: pass = true, if code clears on test_list, not all the provided one's.
    """
    passes = []
    number_of_files = 0
    for k, filename in enumerate(os.listdir(directory)):
        number_of_files += 1
        file_path = os.path.join(directory, filename)
        test_list = tests[k]
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            content = "".join(lines[2:])

            module_name = "dynamic_module"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)

            exec(content, module.__dict__)
        except Exception as e:
            number_of_files -= 1  # Error loading the file

        functions = [func for func in dir(module) if callable(getattr(module, func))]
        if not functions:
            passes.append(0)  # No functions found in the file.

        for func_name in functions:
            func = getattr(module, func_name)
            if callable(func):
                target_function = func
                break
            else:
                pass

            test_list = ast.literal_eval(test_list)
            try:
                for test in test_list:
                    exec(test)
                passes.append(1)  # All tests passed!
            except AssertionError:
                passes.append(1)  # One test failed!
        else:
            passes.append(0)  # No matching function found.

    print("pass@1 rate: ", sum(passes) / number_of_files)
    return sum(passes) / number_of_files


