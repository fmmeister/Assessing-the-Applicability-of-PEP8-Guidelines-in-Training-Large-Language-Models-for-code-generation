import ast
import importlib.util
import os
import re
import tempfile
from io import StringIO
import sys
import codecs
from typing import List
import torch
from torch import Tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pycodestyle import Checker

from data import GeneratorDataset, DiscriminatorDataset


def get_rewards(generation_text: List[str], device, test_list: List[str],
                discriminator: torch.nn.Module, current_epoch: int, avg_rewards: list,
                tokenizer: AutoTokenizer, prompts: List[str], padding_length: dict,
                disc_weight: int = 1) -> tuple[torch.Tensor, List[float]]:

    samples = Tensor.int(torch.Tensor(tokenizer(generation_text, padding='max_length', max_length=padding_length['code'])['input_ids']).to(device))
    discriminator.to(device)
    pred = discriminator.forward(samples)
    # get predictions for positive class,
    # the more the discriminator is certain, that the sample is real (1), the higher the reward for the generator
    disc_reward = softmax(pred, dim=-1)[:, 1]
    # print(" >> rewards ", disc_reward)

    obj_rewards, avg_rewards = collect_rewards(generation_text, discount=1, current_epoch=current_epoch,
                                               avg_rewards=avg_rewards, prompts=prompts, test_list=test_list)

    obj_rewards = obj_rewards.to(disc_reward.device)

    rewards = disc_weight * disc_reward + obj_rewards
    return rewards, avg_rewards


def collect_rewards(samples: List[str],
                    current_epoch: int,
                    avg_rewards: List[float],
                    prompts: List[str], test_list: List[str],
                    discount: float = 1.0) -> tuple[torch.Tensor, List[float]]:
    """
    Compute the weighted sum specified metrics for the list of samples.

    :param samples: list of strings of code produced by a generative model
    :param objectives: list of objective functions return a reward each
    :param discount: float in (0, 1]; in case the samples represent successive steps, the importance of the later steps can be reduced; 1 for no discount
    (I use this in the reinforcement learning as implemented in SeqGan)
    :return: list of rewards for the given samples as float
    """
    collected = torch.zeros(len(samples))
    reward_list_during_epoch = []
    for k, sample in enumerate(samples):

        try:
            header_prompt = "\n# " + prompts[k] + "\n\n"
            code = codecs.unicode_escape_decode(sample)[0]

            if not os.path.exists("./temp_files"):
                os.makedirs("./temp_files")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir="./temp_files",
                                             prefix="Epoch " + str(current_epoch) + "_") as temp_file:
                temp_file.write(header_prompt.encode('utf-8'))
                temp_file.write(code.encode('utf-8'))
                temp_file_path = temp_file.name

            pep08_reward, output = pep08(temp_file_path)  # pep8 reward and output

            # compile_reward = compilable(temp_file_path)  # compilable reward

            test_list_reward = try_test_list(temp_file_path, test_list[k])  # test_list reward

            with open(temp_file_path, "a") as temp_file:
                temp_file.write("\n\n# Errorcodes: " + str(output) + "\n# pep08_reward: " + str(pep08_reward) +
                                "\n# test_list_reward: " + str(test_list_reward))

            combined_reward = pep08_reward + test_list_reward

            reward_list_during_epoch.append(combined_reward)
            rewards = torch.tensor(combined_reward)
            collected[k] = torch.sum(torch.tensor(rewards)) * torch.pow(torch.tensor(discount), torch.tensor(k))
        except UnicodeDecodeError:
            collected[k] = torch.tensor(-1)

    if not reward_list_during_epoch:
        print("No rewards collected")
    else:
        print("Average Rewards collected:", sum(reward_list_during_epoch) / len(reward_list_during_epoch))
        avg_rewards.append(sum(reward_list_during_epoch) / len(reward_list_during_epoch))
    return collected, avg_rewards


def compilable(temp_file_path: str) -> int:
    """Can the code be compiled by the standard python compiler"""
    try:
        compile(temp_file_path, "<string>", "exec")
        reward = 10
    except SyntaxError:
        reward = -1
    except ValueError:
        reward = -1

    return reward


class Capturing(list):
    """Context Manager to capture pycodestyle's print statements"""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.pep08_output = self._stringio.getvalue()
        del self._stringio
        sys.stdout = self._stdout
        # Filter specific error codes and store them in the list
        error_code_pattern = r'\b[EWF]\d{3}\b'  # Regex for error codes like E123, W456, F789
        self.extend(re.findall(error_code_pattern, self.pep08_output))


def pep08(temp_file_path: str) -> tuple[int, List[str]]:
    """
    How much does the code adhere to pep08 standards.
    Could potentially be more elaborated by utilizing the specific messages in output.
    """
    checker = Checker(temp_file_path, show_source=False, show_pep8=True)
    num_errors = 10  # default value to avoid stopping the training when hitting exception
    with Capturing() as output:
        try:
            num_errors = checker.check_all()
        except Exception as e:
            print(f"An error occurred while checking the code: {e}")

    # print(f"Pep08 Errors in file {temp_file_path}: {output}")

    if num_errors == 0:
        reward = 100
    else:
        reward = -num_errors

    return reward, output


def try_test_list(temp_file_path: str, test_list: str) -> int:

    spec = importlib.util.spec_from_file_location(temp_file_path, "./temp_files/" + temp_file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return -4  # Error loading the file

    functions = [func for func in dir(module) if callable(getattr(module, func))]
    if not functions:
        return -3  # No functions found in the file.

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
            return 100  # All tests passed!
        except AssertionError:
            return -1  # A test failed!
    else:
        return -2  # No matching function found.


# ==================== Code Metrics =====================
# (from https://radon.readthedocs.io/en/latest/intro.html)
