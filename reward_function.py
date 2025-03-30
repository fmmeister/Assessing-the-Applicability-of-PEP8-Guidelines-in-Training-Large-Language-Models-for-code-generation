import ast
import importlib.util
import re
from io import StringIO
import sys
from typing import List
import torch
from torch import Tensor
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from pycodestyle import Checker


def get_rewards(generation_tensor: List[torch.Tensor],
                generation_text: List[str],
                device,
                discriminator: torch.nn.Module,
                avg_rewards_during_batches: list,
                tokenizer: AutoTokenizer,
                padding_length: dict,
                disc_weight: int,
                temp_file_path: List[str],
                test_list: List[str]) -> tuple[torch.Tensor, List[float]]:
    """
    Collect general reward for the generator (combination of discriminator reward and objective reward).
    """
    samples = Tensor.int(torch.Tensor(
        tokenizer(generation_text, padding='max_length', max_length=padding_length['code'])['input_ids']).to(device))
    discriminator.to(device)
    pred = discriminator.forward(samples)
    # get predictions for positive class,
    # the more the discriminator is certain, that the sample is real (1), the higher the reward for the generator
    disc_reward = softmax(pred, dim=-1)[:, 1]

    # mapping from [0, 1] to [-1, 1]
    disc_reward_transformation = disc_reward * 2 - 1

    if disc_weight == 1:
        rewards = disc_reward_transformation
        avg_rewards = disc_reward_transformation.tolist()
    elif disc_weight == 0:
        obj_rewards, avg_rewards = collect_rewards(generation_tensor,
                                                   generation_text,
                                                   discount=1,
                                                   avg_rewards_during_batches=avg_rewards_during_batches,
                                                   temp_file_path=temp_file_path,
                                                   test_list=test_list)

        rewards = obj_rewards.to(disc_reward_transformation.device)
    else:
        obj_rewards, avg_rewards = collect_rewards(generation_tensor,
                                                   generation_text,
                                                   discount=1,
                                                   avg_rewards_during_batches=avg_rewards_during_batches,
                                                   temp_file_path=temp_file_path,
                                                   test_list=test_list)

        obj_rewards = obj_rewards.to(disc_reward_transformation.device)

        rewards = disc_weight * disc_reward_transformation + (1 - disc_weight) * obj_rewards

    return rewards, avg_rewards


def collect_rewards(generation_tensor: List[torch.Tensor],
                    generation_text: List[str],
                    avg_rewards_during_batches: List[float],
                    temp_file_path: List[str],
                    test_list: List[str],
                    discount: float = 1.0) -> tuple[torch.Tensor, List[float]]:
    """
    Collects rewards of the objective function (pep8 / try_test_list).
    """
    collected = torch.zeros(len(generation_text))
    reward_list_during_epoch = []
    for k, sample in enumerate(temp_file_path):

        try:
            pep08_reward, output = pep08(sample)  # pep8 reward and output

            with open(sample, "a+") as temp_file:
                code = temp_file.read()
                temp_file.write("\n\n# Errorcodes: " + str(output) + "\n# pep08_reward: " + str(pep08_reward))
            # test_list_reward = try_test_list(sample, test_list[k])

            length_penalty = (generation_tensor[k].shape[0] / 100) - 1
            # looks up token length of generated code and maps it to [-1, 1] (max_new_token_length = 200)

            reward_list_during_epoch.append(pep08_reward + length_penalty)
            rewards = torch.tensor(pep08_reward)
            collected[k] = torch.sum(torch.tensor(rewards)) * torch.pow(torch.tensor(discount), torch.tensor(k))
        except UnicodeDecodeError:
            collected[k] = torch.tensor(-1)

    if not reward_list_during_epoch:
        print("No rewards collected")
    else:
        avg_rewards_during_batches.append(sum(reward_list_during_epoch) / len(reward_list_during_epoch))
    return collected, avg_rewards_during_batches


def compilable(temp_file_path: str) -> int:
    """
    Can the code be compiled by the standard python compiler.
    """
    try:
        compile(temp_file_path, "<string>", "exec")
        reward = 100
    except SyntaxError:
        reward = -5
    except ValueError:
        reward = -5

    return reward


class Capturing(list):
    """
    Context Manager to capture pycodestyle's print statements.
    """

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
    """
    checker = Checker(temp_file_path, show_source=False, show_pep8=True)
    num_errors = 15
    with Capturing() as output:
        try:
            num_errors = checker.check_all()
        except Exception as e:
            print(f"An error occurred while checking the code: {e}")

    if num_errors == 0:
        reward = 1
    elif num_errors <= 3 & num_errors > 0:
        reward = 0.5
    elif num_errors <= 5 & num_errors > 3:
        reward = 0.25
    elif num_errors <= 10 & num_errors > 5:
        reward = -0.5
    elif num_errors <= 15 & num_errors > 10:
        reward = -0.75
    else:
        reward = -1

    return reward, output


def try_test_list(temp_file_path: str, test_list: str) -> float:
    """
    Function tries to find a function in the generated file and uses 'test_list' from mbpp dataset to evaluate the code.
    """
    spec = importlib.util.spec_from_file_location(temp_file_path, "./temp_files/" + temp_file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return -1.0  # Error loading the file

    functions = [func for func in dir(module) if callable(getattr(module, func))]
    if not functions:
        return -0.5  # No functions found in the file.

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
            return 1  # All tests passed!
        except AssertionError:
            return 0.2  # A test failed!
    else:
        return -0.25  # No matching function found.

# ==================== Code Metrics =====================
# (from https://radon.readthedocs.io/en/latest/intro.html)
