import os
import re
import tempfile
from io import StringIO
import sys
import codecs
from typing import List
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from pycodestyle import Checker


def get_rewards(sample_ids: torch.LongTensor, sample_texts: List[str],
                discriminator: torch.nn.Module, padding: AutoTokenizer, current_epoch: int, avg_rewards: list,
                tokenizer: AutoTokenizer, batch_indices: List[int], dataset_train: torch.utils.data.Dataset,
                dataset_test_list: torch.utils.data.Dataset, train: bool,
                dataset_test: torch.utils.data.Dataset, disc_weight: int = 1) -> tuple[torch.Tensor, List[float]]:
    sample_stack = padding.pad({"input_ids": sample_ids},
                               padding="longest", return_tensors='pt')['input_ids'].to(sample_ids[0].device)
    pred = discriminator.forward(sample_stack)
    # get predictions for positive class,
    # the more the discriminator is certain, that the sample is real (1), the higher the reward for the generator
    disc_reward = softmax(pred, dim=-1)[:, 1]
    # print(" >> rewards ", disc_reward)

    obj_rewards, avg_rewards = collect_rewards(sample_texts, discount=1, current_epoch=current_epoch,
                                               avg_rewards=avg_rewards, dataset_train=dataset_train,
                                               dataset_test_list=dataset_test_list, dataset_test=dataset_test,
                                               tokenizer=tokenizer, batch_indices=batch_indices, train=train)

    obj_rewards = obj_rewards.to(disc_reward.device)

    rewards = disc_weight * disc_reward + obj_rewards
    return rewards, avg_rewards


def collect_rewards(samples: List[str],
                    current_epoch: int,
                    avg_rewards: List[float], dataset_train: torch.utils.data.Dataset,
                    tokenizer: AutoTokenizer, dataset_test_list: torch.utils.data.Dataset,
                    batch_indices: List[int], train: bool, dataset_test: torch.utils.data.Dataset,
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
            if train:
                header_prompt = "\n# " + tokenizer.decode(
                    [dataset_train[i]['input_ids'] for i in batch_indices][k]) + "\n\n"
            else:
                header_prompt = "\n# " + tokenizer.decode(
                    [dataset_test[i]['input_ids'] for i in batch_indices][k]) + "\n\n"
            code = codecs.unicode_escape_decode(sample)[0]

            if not os.path.exists("./temp_files"):
                os.makedirs("./temp_files")
            # print("current_epoch_pep08", current_epoch)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir="./temp_files",
                                             prefix="Epoch " + str(current_epoch) + "_") as temp_file:
                temp_file.write(header_prompt.encode('utf-8'))
                temp_file.write(code.encode('utf-8'))
                temp_file_path = temp_file.name

            pep08_reward = pep08(temp_file_path)
            reward_list_during_epoch.append(pep08_reward)

            compile_reward = compilable(temp_file_path)

            combined_reward = pep08_reward + compile_reward

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


def pep08(temp_file_path: str) -> int:
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

    print(f"Pep08 Errors in file {temp_file_path}: {output}")

    if num_errors == 0:
        reward = 10
    else:
        reward = -num_errors

    return reward


def test_list(temp_file_path: str, dataset_test_list: torch.utils.data.Dataset) -> int:
    return 0

# ==================== Code Metrics =====================
# (from https://radon.readthedocs.io/en/latest/intro.html)
