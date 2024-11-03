from io import StringIO
from typing import Callable, List, Tuple, Union
import sys
import codecs
import re
from typing import List

from numpy import log, log2, sqrt, radians
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from pycodestyle import Checker


def get_rewards(sample_ids: torch.LongTensor, sample_texts: List[str],
                discriminator: torch.nn.Module, padding: AutoTokenizer,
                objectives: Union[List[str], None] = None, weights: Union[torch.Tensor, None] = None,
                disc_weight: int = 1) -> torch.Tensor:
    sample_stack = padding.pad({"input_ids": sample_ids},
                               padding="longest", return_tensors='pt')['input_ids'].to(sample_ids[0].device)
    pred = discriminator.forward(sample_stack)
    # get predictions for positive class,
    # the more the discriminator is certain, that the sample is real (1), the higher the reward for the generator
    disc_reward = softmax(pred, dim=-1)[:, 1]
    print(" >> rewards ", disc_reward)

    if objectives:
        # OBJ is defined at bottom
        obj_functions = [f for key, f in OBJ if key in objectives]
        obj_rewards = collect_rewards(sample_texts, obj_functions, weights, 1)
    else:
        obj_rewards = torch.zeros(len(sample_stack))
    obj_rewards = obj_rewards.to(disc_reward.device)

    rewards = disc_weight * disc_reward + obj_rewards
    return rewards


class Capturing(list):
    """Context Manager to capture pycodestyle's print statements"""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def collect_rewards(samples: List[str],
                    objectives: List[Callable],
                    weights: torch.Tensor,
                    discount: float = 1.0) -> torch.Tensor:
    """
    Compute the weighted sum specified metrics for the list of samples.

    :param samples: list of strings of code produced by a generative model
    :param objectives: list of objective functions return a reward each
    :param weights: torch.Tensor of weights for the objectives in the same order
    :param discount: float in (0, 1]; in case the samples represent successive steps, the importance of the later steps can be reduced; 1 for no discount
    (I use this in the reinforcement learning as implemented in SeqGan)
    :return: list of rewards for the given samples as float
    """

    collected = torch.zeros(len(samples))
    for k, sample in enumerate(samples):

        try:
            code = codecs.unicode_escape_decode(sample)[0]
            rewards = torch.tensor([objective(code) for objective in objectives])
            weighted_rewards = rewards * weights
            collected[k] = torch.sum(weighted_rewards) * torch.pow(torch.tensor(discount), torch.tensor(k))
        except UnicodeDecodeError:
            collected[k] = torch.tensor(-1)

    return collected


def compilable(code: str) -> int:
    """Can the code be compiled by the standard python compiler"""
    try:
        compile(code, "<string>", "exec")
        reward = 1
    except SyntaxError:
        reward = -1
    except ValueError:
        reward = -1

    return reward


def pass_test(code: str, test: str) -> int:
    # ToDo: Consider Sandboxing somehow
    # ToDo: adapt as in CodeEval
    code_test = code + "\r\n" + test
    try:
        compiled = compile(code_test, "<string>", "exec")
        exec(test)
        reward = 1
    except SyntaxError:
        reward = -1
    except ValueError:
        reward = -1

    return reward


def pep08(code: str) -> int:
    """
    How much does the code adhere to pep08 standards.
    Could potentially be more elaborated by utilizing the specific messages in output.
    """
    checker = Checker(lines=[code], show_source=False)
    with Capturing() as output:
        num_errors = checker.check_all()

    return -num_errors


# ==================== Code Metrics =====================
# (from https://radon.readthedocs.io/en/latest/intro.html)

def maintainability(code: str) -> float:
    """
    The maintainability index is calculated as a factored formula
    consisting of SLOC (Source Lines Of Code), Cyclomatic Complexity and Halstead volume.

    The index is in [0, 100] where a higher value implies better maintainability, i.e. 'better code'.
    """
    code = code.replace("\r", "")

    v = _halstead_vol(code)
    g = _cyclo_comp(code)
    sloc, c = _sloc(code)

    mi = max(
        0,
        (100 * (171 - 5.2 * log(v) - 0.23 * g - 16.2 * log(sloc) + 50 * sqrt(2.4 * c)) / 171)
    )

    return mi


def _cyclo_comp(code: str) -> int:
    """cyclomatic complexity /McCabe number"""

    n_cont = len(re.findall(
        r'(\nif|^if|\nelif|\nfor|^for|\nwhile|^while'
        r'|\nexcept|^except|\nwith|^with|n\assert|^assert)',
        code))
    n_list_comp = len(re.findall(r'\[.+ for .+ in .+]', code))
    n_bool_op = len(re.findall(
        r'(is|not|is not) | in | and | or |==|!=x', code))
    cc = n_cont + n_list_comp + n_bool_op + 1

    return cc


def _halstead_vol(code: str) -> float:
    """"Halstead Volume Metric"""
    tokens = code.strip().split()
    vocabulary = set(tokens)
    length = len(tokens)

    volume = length * log2(vocabulary)
    return volume


def _sloc(code: str) -> Tuple[int, float]:
    """number of source lines of code and percentage of comment lines"""
    loc_lines = code.split("\n")
    sloc = 0
    comments = 0
    for line in loc_lines:
        stripped = line.strip()
        if not (stripped.startswith("#")
                or stripped.startswith('"')
                or stripped.startswith("'")
                or stripped == ""):
            sloc += 1
        elif not stripped == "":
            comments += 1

    comments = radians(comments * 100 / len(loc_lines))

    return sloc, comments


OBJ = {"compilable": compilable, "pep08:": pep08, "maintainability": maintainability}
