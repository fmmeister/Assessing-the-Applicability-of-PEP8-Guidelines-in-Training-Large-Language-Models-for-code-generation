import ast
import os
import re
import torch
from pycodestyle import Checker
from transformers import AutoTokenizer
from trl import PPOTrainer

from data import GeneratorDataset
from reward_function import Capturing


def log_error(message, verbose: bool) -> None:
    """
    Log an error message to the console if verbose is True.
    """
    if verbose:
        print(message)


def loading_bar(current, total, width=50) -> None:
    """
    Custom progress bar for generating test files.
    """
    progress = (current / total)
    progress_bar = "[" + "=" * int(progress * width) + " " * (width - int(progress * width)) + "]"
    percent = progress * 100
    print(f"\r{progress_bar} {percent:.1f}%", end="", flush=True)


def mbpp_sample_generieren(idx: int, device: str, dataset_test: GeneratorDataset,
                           generator: PPOTrainer, tokenizer: AutoTokenizer,
                           generation_kwargs: dict, destination_dir_path: str) -> None:
    """
    Generate code files from mbpp dataset.
    """
    prompts_tensor = dataset_test.__getitem__(idx)['prompts'].to(torch.int64).to(device)
    generation_tensor = generator.generate(query_tensor=prompts_tensor, return_prompt=False, **generation_kwargs)
    generated_txts = tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)
    with open(f"{destination_dir_path}/sample_{idx}", "w", encoding='utf-8') as file:
        file.write(generated_txts[0])


def human_eval_generieren(idx: int, generator: PPOTrainer, tokenizer: AutoTokenizer,
                          generation_kwargs: dict, device: str, destination_dir_path: str,
                          dataset_test: torch.Tensor) -> None:
    """
    Generate code files from human_eval dataset.
    """
    prompts_tensor = dataset_test[idx].to(torch.int64).to(device)
    generation_tensor = generator.generate(query_tensor=prompts_tensor, return_prompt=False, **generation_kwargs)
    generated_txts = tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)
    with open(f"{destination_dir_path}/sample_{idx}", "w", encoding='utf-8') as file:
        file.write(generated_txts[0])


def eval_pep8(directory: str) -> float:
    """
    Check all files in directory for pep8 errors.
    """
    if not os.path.isdir(directory):
        print(f"Der Pfad {directory} ist kein gültiger Ordner.")
        return -1.0

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
    return sum(num_error_list) / count


def extract_functions_from_code(code, verbose=False):
    """
    Extract all functions defined in the code using regex.
    """
    functions = []
    try:
        # Use regex to find all function definitions
        matches = re.finditer(r"def\s+(\w+)\s*\([^)]*\)\s*:", code)
        for match in matches:
            functions.append(match.group(1))  # Extract the function name
    except Exception as e:
        log_error(f"Error extracting functions from code: {e}", verbose)
    return functions


def extract_function_code(code, function_name, verbose=False):
    """
    Extract the code for a specific function from the code file.
    """
    try:
        # Use regex to find the function definition and its body
        pattern = rf"def\s+{function_name}\s*\([^)]*\)\s*:\s*((?:\n\s+.*)+)"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            # Return the entire matched function code
            return match.group(0)
        return None
    except Exception as e:
        log_error(f"Error extracting function code for {function_name}: {e}", verbose)
        return None


def can_parse_ast(code):
    """
    Check if the code can be parsed into an AST.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception as e:
        print(f"Error parsing code into AST: {e}")
        return False


def evaluate_test_case(test_case, global_namespace, verbose=False):
    """
    Evaluate a test case by extracting the boolean expression and evaluating it.
    """
    try:
        # Remove the "assert " prefix to get the boolean expression
        expression = test_case.replace("assert ", "")
        # Evaluate the expression using eval and the provided global namespace
        return eval(expression, global_namespace)
    except Exception as e:
        log_error(f"Error evaluating test case {test_case}: {e}", verbose)
        return False


def execute_code_and_check_tests(code_file, test_cases, verbose=False):
    """
    Execute the code in the file and check if all test cases pass.
    If the code file is invalid or raises an error, return False.
    """
    try:
        with open(code_file, 'r') as file:
            code = file.read()

        # Skip if the file is empty
        if not code.strip():
            log_error(f"Error: {code_file} is empty.", verbose)
            return False

        # Determine the expected function name from the first test case
        expected_function_name = test_cases[0].split("(")[0].replace("assert ", "")

        # Check if the entire file can be parsed into an AST
        if can_parse_ast(code):
            # Use the entire code for testing
            global_namespace = {}
            try:
                exec(code, global_namespace)
            except SyntaxError as e:
                log_error(f"Skipping file due to SyntaxError: {e}", verbose)
                return False
            except Exception as e:
                log_error(f"Error executing code: {e}", verbose)
                return False
        else:
            # Extract all function names from the code
            function_names = extract_functions_from_code(code, verbose)
            if not function_names:
                log_error(f"Error: No functions found in {code_file}.", verbose)
                return False

            # Check if the expected function exists in the code
            if expected_function_name not in function_names:
                log_error(f"Error: Function {expected_function_name} not found in {code_file}.", verbose)
                return False

            # Extract the code for the expected function
            function_code = extract_function_code(code, expected_function_name, verbose)
            if not function_code:
                log_error(f"Error: Could not extract code for function {expected_function_name}.", verbose)
                return False

            # Execute only the extracted function code
            global_namespace = {}
            try:
                exec(function_code, global_namespace)
            except SyntaxError as e:
                log_error(f"Skipping function due to SyntaxError: {e}", verbose)
                return False
            except Exception as e:
                log_error(f"Error executing function {expected_function_name}: {e}", verbose)
                return False

        # Check if all test cases pass with this function
        for test_case in test_cases:
            if not evaluate_test_case(test_case, global_namespace, verbose):
                return False

        return True
    except Exception as e:
        log_error(f"Error executing {code_file}: {e}", verbose)
        return False


def calculate_pass_at_1_rate(test_list, code_directory, verbose=False):
    """
    Calculate the pass@1 rate by checking if the code files pass their respective test cases.
    Skip invalid or problematic code files.
    """
    passed = 0
    total = 0

    for idx, test_cases in enumerate(test_list):
        code_file = os.path.join(code_directory, f"sample_{idx}")
        if os.path.exists(code_file):
            total += 1
            if execute_code_and_check_tests(code_file, test_cases, verbose):
                passed += 1

    if total == 0:
        return 0.0

    return (passed / total) * 100


