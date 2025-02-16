import os
import random

import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import Tensor, softmax
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import DiscriminatorDataset, get_index_length_datasets, sample_from_dataset, load_gen_data, GeneratorDataset
from discriminator import CNNDiscriminator

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
discriminator = CNNDiscriminator(embed_dim=64, vocab_size=len(tokenizer),
                                 filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                                 num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                                 padding_idx=tokenizer.pad_token_id, gpu=True,
                                 dropout=0.25)
file_path = "./finished_models/discriminator.pt"
state_dict = torch.load(file_path, weights_only=True)
discriminator.load_state_dict(state_dict)
discriminator.to(device)
adv_loss = torch.nn.CrossEntropyLoss()

dataset_train = load_gen_data(os.path.join("train.json"), tokenizer, train=True)


# dataset_test = load_gen_data(os.path.join("test.json"), tokenizer, train=False)


def process_tempfiles_to_dataset(directory: str, tokenizer: AutoTokenizer):
    data_dicts = {"prompts": [],
                  "code": [],
                  "test_list": []}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Open and read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()

                data_dicts['prompts'].append("")
                data_dicts['code'].append(content)
                data_dicts['test_list'].append("")

    def longest_value_token_length(dictionary):
        lengths = {}
        for key, values in dictionary.items():
            if hasattr(values, '__iter__') and not isinstance(values, (str, bytes)):
                lengths[key] = max(len(tokenizer(value)["input_ids"]) for value in values)
            elif isinstance(values, str):
                lengths[key] = len(tokenizer(values)["input_ids"])
            else:
                lengths[key] = 0
        return lengths

    padding_length = longest_value_token_length(data_dicts)
    print(padding_length)
    dataset = GeneratorDataset(data_dicts, tokenizer=tokenizer, padding_length=padding_length)
    return dataset


dir = "./finished_models/pretrained_disc_plus_test_list/temp_files"
dataset_test = process_tempfiles_to_dataset(dir, tokenizer)


def sample_from_test_data(dataset: GeneratorDataset, num_samples: int, tokenizer: AutoTokenizer,
                          padding_length: dict) -> DiscriminatorDataset:
    sample_ids = random.sample(range(len(dataset)), num_samples)
    samples = dataset.select_for_discriminator(sample_ids, tokenizer, padding_length, ground_truth="0")

    return samples


def prepare_data_test_overfitting(tokenizer: AutoTokenizer, train: bool,
                                  num_samples: int = 100) -> DiscriminatorDataset:
    padding_length = get_index_length_datasets(train)

    train_samples = sample_from_dataset(dataset_train, num_samples, tokenizer, padding_length)
    test_samples = sample_from_test_data(dataset_test, num_samples, tokenizer, padding_length)
    dataset = DiscriminatorDataset.interleave(train_samples, test_samples, num_samples * 2, tokenizer, padding_length)
    return dataset


dataset_disc = prepare_data_test_overfitting(tokenizer=tokenizer, train=False)
data = DataLoader(dataset=dataset_disc, batch_size=64, shuffle=True, drop_last=True)

# Initialize variables to track loss and accuracy for both classes
total_loss = 0.0
total_correct_class_0 = 0  # Correct predictions for class 0
total_correct_class_1 = 0  # Correct predictions for class 1
total_class_0 = 0  # Total samples for class 0
total_class_1 = 0  # Total samples for class 1
all_trues = []
all_pred = []

for batch in data:
    samples = Tensor.int(batch["code"]).to(device)
    classes = batch["ground_truth"]
    classes = list(torch.unbind(Tensor.int(classes), dim=0))
    classes_decoded_str = tokenizer.batch_decode(classes, skip_special_tokens=True)
    classes_decoded_int = torch.tensor([int(item) for item in classes_decoded_str], device=device,
                                       dtype=torch.int64)

    classification = discriminator.forward(samples)
    disc_reward = softmax(classification, dim=-1)[:, 1]

    loss = adv_loss(classification, classes_decoded_int)
    total_loss += loss.item()

    predictions = classification.argmax(dim=-1)

    # Update accuracy for each class
    for pred, true_class in zip(predictions, classes_decoded_int):
        if true_class == 0:
            total_class_0 += 1
            if pred == true_class:
                total_correct_class_0 += 1
        elif true_class == 1:
            total_class_1 += 1
            if pred == true_class:
                total_correct_class_1 += 1

    all_pred.extend([tensor.item() for tensor in predictions])
    all_trues.extend([tensor.item() for tensor in classes_decoded_int])

total_loss /= len(data)
acc_class_0 = total_correct_class_0 / total_class_0 if total_class_0 > 0 else 0
acc_class_1 = total_correct_class_1 / total_class_1 if total_class_1 > 0 else 0

print(f"Total Loss: {total_loss}")
print(f"Accuracy for Class Test Data: {acc_class_0:.4f}")
print(f"Accuracy for Class Real Train Data: {acc_class_1:.4f}")

# Confusion matrix
cm = confusion_matrix(all_trues, all_pred)

# Plot
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Test Data", "Train Data"],
            yticklabels=["Test Data", "Train Data"])
plt.title("Generated Data of Base Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
