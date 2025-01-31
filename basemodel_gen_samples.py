import os
import time
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig

from data import load_gen_data
import warnings

warnings.filterwarnings("ignore")

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
generator = AutoModelForCausalLMWithValueHead.from_pretrained("codeparrot/codeparrot-small")
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 200,
    "bos_token_id": tokenizer.bos_token_id,
    'num_return_sequences': 1
}
ppo_cfg = PPOConfig(**{"batch_size": 32,
                       "mini_batch_size": 32,
                       "optimize_device_cache": False})
generator = generator.to(device)
ppo_trainer = PPOTrainer(config=ppo_cfg, model=generator, tokenizer=tokenizer)

dataset_test = load_gen_data(os.path.join("test.json"), tokenizer, train=False)


def ladezeile(aktuell, gesamt, breite=50):
    fortschritt = (aktuell / gesamt)
    ladebalken = "[" + "=" * int(fortschritt * breite) + " " * (breite - int(fortschritt * breite)) + "]"
    prozent = fortschritt * 100
    print(f"\r{ladebalken} {prozent:.1f}%", end="", flush=True)


def sample_generieren(idx):
    prompts_tensor = dataset_test.__getitem__(idx)['prompts'].to(torch.int64).to(device)
    prompts_txt = tokenizer.decode(prompts_tensor, skip_special_tokens=True)
    generation_tensor = ppo_trainer.generate(query_tensor=prompts_tensor, return_prompt=False, **generation_kwargs)
    generated_txts = tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
    header_prompt = "\n# " + prompts_txt + "\n\n"
    if not os.path.exists("./temp_files_basemodel"):
        os.makedirs("./temp_files_basemodel")
    with open(f"./temp_files_basemodel/sample_{idx}", "w", encoding='utf-8') as file:
        file.write(header_prompt)
        file.write(generated_txts[0])


for idx in range(dataset_test.__len__()):
    sample_generieren(idx)
    ladezeile(idx, dataset_test.__len__())

print("\nFertig! Alle Samples wurden generiert.")
