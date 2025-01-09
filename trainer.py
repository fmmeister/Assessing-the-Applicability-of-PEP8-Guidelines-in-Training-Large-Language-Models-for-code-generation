import os.path
import shutil
import gc
from statistics import mean

from torch import Tensor
from argparse import Namespace
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from discriminator import CNNDiscriminator
from data import prepare_data, load_gen_data, get_index_length_datasets
from eval import get_rewards
from time import gmtime, strftime

# ToDo: Logging of parameters and metrics
# ToDo: take out again:
import warnings

warnings.filterwarnings("ignore")


class GANTrainer:
    # ToDo: seq2seq training

    def __init__(self, cfg: Namespace) -> None:
        self.cfg = cfg
        self.device = "cuda" if self.cfg.gpu else "cpu"
        print("device: ", self.device)
        print(" >> init tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(" >> init generator")
        if self.cfg.load_generator:
            if self.cfg.pretrain:
                self.generator = AutoModelForCausalLM.from_pretrained(self.cfg.gen_dir)
            else:
                self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir + "./2025-01-05 "
                                                                                                      "09_38_00"
                                                                                                      "/generator")
        else:
            if self.cfg.pretrain:
                self.generator = AutoModelForCausalLM.from_pretrained(self.cfg.base_model)
            else:
                self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.base_model)
            self.generator.save_pretrained(self.cfg.gen_dir)
        # self.pretrain_args = TrainingArguments(
        #     output_dir=self.cfg.gen_dir,
        #     eval_strategy="epoch",
        #     num_train_epochs=self.cfg.num_pretrain_epochs,
        #     learning_rate=self.cfg.pretrain_lr,
        #     weight_decay=self.cfg.weight_decay,
        #     push_to_hub=False,
        #     save_strategy="epoch",
        #     save_total_limit=5,
        #     load_best_model_at_end=True
        # )
        print(" >> init discriminator")
        self.discriminator = CNNDiscriminator(embed_dim=self.cfg.embed_dim, vocab_size=len(self.tokenizer),
                                              filter_sizes=cfg.filter_sizes, num_filters=cfg.num_filters,
                                              padding_idx=self.tokenizer.pad_token_id, gpu=self.cfg.gpu,
                                              dropout=self.cfg.dropout)
        if self.cfg.load_discriminator:
            state_dict = torch.load(self.cfg.disc_dir)
            self.discriminator.load_state_dict(state_dict)
        self.discriminator = self.discriminator.to(self.device)
        self.ppo_cfg = PPOConfig(**{"batch_size": self.cfg.batch_size,
                                    "mini_batch_size": self.cfg.batch_size,
                                    "optimize_device_cache": False})
        self.ppo_trainer = None
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.cfg.max_new_tokens,
            "bos_token_id": self.tokenizer.bos_token_id,
            'num_return_sequences': 1
        }
        self.adv_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.cfg.disc_lr)
        print(" >> read data")
        self.dataset_train = load_gen_data(os.path.join(self.cfg.data_dir, "train.json"), self.tokenizer, train=True)
        # self.dataset_test = load_gen_data(os.path.join(self.cfg.data_dir, "test.json"), self.tokenizer, self.device, train=False)

    # def gen_pretrain(self) -> None:
    #     self.tokenizer.padding_side = "right"
    #     trainer = Trainer(
    #         model=self.generator,
    #         args=self.pretrain_args,
    #         train_dataset=self.dataset_train,
    #         eval_dataset=self.dataset_test,
    #         data_collator=self.lm_data_collator,
    #     )
    #     print("=== Generator Pretraining ===")
    #     trainer.train()
    #     self._write_samples(self.cfg.num_pretrain_epochs - 1, pretrain=True)
    #     print(" >> save pretrained generator to", self.cfg.gen_dir)
    #     trainer.save_model(self.cfg.gen_dir)

    def adversarial_train(self) -> None:
        print(" >> prepare Generator")
        self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir)
        self.generator = self.generator.to(self.device)
        self.tokenizer.padding_side = "left"
        print(" =========== Discriminator Pretraining ===========")
        disc_stats = self._disc_adv_train()
        print(" > ", disc_stats)
        print(" >> prepare PPOTrainer")
        self.ppo_trainer = PPOTrainer(config=self.ppo_cfg, model=self.generator, tokenizer=self.tokenizer)
        print(" =========== Start Adversarial Training ===========")
        avg_rewards = []
        for epoch in range(self.cfg.num_adv_epochs):
            print(f" --- Epoch {epoch}: Generator ---")
            avg_rewards = self._gen_adv_train(epoch, avg_rewards)
            print("Average reward: ", avg_rewards[epoch])
            print(f" --- Epoch {epoch}: Discriminator---")
            disc_stats = self._disc_adv_train()
            print(" > ", disc_stats)
            print(f" -----------------------------------")
        print("Average Rewards for the epochs: ", avg_rewards)

        if self.cfg.save_RL:
            print(" >> save RL model")
            model_save_now_path = self.cfg.gen_dir + "/" + str(strftime("%Y-%m-%d %H_%M_%S", gmtime()))
            self.generator.save_pretrained(model_save_now_path + "/generator")
            rewards_file_path = model_save_now_path + "/rewards.txt"
            with open(rewards_file_path, 'w') as f:
                for reward in avg_rewards:
                    f.write(f"{reward}\n")

        if self.cfg.delete_temp_files:
            print(" >> delete temporary files")
            if os.path.exists('./temp_files'):
                shutil.rmtree('./temp_files')

    def _gen_adv_train(self, current_epoch: int, avg_rewards: list) -> list:
        data = DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        padding_length = get_index_length_datasets(train=True)
        for batch_number, batch in enumerate(data):
            prompts_tensor = [prompt.to(torch.int64).to(self.device) for prompt in
                              torch.unbind(batch['prompts'], dim=0)]
            prompts_txt = self.tokenizer.batch_decode(prompts_tensor, skip_special_tokens=True)
            generation_tensor = self.ppo_trainer.generate(query_tensor=prompts_tensor, return_prompt=False,
                                                          **self.generation_kwargs)
            generated_txts = self.tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
            rewards, avg_rewards = get_rewards(generation_text=generated_txts, discriminator=self.discriminator,
                                               current_epoch=current_epoch, avg_rewards=avg_rewards,
                                               prompts=prompts_txt,
                                               tokenizer=self.tokenizer, device=self.device,
                                               padding_length=padding_length)
            rewards = [reward.detach().to(self.device) for reward in rewards]
            self.ppo_trainer.step(prompts_tensor, generation_tensor, rewards)
            del prompts_tensor
            del prompts_txt
            del generation_tensor
            del generated_txts
            del rewards
            gc.collect()
        self.generator.save_pretrained(os.path.join(self.cfg.gen_dir, "generator"))
        avg_rewards = [mean(avg_rewards)]  # mean(avg_reward per batch) = avg_reward(epoch)
        return avg_rewards

    def _disc_adv_train(self) -> dict:
        directory_path = "./save/huggan/gen/generator"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            # Copy the base-model to the generator dir for disc pretraining
            source_path_config = "./save/huggan/gen/config.json"
            source_path_generation_config = "./save/huggan/gen/generation_config.json"
            source_path_model = "./save/huggan/gen/model.safetensors"
            shutil.copy(source_path_config, directory_path)
            shutil.copy(source_path_generation_config, directory_path)
            shutil.copy(source_path_model, directory_path)
        else:
            pass

        lm_generator = AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=self.cfg.gen_dir + "generator").to(self.device)

        dataset_disc = prepare_data(lm_generator, self.dataset_train,
                                    self.cfg.num_samples, self.generation_kwargs, self.tokenizer, train=True)
        data = DataLoader(dataset=dataset_disc, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        total_loss = 0
        total_acc = 0
        total_num = 0
        for batch in data:
            samples = Tensor.int(batch["code"]).to(self.device)
            classes = batch["ground_truth"]

            classes = list(torch.unbind(Tensor.int(classes), dim=0))
            classes_decoded_str = self.tokenizer.batch_decode(classes, skip_special_tokens=True)
            classes_decoded_int = torch.tensor([int(item) for item in classes_decoded_str], device=self.device,
                                               dtype=torch.int64)

            classification = self.discriminator.forward(samples)

            loss = self.adv_loss(classification, classes_decoded_int)
            self._optimize_discriminator(loss)

            total_loss += loss.item()
            total_acc += (classification.argmax(dim=-1) == classes_decoded_int).sum().item()
            total_num += len(classes_decoded_int)
            del samples
            del classes_decoded_int
            gc.collect()
        total_loss /= len(data)
        total_acc /= total_num
        torch.save(self.discriminator.state_dict(), os.path.join(self.cfg.disc_dir, "discriminator.pt"))
        return {"loss": total_loss, "acc": total_acc}

    def _optimize_discriminator(self, loss: _Loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.clip_norm)
        self.optimizer.step()

    def eval(self, current_epoch: int, avg_rewards: list) -> list:
        pass
