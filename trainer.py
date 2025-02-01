import codecs
import json
import os.path
import shutil
import gc
import tempfile
from statistics import mean

import mlflow
from torch import Tensor
from argparse import Namespace
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from discriminator import CNNDiscriminator
from data import prepare_data, load_gen_data, get_index_length_datasets
from eval import get_rewards
from time import gmtime, strftime

import warnings

warnings.filterwarnings("ignore")


class GANTrainer:

    def __init__(self, cfg: Namespace) -> None:
        self.cfg = cfg
        self.device = "cuda" if self.cfg.gpu else "cpu"
        print("device: ", self.device)
        print(" >> init tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(" >> init generator")
        if self.cfg.load_generator:
            self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir + "generator")
        else:
            self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.base_model)
            self.generator.save_pretrained(self.cfg.gen_dir)
        print(" >> init discriminator")
        self.discriminator = CNNDiscriminator(embed_dim=self.cfg.embed_dim, vocab_size=len(self.tokenizer),
                                              filter_sizes=cfg.filter_sizes, num_filters=cfg.num_filters,
                                              padding_idx=self.tokenizer.pad_token_id, gpu=self.cfg.gpu,
                                              dropout=self.cfg.dropout)
        if self.cfg.load_discriminator:
            state_dict = torch.load(self.cfg.disc_dir + "/discriminator.pt")
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
        self.dataset_test = load_gen_data(os.path.join(self.cfg.data_dir, "test.json"), self.tokenizer, train=False)

        # Logging parameters
        mlflow.log_param("num_adv_epochs", self.cfg.num_adv_epochs)
        mlflow.log_param("gen_batch_size", self.cfg.batch_size)
        mlflow.log_param("disc_weight", self.cfg.disc_weight)
        mlflow.log_param("gen_max_new_tokens", self.cfg.max_new_tokens)
        mlflow.log_param("disc_embed_dim", self.cfg.embed_dim)
        mlflow.log_param("disc_filter_sizes", self.cfg.filter_sizes)
        mlflow.log_param("disc_num_filters", self.cfg.num_filters)
        mlflow.log_param("disc_dropout", self.cfg.dropout)
        mlflow.log_param("disc_lr", self.cfg.disc_lr)
        mlflow.log_param("disc_num_samples", self.cfg.num_samples)
        mlflow.log_param("disc_clip_norm", self.cfg.clip_norm)
        if self.cfg.load_generator:
            mlflow.log_param("load_generator", True)
        else:
            mlflow.log_param("base_model", self.cfg.base_model)

    def adversarial_train(self) -> None:
        print(" >> prepare Generator")
        self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir)
        self.generator = self.generator.to(self.device)
        self.tokenizer.padding_side = "left"
        print(" =========== Discriminator Pretraining ===========")
        disc_loss = []
        disc_acc = []
        disc_loss, disc_acc = self._disc_adv_train(-1, disc_loss, disc_acc)
        print(" > ", {"loss": disc_loss[0], "acc": disc_acc[0]})
        print(" >> prepare PPOTrainer")
        self.ppo_trainer = PPOTrainer(config=self.ppo_cfg, model=self.generator, tokenizer=self.tokenizer)
        print(" =========== Start Adversarial Training ===========")
        avg_rewards = []
        for epoch in range(self.cfg.num_adv_epochs):
            print(f" --- Epoch {epoch}: Generator ---")
            avg_rewards = self._gen_adv_train(epoch, avg_rewards)
            print("Average reward: ", avg_rewards[epoch])
            print(f" --- Epoch {epoch}: Discriminator---")
            disc_loss, disc_acc = self._disc_adv_train(epoch, disc_acc, disc_loss)
            print(" > ", {"loss": disc_loss[epoch + 1], "acc": disc_acc[epoch + 1]})
            print(f" -----------------------------------")
        print("Average Rewards for the epochs: ", avg_rewards)

        if self.cfg.save_RL:
            print(" >> save RL model")
            model_save_now_path = self.cfg.gen_dir + "/" + str(strftime("%Y-%m-%d %H_%M_%S", gmtime()))
            self.generator.save_pretrained(model_save_now_path + "/generator")
            torch.save(self.discriminator.state_dict(), os.path.join(model_save_now_path, "discriminator.pt"))
            data = {"disc_loss": disc_loss, "disc_acc": disc_acc, "gen_rewards": avg_rewards}
            # data = {"gen_rewards": avg_rewards}
            with open(model_save_now_path + "output.json", "w") as file:
                json.dump(data, file, indent=4)

        if self.cfg.delete_temp_files:
            print(" >> delete temporary files")
            if os.path.exists('./temp_files'):
                shutil.rmtree('./temp_files')

    def _gen_adv_train(self, current_epoch: int, avg_rewards: list) -> list:
        data = DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        padding_length = get_index_length_datasets(train=True)
        avg_rewards_during_batches = []
        for batch_number, batch in enumerate(data):
            prompts_tensor = [prompt.to(torch.int64).to(self.device) for prompt in
                              torch.unbind(batch['prompts'], dim=0)]
            prompts_txt = self.tokenizer.batch_decode(prompts_tensor, skip_special_tokens=True)
            generation_tensor = self.ppo_trainer.generate(query_tensor=prompts_tensor, return_prompt=False,
                                                          **self.generation_kwargs)
            generated_txts = self.tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
            rewards, avg_rewards_during_batches = get_rewards(generation_text=generated_txts,
                                                              discriminator=self.discriminator,
                                                              current_epoch=current_epoch,
                                                              avg_rewards_during_batches=avg_rewards_during_batches,
                                                              prompts=prompts_txt,
                                                              tokenizer=self.tokenizer, device=self.device,
                                                              padding_length=padding_length,
                                                              disc_weight=self.cfg.disc_weight)
            rewards = [reward.detach().to(self.device) for reward in rewards]
            self.ppo_trainer.step(prompts_tensor, generation_tensor, rewards)
            for k, sample in enumerate(generated_txts):
                try:
                    header_prompt = "\n# " + prompts_txt[k] + "\n\n"
                    code = codecs.unicode_escape_decode(sample)[0]
                    if not os.path.exists("./temp_files"):
                        os.makedirs("./temp_files")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir="./temp_files",
                                                     prefix="Epoch " + str(current_epoch) + "_") as temp_file:
                        temp_file.write(header_prompt.encode('utf-8'))
                        temp_file.write(code.encode('utf-8'))
                        temp_file_path = temp_file.name
                    mlflow.log_artifact(temp_file_path)
                    break
                except UnicodeError:
                    pass
            del prompts_tensor
            del prompts_txt
            del generation_tensor
            del generated_txts
            del rewards
            gc.collect()
        self.generator.save_pretrained(os.path.join(self.cfg.gen_dir, "generator"))
        avg_rewards_during_epoch = mean(avg_rewards_during_batches)  # mean(avg_reward per batch) = avg_reward(epoch)
        avg_rewards.append(avg_rewards_during_epoch)
        mlflow.log_metric("gen_rewards", avg_rewards[current_epoch], step=current_epoch)
        return avg_rewards

    def _disc_adv_train(self, current_epoch: int, disc_loss_list: list, disc_acc_list: list) -> tuple[list, list]:
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
        mlflow.log_metric("disc_loss", total_loss, step=current_epoch)
        mlflow.log_metric("disc_acc", total_acc, step=current_epoch)
        disc_acc_list.append(total_acc)
        disc_loss_list.append(total_loss)
        return disc_loss_list, disc_acc_list

    def _optimize_discriminator(self, loss: _Loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.clip_norm)
        self.optimizer.step()

    def eval(self, current_epoch: int, avg_rewards: list):
        data = DataLoader(dataset=self.dataset_test, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        padding_length = get_index_length_datasets(train=True)
        avg_rewards_during_batches = []
        for batch_number, batch in enumerate(data):
            print("Eval batch nr. ", batch_number)
            if batch_number > 2:
                break
            prompts_tensor = [prompt.to(torch.int64).to(self.device) for prompt in
                              torch.unbind(batch['prompts'], dim=0)]
            prompts_txt = self.tokenizer.batch_decode(prompts_tensor, skip_special_tokens=True)
            generation_tensor = self.ppo_trainer.generate(query_tensor=prompts_tensor, return_prompt=False,
                                                          **self.generation_kwargs)
            generated_txts = self.tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)
            if not os.path.exists("./temp_files"):
                os.makedirs("./temp_files")
            for k, sample in enumerate(generated_txts):
                header_prompt = "\n# " + prompts_txt[k] + "\n\n"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir="./temp_files",
                                                 prefix="Eval " + str(current_epoch) + "_") as temp_file:
                    temp_file.write(header_prompt.encode('utf-8'))
                    temp_file.write(sample.encode('utf-8'))
            rewards, avg_rewards_during_batches = get_rewards(generation_text=generated_txts,
                                                              discriminator=self.discriminator,
                                                              current_epoch=current_epoch,
                                                              avg_rewards_during_batches=avg_rewards_during_batches,
                                                              prompts=prompts_txt,
                                                              tokenizer=self.tokenizer, device=self.device,
                                                              padding_length=padding_length,
                                                              disc_weight=self.cfg.disc_weight)
            print(">> computed eval rewards")
        avg_rewards_during_epoch = mean(avg_rewards_during_batches)  # mean(avg_reward per batch) = avg_reward(epoch)
        avg_rewards.append(avg_rewards_during_epoch)
        print("Average Rewards: ", avg_rewards)
