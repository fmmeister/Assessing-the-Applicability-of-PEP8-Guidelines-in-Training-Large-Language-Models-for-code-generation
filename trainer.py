import ast
import codecs
import json
import os.path
import shutil
import gc
import tempfile
from statistics import mean

import pandas as pd
from torch import Tensor
from argparse import Namespace
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from discriminator import CNNDiscriminator
from data import prepare_data, load_gen_data, get_index_length_datasets
from reward_function import get_rewards
from time import gmtime, strftime
from evaluation import loading_bar, mbpp_sample_generieren, calculate_pass_at_1_rate, eval_pep8

import warnings

warnings.filterwarnings("ignore")


class GANTrainer:

    def __init__(self, cfg: Namespace) -> None:
        self.cfg = cfg
        self.device = "cuda" if self.cfg.gpu else "cpu"
        print("device: ", self.device)
        print(" >> init tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(" >> init generator")
        if self.cfg.load_generator:
            self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.load_generator_path)
        else:
            self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.base_model)
            self.generator.save_pretrained(self.cfg.gen_dir)
        if self.cfg.base_model == "codeparrot/codeparrot-small":
            self.code_parrot = True
        else:
            self.code_parrot = False

        print(" >> init discriminator")
        self.discriminator = CNNDiscriminator(embed_dim=self.cfg.embed_dim, vocab_size=len(self.tokenizer),
                                              filter_sizes=cfg.filter_sizes, num_filters=cfg.num_filters,
                                              padding_idx=self.tokenizer.pad_token_id, gpu=self.cfg.gpu,
                                              dropout=self.cfg.dropout)
        if self.cfg.load_discriminator:
            state_dict = torch.load(self.cfg.load_discriminator_file)
            self.discriminator.load_state_dict(state_dict)
        self.discriminator = self.discriminator.to(self.device)
        self.ppo_cfg = PPOConfig(**{"batch_size": self.cfg.batch_size,
                                    "mini_batch_size": self.cfg.batch_size,
                                    "optimize_device_cache": False})
        self.ppo_trainer = PPOTrainer(config=self.ppo_cfg, model=self.generator, tokenizer=self.tokenizer)
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.cfg.max_new_tokens,
            "bos_token_id": 0,
            'num_return_sequences': 1
        }
        self.adv_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.cfg.disc_lr)
        print(" >> read data")
        self.dataset_train = load_gen_data(os.path.join(self.cfg.data_dir, "mbpp_train.json"), self.tokenizer,
                                           train=True, code_parrot=self.code_parrot, mbpp=True)
        self.dataset_test = load_gen_data(os.path.join(self.cfg.data_dir, "mbpp_test.json"), self.tokenizer,
                                          train=False, code_parrot=self.code_parrot, mbpp=True)

    def adversarial_train(self) -> None:
        """
        Trainings process of the generator and the discriminator.
        """
        print(" >> prepare Generator")
        self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir)
        self.generator = self.generator.to(self.device)
        self.tokenizer.padding_side = "left"
        print(" =========== Discriminator Pretraining ===========")
        disc_loss = []
        disc_acc = []
        disc_loss, disc_acc = self._disc_adv_train(-1, disc_loss, disc_acc)
        print(" > ", {"loss": disc_loss[0], "acc": disc_acc[0]})
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
        """
        Trainings process of the generator.
        """
        data = DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        padding_length = get_index_length_datasets(train=True, code_parrot=self.code_parrot, mbpp=True)
        avg_rewards_during_batches = []
        for batch_number, batch in enumerate(data):
            prompts_tensor = [prompt.to(torch.int64).to(self.device) for prompt in
                              torch.unbind(batch['prompts'], dim=0)]
            prompts_txt = self.tokenizer.batch_decode(prompts_tensor, skip_special_tokens=True)

            generation_tensor = self.ppo_trainer.generate(query_tensor=prompts_tensor, return_prompt=False,
                                                          **self.generation_kwargs)
            generated_txts = self.tokenizer.batch_decode(generation_tensor, skip_special_tokens=True)

            test_list_tensor = [prompt.to(torch.int64).to(self.device) for prompt in
                                torch.unbind(batch['test_list'], dim=0)]
            test_list_txt = self.tokenizer.batch_decode(test_list_tensor, skip_special_tokens=True)

            temp_file_path = []
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
                        temp_file_path.append(temp_file.name)
                    break
                except UnicodeError:
                    pass
            rewards, avg_rewards_during_batches = get_rewards(generation_tensor=generation_tensor,
                                                              generation_text=generated_txts,
                                                              discriminator=self.discriminator,
                                                              avg_rewards_during_batches=avg_rewards_during_batches,
                                                              tokenizer=self.tokenizer, device=self.device,
                                                              padding_length=padding_length,
                                                              disc_weight=self.cfg.disc_weight,
                                                              temp_file_path=temp_file_path,
                                                              test_list=test_list_txt)
            rewards = [reward.detach().to(self.device) for reward in rewards]
            self.ppo_trainer.step(prompts_tensor, generation_tensor, rewards)
            del prompts_tensor
            del prompts_txt
            del generation_tensor
            del generated_txts
            del rewards
            gc.collect()
        self.generator.save_pretrained(os.path.join(self.cfg.gen_dir, "generator"))
        avg_rewards_during_epoch = mean(avg_rewards_during_batches)  # mean(avg_reward per batch) = avg_reward(epoch)
        avg_rewards.append(avg_rewards_during_epoch)
        return avg_rewards

    def _disc_adv_train(self, current_epoch: int, disc_loss_list: list, disc_acc_list: list) -> tuple[list, list]:
        """
        Trainings process of the discriminator.
        """
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
                                    self.cfg.num_samples, self.generation_kwargs,
                                    self.tokenizer, train=True, code_parrot=self.code_parrot)
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
        disc_acc_list.append(total_acc)
        disc_loss_list.append(total_loss)
        return disc_loss_list, disc_acc_list

    def _optimize_discriminator(self, loss: _Loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.clip_norm)
        self.optimizer.step()

    def eval(self) -> None:
        print(">> begin evaluation")
        print("> start generating mbpp samples")
        mbpp_testcases = []
        for idx in range(self.dataset_test.__len__()):
            mbpp_sample_generieren(idx=idx, device=self.device, dataset_test=self.dataset_test,
                                   generator=self.ppo_trainer, tokenizer=self.tokenizer,
                                   generation_kwargs=self.generation_kwargs, destination_dir_path="mbpp_samples")
            loading_bar(idx, self.dataset_test.__len__())
            mbpp_testcases.append(self.dataset_test.__getitem__(idx)['test_list'].to(torch.int64))
        mbpp_testcases = self.tokenizer.batch_decode(mbpp_testcases, skip_special_tokens=True)
        mbpp_testcases = [ast.literal_eval(item) for item in mbpp_testcases]
        print("\nAll mbpp test sample generated.")

        humaneval_data = load_gen_data(os.path.join(self.cfg.data_dir, "humaneval.json"), self.tokenizer, train=False,
                                       code_parrot=self.code_parrot, mbpp=False)
        print("> start generating humaneval samples")
        human_eval_testcases = []
        for idx in range(humaneval_data.__len__()):
            mbpp_sample_generieren(idx=idx, device=self.device, dataset_test=humaneval_data,
                                   generator=self.ppo_trainer, tokenizer=self.tokenizer,
                                   generation_kwargs=self.generation_kwargs, destination_dir_path="human_eval_samples")
            loading_bar(idx, humaneval_data.__len__())
            human_eval_testcases.append(humaneval_data.__getitem__(idx)['test_list'].to(torch.int64))
        human_eval_testcases = self.tokenizer.batch_decode(human_eval_testcases, skip_special_tokens=True)
        human_eval_testcases = [ast.literal_eval(item) for item in human_eval_testcases]
        print("\nAll human_eval test sample generated.")

        pep8_mbpp, most_common_error_mbpp, compliance_rate_mbpp = eval_pep8("./mbpp_samples")
        pep8_human_eval, most_common_error_humaneval, compliance_rate_humaneval = eval_pep8("./human_eval_samples")
        pass_at_1_mbpp = calculate_pass_at_1_rate(mbpp_testcases, "./mbpp_samples", False)
        pass_at_1_human_eval = calculate_pass_at_1_rate(human_eval_testcases, "./human_eval_samples", False)
        results = {
            'test_dataset': ["mbpp", "human_eval"],
            'pep8_average_error': [pep8_mbpp, pep8_human_eval],
            'most_common_error': [most_common_error_mbpp, most_common_error_humaneval],
            'compliance_rate': [compliance_rate_mbpp, compliance_rate_humaneval],
            'pass_at_1': [pass_at_1_mbpp, pass_at_1_human_eval]
        }
        df = pd.DataFrame(results)
        df.to_csv('results.csv', index=False)
