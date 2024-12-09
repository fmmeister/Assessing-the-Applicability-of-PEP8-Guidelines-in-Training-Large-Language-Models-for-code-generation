import os.path
from tqdm import tqdm
from argparse import Namespace
import time
import torch
from torch.nn.modules.loss import _Loss
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling, DataCollatorWithPadding)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from discriminator import CNNDiscriminator
from data import prepare_data, load_gen_data, sample_from_dataset
from eval import get_rewards, pep08, compilable
# ToDo: Logging of parameters and metrics
# ToDo: take out again:
import warnings
warnings.filterwarnings("ignore")


class GANTrainer:
    # ToDo: seq2seq training

    def __init__(self, cfg: Namespace) -> None:
        self.cfg = cfg
        self.device = "cuda" if self.cfg.gpu else "cpu"
        print(" >> init tokenizer")
        # if self.cfg.train_tokenizer:
        #     raise NotImplementedError("No implementation for training a tokenizer from scratch at the moment.")
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_src,
        #                                                    padding_side="left")  # assuming cfg.tokenizer_src is either a file path or a huggingface model specifier
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        print(" >> init generator")
        if self.cfg.load_generator:
            if self.cfg.pretrain:
                self.generator = AutoModelForCausalLM.from_pretrained(self.cfg.gen_dir)
            else:
                self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir)
        else:
            if self.cfg.pretrain:
                self.generator = AutoModelForCausalLM.from_pretrained(self.cfg.base_model)
            else:
                self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.base_model)
            self.generator.save_pretrained(self.cfg.gen_dir)
        self.pretrain_args = TrainingArguments(
            output_dir=self.cfg.gen_dir,
            eval_strategy="epoch",
            num_train_epochs=self.cfg.num_pretrain_epochs,
            learning_rate=self.cfg.pretrain_lr,
            weight_decay=self.cfg.weight_decay,
            push_to_hub=False,
            save_strategy="epoch",
            save_total_limit=5,
            load_best_model_at_end=True
        )
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
            "bos_token_id": self.tokenizer.bos_token_id
        }
        self.adv_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.cfg.disc_lr)
        print(" >> read data")
        self.dataset_train = load_gen_data(os.path.join(self.cfg.data_dir, "train.json"),
                                           self.tokenizer, block_size=self.cfg.max_new_tokens+1)
        self.dataset_test = load_gen_data(os.path.join(self.cfg.data_dir, "test.json"),
                                          self.tokenizer, block_size=self.cfg.max_new_tokens+1)

    def gen_pretrain(self) -> None:
        self.tokenizer.padding_side = "right"
        trainer = Trainer(
            model=self.generator,
            args=self.pretrain_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,
            data_collator=self.lm_data_collator,
        )
        print(" === Generator Pretraining ===")
        trainer.train()
        self._write_samples(self.cfg.num_pretrain_epochs - 1, pretrain=True)
        print(" >> save pretrained generator to", self.cfg.gen_dir)
        trainer.save_model(self.cfg.gen_dir)

    def adversarial_train(self) -> None:
        print(" >> prepare Generator")
        self.generator = AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.gen_dir)
        self.generator = self.generator.to(self.device)
        self.tokenizer.padding_side = "left"
        print(" =========== Discriminator Pretraining ===========")
        disc_stats = self._disc_adv_train()
        print(" > ", disc_stats)
        print(" >> prepare PPOTrainer")
        self.ppo_trainer = PPOTrainer(self.ppo_cfg, self.generator, tokenizer=self.tokenizer)
        print(" =========== Start Adversarial Training ===========")
        for epoch in range(self.cfg.num_adv_epochs):
            print(f" --- Epoch {epoch}: Generator ---")
            gen_stats = self._gen_adv_train()
            # print(" > ", gen_stats) ToDo
            print(f" --- Epoch {epoch}: Discriminator---")
            disc_stats = self._disc_adv_train()
            print(" > ", disc_stats)
            print(f" -----------------------------------")
            if epoch % 10 == 0:
                self._write_samples(epoch)

    def _gen_adv_train(self) -> dict:
        start_txt = [self.tokenizer.bos_token + "Response:\n"] * self.cfg.batch_size
        starts = self.tokenizer(start_txt, padding=False)
        starts = [torch.tensor(item).to(self.device) for item in starts['input_ids']]
        generations = self.ppo_trainer.generate(starts, return_prompt=False, **self.generation_kwargs)
        generated_txts = self.tokenizer.batch_decode(generations)

        rewards = get_rewards(generations, generated_txts, self.discriminator, self.tokenizer)
        rewards = [reward for reward in rewards]
        train_stats = self.ppo_trainer.step(starts, generations, rewards)
        # ToDo: more informative file/dir names
        self.generator.save_pretrained(os.path.join(self.cfg.gen_dir, "generator"))
        return train_stats

    def _disc_adv_train(self) -> dict:
        lm_generator = AutoModelForCausalLM.from_pretrained(self.cfg.gen_dir).to(self.device)
        # lm_generator.to(self.device)
        data = prepare_data(lm_generator, self.dataset_train,
                            self.cfg.num_samples, self.generation_kwargs)
        total_loss = 0
        total_acc = 0
        total_num = 0
        for _ in tqdm(range(self.cfg.disc_steps)):
            for i in range(0, len(data), self.cfg.batch_size):
                batch = data[i:min(i + self.cfg.batch_size, len(data) - 1)]
                samples = batch["input_ids"]
                classes = torch.tensor(batch["labels"])
                batch = self.lm_data_collator(samples, return_tensors="pt")['input_ids']
                batch = batch.to(self.device)
                classes = classes.to(self.device)
                classification = self.discriminator.forward(batch)
                loss = self.adv_loss(classification, classes)
                self._optimize_discriminator(loss)
                total_loss += loss.item()
                total_acc += torch.sum((classification.argmax(dim=-1) == classes)).item()
                total_num += batch.size(0)
        total_loss /= len(data)
        total_acc /= total_num
        torch.save(self.discriminator.state_dict(), os.path.join(self.cfg.disc_dir, "discriminator.pt"))
        return {"loss": total_loss, "acc": total_acc}

    def _optimize_discriminator(self, loss: _Loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.clip_norm)
        self.optimizer.step()

    def _write_samples(self, epoch, pretrain=False):
        save_path = os.path.join(os.path.join(self.cfg.gen_dir, "samples/"))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_name = f"pretrain_{self.cfg.now}.txt" if pretrain else f"adv_train_{self.cfg.now}.txt"
        save_path = os.path.join(save_path, file_name)
        print(" >> write samples to ", save_path, "...")
        with torch.no_grad():
            generations = self.generator.generate(
                num_return_sequences=self.cfg.n_samples,
                do_sample=True,
                top_k=50, top_p=0.5,
                temperature=0.3,
                max_new_tokens=self.cfg.max_new_tokens,
                bos_token_id=self.tokenizer.bos_token_id,
            )
            generated_txts = self.tokenizer.batch_decode(generations, skip_special_tokens=True)

        if not os.path.exists(save_path):
            with open(save_path, "w") as f:
                f.write(time.asctime())

        with open(save_path, "a") as f:
            f.write(f"\n >>>>>>>>>>>>>>>>>>>>>>> EPOCH {epoch} <<<<<<<<<<<<<<<<<<<<<<<\n")
            for line in generated_txts:
                f.write(line)
                f.write("\n============================================================\n")

    def eval(self):
        # def read_file_as_string(file_path: str) -> str:
        #     with open(file_path, 'r', encoding='utf-8') as file:
        #         content = file.read()
        #     return content
        #
        # # Beispielverwendung
        # file_path = './save/huggan/gen/samples/pretrain_2024-12-01 12_14_59.txt'
        # file_content = read_file_as_string(file_path)
        # # print(file_content)
        # print("Number of errors: ",pep08(file_content))
        # print("Is compilable: ",compilable(file_content))
        self._write_samples(0, pretrain=True)