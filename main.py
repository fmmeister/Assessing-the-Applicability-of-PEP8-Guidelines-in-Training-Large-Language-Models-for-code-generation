import os.path
import time
import mlflow
import yaml
from argparse import Namespace
from trainer import GANTrainer


def load_config(config_path: str = "config.yaml") -> Namespace:
    """
    Load configuration from a YAML file and return it as a Namespace object.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = Namespace(**config_dict)

    return config


if __name__ == "__main__":

    config = load_config()

    if not os.path.exists(config.gen_dir):
        print(" > makedirs", config.gen_dir)
        os.makedirs(config.gen_dir, exist_ok=True)
    if not os.path.exists(config.disc_dir):
        print(" > makedirs", config.disc_dir)
        os.makedirs(config.disc_dir, exist_ok=True)

    mlflow.set_experiment("RL")
    with mlflow.start_run(run_name="pretrained_disc"):
        trainer = GANTrainer(config)

        if config.adversarial:
            start_time = time.time()
            trainer.adversarial_train()
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 3600
            print(f"Computing time adversarial training: {elapsed_time:.2f} hours")

        if config.eval:
            trainer.eval(9999, [])
