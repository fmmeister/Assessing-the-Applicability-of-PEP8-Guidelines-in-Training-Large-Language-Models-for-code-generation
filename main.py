import json
import os.path
import time
import mlflow
from argparse import ArgumentParser, Namespace

from trainer import GANTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # ToDo: read defaults from config file
    parser.add_argument('--gpu', default=1, type=int,
                        help="whether to use gpu")
    parser.add_argument('--num_adv_epochs', default=100, type=int,
                        help="number of epochs for adversarial training")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="batch size for adversarial generator and discriminator training")
    parser.add_argument('--data_dir', default="", type=str,
                        help="training data directory")
    parser.add_argument('--save_RL', default=1, type=int,
                        help="save the model after training")
    parser.add_argument('--delete_temp_files', default=0, type=int,
                        help="delete temporary files after training")
    parser.add_argument('--eval', default=0, type=int,
                        help="evaluate the model")
    parser.add_argument('--adversarial', default=1, type=int,
                        help="whether to perform adversarial training")
    parser.add_argument('--disc_weight', default=1, type=int,
                        help="weight between disc and obj")
    # Generator:
    parser.add_argument('--gen_dir', default="./save/huggan/gen/", type=str,
                        help="directory from which to load and to which to save the trained generator model")
    parser.add_argument('--load_generator', default=0, type=int,
                        help="whether to load existing generator model from gen_dir")
    parser.add_argument('--base_model', default="codeparrot/codeparrot-small", type=str,
                        help="model to load from huggingface hub as generator base, "
                             "if no local pretrained model is provided")
    parser.add_argument('--max_new_tokens', default=200, type=int,
                        help="max number of tokens for generator to produce during adversarial training")
    # Discriminator:
    parser.add_argument('--disc_dir', default="./save/huggan/dis", type=str,
                        help="directory from which to load and to which to save the trained discriminator state_dict")
    parser.add_argument('--load_discriminator', default=0, type=int,
                        help="whether to load existing discriminator model from disc_dir")
    parser.add_argument('--embed_dim', default=64, type=int,
                        help="embedding dimension size for discriminator")
    parser.add_argument('--filter_sizes', default="1,2,3,4,5,6,7,8,9,10,15,20", type=str,
                        help="listing of filter sizes for discriminator convolutional layers")
    parser.add_argument('--num_filters', default="100,200,200,200,200,100,100,100,100,100,160,160", type=str,
                        help="listing of filter sizes for discriminator convolutional layers")
    parser.add_argument('--dropout', default=0.25, type=float,
                        help="dropout rate for discriminator")
    parser.add_argument('--disc_lr', default=1e-2, type=float,
                        help="adversarial learning rate for discriminator")
    parser.add_argument('--num_samples', default=32, type=int,
                        help="number of generated samples to include for each epoch of discriminator training")
    parser.add_argument('--clip_norm', default=5.0, type=float,
                        help="used in discriminator optimization")

    args = parser.parse_args()
    args.num_filters = json.loads(f"[{args.num_filters}]")
    args.filter_sizes = json.loads(f"[{args.filter_sizes}]")

    return args


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.gen_dir):
        print(" > makedirs", args.gen_dir)
        os.makedirs(args.gen_dir, exist_ok=True)
    if not os.path.exists(args.disc_dir):
        print(" > makedirs", args.disc_dir)
        os.makedirs(args.disc_dir, exist_ok=True)

    mlflow.set_experiment("RL")
    with mlflow.start_run(run_name="disc_weight=1"):
        trainer = GANTrainer(args)

        if args.adversarial:
            start_time = time.time()
            trainer.adversarial_train()
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 3600
            print(f"Computing time adversarial training: {elapsed_time:.2f} hours")

        if args.eval:
            trainer.eval(9999, [])
