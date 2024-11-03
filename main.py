import json
import os.path
from argparse import ArgumentParser, Namespace

from trainer import GANTrainer


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # ToDo: read defaults from config file
    parser.add_argument('--gpu', default=0, type=int,
                        help="whether to use gpu")
    parser.add_argument('--num_adv_epochs', default=200, type=int,
                        help="number of epochs for adversarial training")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="batch size for adversarial generator and discriminator training")
    parser.add_argument('--data_dir', default="data/aimotion/mbpp", type=str,
                        help="training data directory")
    parser.add_argument('--output_dir', default="/RL", type=str,)
    # Pretraining:
    parser.add_argument('--pretrain', default=1, type=int,
                        help="whether to pretrain the generator")
    parser.add_argument('--num_pretrain_epochs', default=80, type=int,
                        help="number of epochs for generator pretraining")
    parser.add_argument('--pretrain_lr', default=2e-5, type=float,
                        help="pretraining learning rate")
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help="pretraining weight decay")
    # Generator:
    parser.add_argument('--gen_dir', default="./save/huggan/gen/", type=str,
                        help="directory from which to load and to which to save the trained generator model")
    parser.add_argument('--load_generator', default=0, type=int,
                        help="whether to load existing generator model from gen_dir")
    parser.add_argument('--base_model', default="gpt2", type=str,
                        help="model to load from huggingface hub as generator base, "
                             "if no local pretrained model is provided")
    parser.add_argument('--max_new_tokens', default=50, type=int,
                        help="max number of tokens for generator to produce during adversarial training")
    parser.add_argument('--n_samples', default=20, type=int,
                        help="number of samples to generate after pretraining and each adversarial training epoch")
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
    parser.add_argument('--dis_init', default="uniform", type=str,
                        help="discriminator weight initialization: normal, uniform, truncated_normal")
    parser.add_argument('--dropout', default=0.25, type=float,
                        help="dropout rate for discriminator")
    parser.add_argument('--disc_lr', default=1e-2, type=float,
                        help="adversarial learning rate for discriminator")
    parser.add_argument('--num_samples', default=32, type=int,
                        help="number of generated samples to include for each epoch of discriminator training")
    parser.add_argument('--disc_steps', default=4, type=int,
                        help="discriminator training steps for each adversarial epoch")
    parser.add_argument('--clip_norm', default=5.0, type=float,
                        help="number of epochs for generator pretraining")

    args = parser.parse_args()
    args.num_filters = json.loads(f"[{args.num_filters}]")
    args.filter_sizes = json.loads(f"[{args.filter_sizes}]")

    return args


if __name__ == "__main__":

    args = parse_args()

    print("Filters:", args.num_filters, args.filter_sizes)

    if not os.path.exists(args.gen_dir):
        print(" > makedirs", args.gen_dir)
        os.makedirs(args.gen_dir, exist_ok=True)
    if not os.path.exists(args.disc_dir):
        print(" > makedirs", args.disc_dir)
        os.makedirs(args.disc_dir, exist_ok=True)

    trainer = GANTrainer(args)

    if args.pretrain:
        trainer.gen_pretrain()

    trainer.adversarial_train()
    trainer.eval()
