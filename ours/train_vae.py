# train_controller.py
import argparse
import os
import sys

import pandas

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "VTFS", "code"))
import pickle
import random
import sys
import json
from typing import List


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

from controller import GAFS
from feature_env import FeatureEvaluator, base_path
from ours.utils_meter import (
    AvgrageMeter,
    pairwise_accuracy,
    hamming_distance,
    count_parameters_in_MB,
    FSDataset,
)
from record import SelectionRecord
from utils.logger import info, error
from ours.sparse_mask import FeatureMaskGenerator
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history_path, save_dir):
    """Plot loss curve"""
    try:
        with open(loss_history_path, "r") as f:
            loss_history = json.load(f)

        epochs = [item["epoch"] for item in loss_history]
        train_loss = [item["train_loss"] for item in loss_history]
        train_mse = [item["train_mse"] for item in loss_history]
        train_ce = [item["train_ce"] for item in loss_history]
        train_kl = [item["train_kl"] for item in loss_history]

        plt.figure(figsize=(12, 8))

        # Plot total loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_loss, "b-", label="Total Loss", linewidth=2)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot MSE loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_mse, "r-", label="MSE Loss", linewidth=2)
        plt.title("MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot Cross Entropy loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_ce, "g-", label="Cross Entropy Loss", linewidth=2)
        plt.title("Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot KL loss
        plt.subplot(2, 2, 4)
        plt.plot(epochs, train_kl, "m-", label="KL Loss", linewidth=2)
        plt.title("KL Divergence Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plot_path = f"{save_dir}/loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        info(f"Loss curve saved to {plot_path}")

    except Exception as e:
        error(f"Error plotting loss curve: {e}")
        plt.close("all")  # Ensure all figures are closed to prevent memory leaks


parser = argparse.ArgumentParser()
# Basic model parameters.

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--new_gen", type=int, default=200)
parser.add_argument(
    "--method_name",
    type=str,
    choices=["rnn", "transformer", "transformerVae"],
    default="transformerVae",
)
parser.add_argument(
    "--task_name",
    type=str,
    default="openml_586",
    choices=[
        "spectf",
        "svmguide3",
        "german_credit",
        "spam_base",
        "ionosphere",
        "megawatt1",
        "uci_credit_card",
        "openml_618",
        "openml_589",
        "openml_616",
        "openml_607",
        "openml_620",
        "openml_637",
        "cifar-10",
        "cifar-100",
        "semeion",
        "openml_586",
        "uci_credit_card",
        "higgs",
        "ap_omentum_ovary",
        "activity",
        "mice_protein",
        "coil-20",
        "isolet",
        "minist",
        "minist_fashion",
        "ap_omentum_ovary",
        "openml_1082",
        "openml_1085",
        "openml_1088"
    ],
)
parser.add_argument("--gpu", type=int, default=0, help="used gpu")
parser.add_argument("--top_k", type=int, default=25)
parser.add_argument(
    "--gen_num", type=int, default=0, help="Data augmentation times, but currently mask doesn't need augmentation"
)
parser.add_argument("--mlp_layers", type=int, default=2)
parser.add_argument("--mlp_hidden_size", type=int, default=200)
parser.add_argument("--l2_reg", type=float, default=0.0)
parser.add_argument("--max_step_size", type=int, default=100)
parser.add_argument("--trade_off", type=float, default=0.8)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--grad_bound", type=float, default=5.0)

parser.add_argument("--transformer_encoder_layers", type=int, default=2)
parser.add_argument("--encoder_nhead", type=int, default=8)
parser.add_argument("--encoder_embedding_size", type=int, default=64)
parser.add_argument("--transformer_encoder_dropout", type=float, default=0.1)
parser.add_argument("--transformer_encoder_activation", type=str, default="relu")
parser.add_argument("--encoder_dim_feedforward", type=int, default=128)
parser.add_argument("--batch_first", type=bool, default=True)
parser.add_argument("--d_latent_dim", type=int, default=64)

parser.add_argument("--transformer_decoder_layers", type=int, default=2)
parser.add_argument("--decoder_nhead", type=int, default=8)
parser.add_argument("--transformer_decoder_dropout", type=float, default=0.1)
parser.add_argument("--transformer_decoder_activation", type=str, default="relu")
parser.add_argument("--decoder_dim_feedforward", type=int, default=128)
parser.add_argument("--decoder_embedding_size", type=int, default=64)
parser.add_argument("--pre_train", type=str, default="True")
parser.add_argument("--skip_train", action="store_true", help="Skip training, directly inference")
parser.add_argument('--max_seq_len', type=int, default=5000, help='Max sequence length for positional encoding')

# Sparse attention mask parameters
parser.add_argument(
    "--use_sparse_mask",
    action="store_true",
    default=False,
    help="Whether to use structured sparse attention mask",
)
parser.add_argument(
    "--sparse_top_k", type=int, default=5, help="Number of neighbors to keep for each feature"
)
parser.add_argument(
    "--sparse_method",
    type=str,
    default="correlation",
    help="Correlation calculation method (currently only supports correlation)",
)

args = parser.parse_args()

baseline_name = [
    "kbest",
    "mrmr",
    "lasso",
    "rfe",
    # 'gfs',
    "lassonet",
    "sarlfs",
    "marlfs",
]


def gafs_train(train_queue, model: GAFS, optimizer):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    kl = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample["encoder_input"]
        encoder_target = sample["encoder_target"]
        decoder_input = sample["decoder_input"]
        decoder_target = sample["decoder_target"]

        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch, mu, logvar = model.forward(
            encoder_input, decoder_input
        )
        # Neither divided by batch size, if need to divide by batch size, set reduction = "mean"
        loss_1 = F.mse_loss(
            predict_value.squeeze(), encoder_target.squeeze()
        )  # mse loss evaluator
        loss_2 = F.nll_loss(
            log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)
        )  # ce loss reconstruction loss
        if args.method_name == "transformerVae":
            # Fix KL divergence calculation: add average over batch and sequence dimensions
            kl_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            ) / mu.numel()
            if args.pre_train == "True":
                loss = (
                    args.trade_off * loss_1
                    + (1 - args.trade_off) * loss_2
                    + 0.001 * kl_loss
                )
            else:
                loss = (
                    args.trade_off * loss_1
                    + (1 - args.trade_off) * loss_2
                    + 0.001 * kl_loss
                )
        elif args.method_name == "transformer":
            kl_loss = torch.tensor(1, dtype=torch.long)
            loss = args.trade_off * loss_1 + (1 - args.trade_off) * loss_2
        else:
            kl_loss = torch.tensor(1, dtype=torch.long)
            loss = args.trade_off * loss_1 + (1 - args.trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        kl.update(kl_loss.data, n)
    return objs.avg, mse.avg, nll.avg, kl.avg


def gafs_valid(queue, model: GAFS):
    pa = AvgrageMeter()
    hs = AvgrageMeter()
    mse = AvgrageMeter()
    ce = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample["encoder_input"]
            encoder_target = sample["encoder_target"]
            decoder_input = sample["decoder_input"]
            decoder_target = sample["decoder_target"]

            encoder_input = encoder_input.cuda(model.gpu)
            encoder_target = encoder_target.cuda(model.gpu)
            decoder_input = decoder_input.cuda(model.gpu)
            decoder_target = decoder_target.cuda(model.gpu)

            predict_value, logits, arch, mu, logvar = model.forward(
                encoder_input, decoder_input
            )
            n = encoder_input.size(0)
            pairwise_acc = pairwise_accuracy(
                encoder_target.data.squeeze().tolist(),
                predict_value.data.squeeze().tolist(),
            )
            hamming_dis = hamming_distance(
                decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist()
            )
            mse.update(
                F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()),
                n,
            )
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
            ce.update(
                F.nll_loss(
                    logits.contiguous().view(-1, logits.size(-1)),
                    decoder_target.view(-1),
                ),
                n,
            )
    return mse.avg, pa.avg, hs.avg, ce.avg


def choice_to_onehot(choice: List[int]):
    size = len(choice)
    onehot = torch.zeros(size + 1)
    onehot[torch.tensor(choice)] = 1
    return onehot[:-1]


def gafs_infer(queue, model, step, direction="+"):
    new_gen_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample["encoder_input"]
        encoder_input = encoder_input.cuda(model.gpu)
        model.zero_grad()
        new_gen = model.generate_new_feature(
            encoder_input, predict_lambda=step, direction=direction
        )
        new_gen_list.extend(new_gen.data.squeeze().tolist())
    return new_gen_list


def select_top_k(choice: Tensor, labels: Tensor, k: int) -> (Tensor, Tensor):
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]


def main():
    if not torch.cuda.is_available():
        info("No GPU found!")
        sys.exit(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    info(f"Args = {args}")

    with open(f"{base_path}/history/{args.task_name}/fe.pkl", "rb") as f:
        fe: FeatureEvaluator = pickle.load(f)
    model = GAFS(fe, args)
    if args.pre_train == "False":
        model.load_state_dict(
            torch.load(
                f"{base_path}/history/{args.task_name}/GAFS_pretrain_{args.method_name}.model_dict"
            )
        )
    elif args.pre_train == "Search":
        model.load_state_dict(
            torch.load(
                f"{base_path}/history/{args.task_name}/GAFS_{args.method_name}.model_dict"
            )
        )

    info(f"param size = {count_parameters_in_MB(model)}MB")
    model = model.cuda(device)

    # Initialize structured sparse attention mask
    if args.use_sparse_mask:
        info(
            f"🔧 Initializing structured sparse attention mask (top_k={args.sparse_top_k}, method={args.sparse_method})"
        )

        # Use original data from FeatureEvaluator to generate mask
        mask_generator = FeatureMaskGenerator(
            data=fe.original, top_k=args.sparse_top_k, method=args.sparse_method
        )

        # Set mask for VAE
        model.set_sparse_mask(mask_generator)
        info(f"✅ VAE Encoder has set sparse mask (Decoder remains unchanged)")

    # Setting to 0 means no data augmentation
    choice, labels = fe.get_record(args.gen_num, eos=fe.ds_size)
    valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)

    info("Training Encoder-Predictor-Decoder")
    min_val = min(labels)
    max_val = max(labels)
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]

    # For 0/1 mask mode, use valid SOS/EOS values
    sos_id = 0  # SOS: do not select feature (corresponds to 0 in vocab_size=2)
    eos_id = 1  # EOS: select feature (corresponds to 1 in vocab_size=2, although may not actually be needed)
    train_dataset = FSDataset(
        choice,
        train_encoder_target,
        train=True,
        sos_id=sos_id,
        eos_id=eos_id,
        ds_size=fe.ds_size,
    )
    valid_dataset = FSDataset(
        valid_choice,
        valid_encoder_target,
        train=True,
        sos_id=sos_id,
        eos_id=eos_id,
        ds_size=fe.ds_size,
    )
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False, pin_memory=True
    )

    # Output training sample count information
    info(f"Training dataset size: {len(train_dataset)} samples")
    info(f"Validation dataset size: {len(valid_dataset)} samples")
    info(f"Training batch size: {args.batch_size}")
    info(f"Number of training batches: {len(train_queue)}")
    info(f"Number of validation batches: {len(valid_queue)}")
    nao_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_reg
    )
    save_model = model
    cur_loss = float("inf")
    best_epoch = 0

    # Initialize loss history
    loss_history = []

    # If skipping training, use current model directly
    if args.skip_train:
        info("Skipping training, using loaded model for inference...")
    else:
        for nao_epoch in range(1, args.epochs + 1):
            sys.stdout.flush()
            sys.stderr.flush()
            nao_loss, nao_mse, nao_ce, kl = gafs_train(
                train_queue, model, nao_optimizer
            )
            if nao_epoch % 10 == 0 or nao_epoch == 1:
                info(
                    "epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}, kl {:.6f}".format(
                        nao_epoch, nao_loss, nao_mse, nao_ce, kl
                    )
                )
                if nao_loss < cur_loss:
                    save_model = model
                    cur_loss = float(
                        nao_loss.item() if torch.is_tensor(nao_loss) else nao_loss
                    )
                    best_epoch = nao_epoch

            # Record loss history - ensure all values are basic data types
            nao_loss_val = float(
                nao_loss.item() if torch.is_tensor(nao_loss) else nao_loss
            )
            loss_history.append(
                {
                    "epoch": nao_epoch,
                    "train_loss": nao_loss_val,
                    "train_mse": float(
                        nao_mse.item() if torch.is_tensor(nao_mse) else nao_mse
                    ),
                    "train_ce": float(
                        nao_ce.item() if torch.is_tensor(nao_ce) else nao_ce
                    ),
                    "train_kl": float(kl.item() if torch.is_tensor(kl) else kl),
                    "is_best": nao_loss_val < cur_loss,
                }
            )

            if nao_epoch % 100 == 0 or nao_epoch == 1:
                mse, pa, hs, ce = gafs_valid(train_queue, model)
                info("Evaluation on train data")
                info(
                    "epoch {:04d} mse {:.6f} ce {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}".format(
                        nao_epoch, mse, ce, pa, hs
                    )
                )
                mse, pa, hs, ce = gafs_valid(valid_queue, model)
                info("Evaluation on valid data")
                info(
                    "epoch {:04d} mse {:.6f} ce {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}".format(
                        nao_epoch, mse, ce, pa, hs
                    )
                )
    model = save_model
    info("best model from epoch {:04d}".format(best_epoch))

    top_selection, top_performance = select_top_k(
        valid_choice, valid_labels, args.top_k
    )

    infer_dataset = FSDataset(
        top_selection,
        top_performance,
        train=False,
        sos_id=sos_id,
        eos_id=eos_id,
        ds_size=fe.ds_size,
    )
    infer_queue = DataLoader(
        infer_dataset, batch_size=len(infer_dataset), shuffle=False, pin_memory=True
    )
    if args.method_name != "transformerVae" or (
        args.method_name == "transformerVae" and args.pre_train != "True"
    ):
        new_selection = []
        new_choice = []
        predict_step_size = 0
        while len(new_selection) < args.new_gen:
            predict_step_size += 1
            # info('Generate new architectures with step size {:.2f}'.format(predict_step_size))
            new_record = gafs_infer(
                infer_queue, model, direction="+", step=predict_step_size
            )
            # info(f'Generated sequences from step {predict_step_size}:')
            for idx, choice in enumerate(new_record):
                # info(f'Sequence {idx}: {choice}')
                # Now choice is already in 0/1 mask format, use directly
                onehot_choice = torch.tensor(choice, dtype=torch.float)
                if onehot_choice.sum() <= 0:
                    error("insufficient selection")
                    continue
                record = SelectionRecord(onehot_choice.numpy(), -1)
                if record not in fe.records.r_list and record not in new_selection:
                    new_selection.append(record)
                    new_choice.append(onehot_choice)
                if len(new_selection) >= args.new_gen:
                    break
            # info(f'{len(new_selection)} new choice generated now', )
            if predict_step_size > args.max_step_size:
                break
        # info(f'build {len(new_selection)} new choice !!!')

        new_choice_pt = torch.stack(new_choice)
        if args.gen_num == 0:
            choice_path = f"{base_path}/history/{fe.task_name}/generated_choice_{args.method_name}.pt"
        else:
            choice_path = f"{base_path}/history/{fe.task_name}/generated_choice_{args.method_name}.pt"
        torch.save(new_choice_pt, choice_path)
        info(f"save generated choice to {choice_path}")

    if args.pre_train == "True":
        torch.save(
            model.state_dict(),
            f"{base_path}/history/{fe.task_name}/GAFS_pretrain_{args.method_name}.model_dict",
        )
        torch.save(
            model.state_dict(),
            f"{base_path}/history/{fe.task_name}/GAFS_{args.method_name}.model_dict",
        )
    else:
        torch.save(
            model.state_dict(),
            f"{base_path}/history/{fe.task_name}/GAFS_{args.method_name}.model_dict",
        )

    # Save loss history
    import json

    loss_history_path = (
        f"{base_path}/history/{fe.task_name}/loss_history_controller.json"
    )
    with open(loss_history_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    info(f"Loss history saved to {loss_history_path}")

    # Plot loss curve
    try:
        plot_loss_curve(loss_history_path, f"{base_path}/history/{fe.task_name}")
    except Exception as e:
        error(f"Failed to plot loss curve: {e}")

    if args.pre_train == "True":
        return
    best_selection = None
    best_optimal = -1000
    best_selection_test = None
    best_optimal_test = -1000
    # info(f'the best performance for this task is {previous_optimal}')
    for s in new_selection:
        train_data = fe.generate_data(s.operation, "train")
        result = fe.get_performance(train_data)
        test_data = fe.generate_data(s.operation, "test")
        test_result = fe.get_performance(test_data)
        # if result > previous_optimal:
        #     optimal_selection = s.operation
        #     previous_optimal = result
        #     info(f'found optimal selection! the choice is {s.operation}, the performance on train is {result}')
        if result > best_optimal:
            best_selection = s.operation
            best_optimal = result
            info(f"found best on train : {best_optimal}")
        if test_result > best_optimal_test:
            best_selection_test = s.operation
            best_optimal_test = test_result
            info(f"found best on test : {best_optimal_test}")

    opt_path = f"{base_path}/history/{fe.task_name}/best-ours.hdf"
    ori_p = fe.report_performance(best_selection, flag="test")
    info(
        f"found train generation in our method! the choice is {best_selection}, the performance is {ori_p}"
    )
    fe.generate_data(best_selection, "train").to_hdf(opt_path, key="train")
    fe.generate_data(best_selection, "test").to_hdf(opt_path, key="test")

    opt_path_test = f"{base_path}/history/{fe.task_name}/best-ours-test.hdf"
    test_p = fe.report_performance(best_selection_test, flag="test")
    info(
        f"found test generation in our method! the choice is {best_selection_test}, the performance is {test_p}"
    )
    fe.generate_data(best_selection_test, "train").to_hdf(opt_path_test, key="train")
    fe.generate_data(best_selection_test, "test").to_hdf(opt_path_test, key="test")
    ps = []
    info("given overall validation")
    report_head = "RAW\t"
    raw_test = pandas.read_hdf(
        f"{base_path}/history/{fe.task_name}.hdf", key="raw_test"
    )
    ps.append("{:.2f}".format(fe.get_performance(raw_test) * 100))
    for method in baseline_name:
        report_head += f"{method}\t"
        spe_test = pandas.read_hdf(
            f"{base_path}/history/{fe.task_name}.hdf", key=f"{method}_test"
        )
        ps.append("{:.2f}".format(fe.get_performance(spe_test) * 100))
    report_head += "Ours\tOurs_Test"
    report = ""
    print(report_head)
    for per in ps:
        report += f"{per}&\t"
    report += "{:.2f}&\t".format(ori_p * 100)
    report += "{:.2f}&\t".format(test_p * 100)
    print(report)

if __name__ == "__main__":
    main()
