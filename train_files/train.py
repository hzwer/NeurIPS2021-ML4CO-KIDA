import os
import sys
import glob
import argparse
import pathlib
import numpy as np
import configparser
import random

cf = configparser.ConfigParser()
cf.read("./configs/train.ini")
STORE_DIR = cf.get("train", "STORE_DIR")
# hyper parameters
BATCH_SIZE = cf.getint("train", "BATCH_SIZE")
PRETRAIN_BATCH_SIZE = cf.getint("train", "PRETRAIN_BATCH_SIZE")
VALID_BATCH_SIZE = cf.getint("train", "VALID_BATCH_SIZE")
LR = cf.getfloat("train", "LR")
POLICY_TYPE = cf.getint("train", "POLICY_TYPE")
TRAIN_NUM = cf.getint("train", "TRAIN_NUM")
VALID_NUM = cf.getint("train", "VALID_NUM")


def pretrain(policy, pretrain_loader):
    """
    Pre-trains all PreNorm layers in the model.
    Parameters
    ----------
    policy : torch.nn.Module
        Model to pre-train.
    pretrain_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of pre-training samples.
    Returns
    -------
    i : int
        Number of pre-trained layers.
    """
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            ):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, top_k=[1, 3, 5, 10], optimizer=None):
    """
    Process samples. If an optimizer is given, also train on those samples.
    Parameters
    ----------
    policy : torch.nn.Module
        Model to train/evaluate.
    data_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of training samples.
    top_k : list
        Accuracy will be computed for the top k elements, for k in this list.
    optimizer : torch.optim (optional)
        Optimizer object. If not None, will be used for updating the model parameters.
    Returns
    -------
    mean_loss : float in [0, 1e+20]
        Mean cross entropy loss.
    mean_kacc : np.ndarray
        Mean top k accuracy, for k in the user-provided list top_k.
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    n_samples_processed = 0
    if optimizer is None:
        policy.eval()
    else:
        policy.train()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy_loss = F.cross_entropy(
                logits, batch.candidate_choices, reduction="mean"
            )

            # if an optimizer is provided, update parameters
            if optimizer is not None:
                optimizer.zero_grad()
                cross_entropy_loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            # calculate top k accuracy
            kacc = []
            for k in top_k:
                # check if there are at least k candidates
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                accuracy = (
                    (pred_top_k_true_scores == true_bestscore)
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
                kacc.append(accuracy)
            kacc = np.asarray(kacc)

            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    return mean_loss, mean_kacc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "problem",
        help="MILP instance type to process.",
        choices=["item_placement", "load_balancing", "anonymous"],
    )
    parser.add_argument(
        "--exp_name", help="set experiment name", type=str, required=True,
    )
    parser.add_argument(
        "-s", "--seed", help="Random generator seed.", type=int, default=0,
    )
    parser.add_argument(
        "--file_count", help="Random generator seed.", type=int, default=1,
    )
    parser.add_argument(
        "--epoch", help="Random generator seed.", type=int, default=1000,
    )
    parser.add_argument(
        "-g", "--gpu", help="CUDA GPU id (-1 for CPU).", type=int, default=0,
    )
    args = parser.parse_args()

    top_k = [1, 3, 5, 10]
    if args.problem == "item_placement":
        train_files = []
        valid_files = []
        for i in range(1, args.file_count + 1):
            train_files.extend(
                glob.glob(
                    f"{STORE_DIR}/samples/1_item_placement_dagger{i}/train/sample_*.pkl"
                )
            )
            valid_files.extend(
                glob.glob(
                    f"{STORE_DIR}/samples/1_item_placement_dagger{i}/valid/sample_*.pkl"
                )
            )
        print(len(train_files))
        running_dir = f"{STORE_DIR}/train_files/{args.exp_name}{i}"

    elif args.problem == "load_balancing":
        train_files = []
        valid_files = []
        for i in range(1, args.file_count + 1):
            train_files.extend(
                glob.glob(
                    f"{STORE_DIR}/samples/2_load_balancing_dagger{i}/train/sample_*.pkl"
                )
            )
            valid_files.extend(
                glob.glob(
                    f"{STORE_DIR}/samples/2_load_balancing_dagger{i}/valid/sample_*.pkl"
                )
            )
        print(len(train_files))
        running_dir = f"{STORE_DIR}/train_files/{args.exp_name}{i}"

    elif args.problem == "anonymous":
        train_files = []
        valid_files = []
        for i in range(1, args.file_count + 1):
            train_files.extend(
                glob.glob(
                    f"{STORE_DIR}/samples/3_anonymous_dagger{i}/train/sample_*.pkl"
                )
            )
            valid_files.extend(
                glob.glob(
                    f"{STORE_DIR}/samples/3_anonymous_dagger{i}/valid/sample_*.pkl"
                )
            )
        print(len(train_files))
        running_dir = f"{STORE_DIR}/train_files/{args.exp_name}{i}"

    else:
        raise NotImplementedError

    pretrain_files = [f for i, f in enumerate(train_files) if i % 20 == 0]
    print(running_dir)
    # working directory setup
    os.makedirs(running_dir, exist_ok=True)

    # write config file
    cf.write(open(f"{running_dir}/train.ini", "w"))

    # cuda setup
    if args.gpu == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
        device = f"cuda:0"

    # import pytorch **after** cuda setup
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler
    from agent_model import GNNPolicyItem, GNNPolicyLoad, GNNPolicyAno

    # randomization setup
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # logging setup
    logfile = os.path.join(running_dir, "train_log.txt")
    if os.path.exists(logfile):
        os.remove(logfile)
    log(f"max_epochs: {args.epoch}", logfile)
    log(f"batch_size: {BATCH_SIZE}", logfile)
    log(f"pretrain_batch_size: {PRETRAIN_BATCH_SIZE}", logfile)
    log(f"valid_batch_size : {VALID_BATCH_SIZE }", logfile)
    log(f"lr: {LR}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # data setup
    valid_data = GraphDataset(valid_files)
    pretrain_data = GraphDataset(pretrain_files)

    pretrain_loader = torch_geometric.data.DataLoader(
        pretrain_data, PRETRAIN_BATCH_SIZE, shuffle=False
    )

    if POLICY_TYPE == 0:
        policy = GNNPolicyItem().to(device)
    elif POLICY_TYPE == 1:
        policy = GNNPolicyLoad().to(device)
    elif POLICY_TYPE == 2:
        policy = GNNPolicyAno().to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = Scheduler(optimizer, mode="min", patience=10, factor=0.2, verbose=True)
    for epoch in range(args.epoch + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            n = pretrain(policy, pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)
            # continue
        else:
            epoch_train_files = rng.choice(
                train_files,
                int(np.floor(TRAIN_NUM / BATCH_SIZE)) * BATCH_SIZE,
                replace=True,
            )
            train_data = GraphDataset(epoch_train_files)

            train_loader = torch_geometric.data.DataLoader(
                train_data, BATCH_SIZE, shuffle=True
            )
            train_loss, train_kacc = process(policy, train_loader, top_k, optimizer)
            log(
                f"TRAIN LOSS: {train_loss:0.3f} "
                + "".join(
                    [f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]
                ),
                logfile,
            )

        epoch_valid_files = rng.choice(
            valid_files,
            int(np.floor(VALID_NUM / BATCH_SIZE)) * BATCH_SIZE,
            replace=True,
        )
        valid_data = GraphDataset(epoch_valid_files)
        valid_loader = torch_geometric.data.DataLoader(
            valid_data, BATCH_SIZE, shuffle=True
        )
        valid_loss, valid_kacc = process(policy, valid_loader, top_k, None)
        log(
            f"VALID LOSS: {valid_loss:0.3f} "
            + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]),
            logfile,
        )

        scheduler.step(valid_loss)
        torch.save(
            policy.state_dict(), pathlib.Path(running_dir) / f"params_{epoch}",
        )
        if scheduler.num_bad_epochs == 0:
            torch.save(
                policy.state_dict(), pathlib.Path(running_dir) / "best_params.pkl"
            )
            log(f"  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log(f"  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log(f"  20 epochs without improvement, early stopping", logfile)
            break

    # load best parameters and run a final validation step
    policy.load_state_dict(torch.load(pathlib.Path(running_dir) / "best_params.pkl"))
    valid_loss, valid_kacc = process(policy, valid_loader, top_k, None)
    log(
        f"BEST VALID LOSS: {valid_loss:0.3f} "
        + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]),
        logfile,
    )
