from sklearn.metrics import roc_auc_score, auc
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler,
                              SequentialSampler)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import logging
import multiprocessing
import os
import random
import sys
import time
import math
import numpy as np
import torch
from tqdm import tqdm
import pickle
sys.path.append(os.getcwd())
from src.models import build_or_load_gen_model
from src.configs import add_args, set_dist
from src.utils import SimpleJITDPDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def load_scheduler_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    if args.warmup_steps == -1:
        args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )
    return scheduler, optimizer


def get_loader(data_file, args, tokenizer, pool, eval=False, batch_size=None, oversample=False, rand=False):
    def fn(features):
        return features
    global_rank = args.global_rank
    # only diff and message without code context
    dataset = SimpleJITDPDataset(
        tokenizer, pool, args, data_file, oversample=oversample)
    data_len = len(dataset)
    if global_rank == 0:
        logger.info(f"Data length: {data_len}.")

    if eval and args.evaluate_sample_size == -1:
        sampler = SequentialSampler(dataset)
    elif eval and args.evaluate_sample_size != -1 and len(dataset) > args.evaluate_sample_size:
        random.seed(args.seed)
        sampler = SubsetRandomSampler(random.sample(
            range(len(dataset)), args.evaluate_sample_size))
    elif args.n_gpu > 1 and args.no_cuda is False:
        sampler = DistributedSampler(dataset)
    elif eval is False and rand is True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    logger.info(f"Sample size: {len(sampler)}.")
    if batch_size is None:
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    else:
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, num_workers=args.cpu_count, collate_fn=fn)
    return dataset, sampler, dataloader


def save_model(model, optimizer, scheduler, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    config.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
    torch.save(
        optimizer.state_dict(),
        output_optimizer_file,
        _use_new_zipfile_serialization=False,
    )
    output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
    torch.save(
        scheduler.state_dict(),
        output_scheduler_file,
        _use_new_zipfile_serialization=False,
    )


def finetune(args, config, model, tokenizer, scheduler, optimizer):
    global_step, best_acc, best_f1 = 0, -1, -1
    save_steps = args.save_steps
    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)
    pool = multiprocessing.Pool(args.cpu_count)

    _, _, train_dataloader = get_loader(
        args.train_filename, args, tokenizer, pool, oversample=False, rand=True)
    model.init_classifier()
    model.train()
    model.zero_grad()
    for epoch in range(1, args.num_train_epochs + 1):
        save_seed = args.seed
        args.seed += epoch
        set_seeds(args.seed)
        args.seed = save_seed
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        bar = tqdm(train_dataloader, total=len(
            train_dataloader), desc="Training")
        for step, examples in enumerate(bar, 1):
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_id)
            ys = torch.tensor(
                [ex.y for ex in examples], dtype=torch.long
            ).to(args.device)
            loss = model(
                cls=True,
                SF=True,
                input_ids=source_ids,
                labels=ys,
                attention_mask=source_mask
            )
            if args.gpu_per_node > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if nb_tr_steps % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if args.global_rank == 0 and global_step % args.log_steps == 0:
                    train_loss = round(
                        tr_loss * args.gradient_accumulation_steps / nb_tr_steps,
                        4,
                    )
                    bar.set_description(
                        "step {}/{}: Train loss {}".format(global_step, args.train_steps, round(train_loss, 3)))

            if global_step == args.train_steps and args.global_rank == 0:
                output_dir = os.path.join(
                    args.output_dir, "checkpoints-last")
                save_model(model, optimizer, scheduler, output_dir, config)
                logger.info(f"Reach max steps {args.train_steps}.")
                time.sleep(5)
                return

            if args.global_rank == 0 and \
                    global_step % save_steps == 0 and \
                    nb_tr_steps % args.gradient_accumulation_steps == 0:

                output_dir = os.path.join(
                    args.output_dir, "result-checkpoints-" + str(global_step))
                _, _, valid_dataloader = get_loader(
                    args.dev_filename, args, tokenizer, pool, eval=True)
                cur_res = eval_metrics_epoch(
                    args, valid_dataloader, model, tokenizer)
                f1 = cur_res["f1"]
                logger.warning(f"Current Res: {cur_res}")
                output_dir = os.path.join(
                    args.output_dir, "result-checkpoints-" + str(global_step) + "-" + str(round(f1, 3)))
                save_model(model, optimizer, scheduler, output_dir, config)
                f1 = cur_res["f1"]
                if f1 > best_f1:
                    logger.info(
                        "  [%d] Best f1: %.2f ",
                        epoch, f1
                    )
                    best_f1 = f1
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-best-f1')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, optimizer, scheduler, output_dir, config)

                logger.info(
                    "Save the {}-step model and optimizer into {}".format(
                        global_step, output_dir
                    )
                )
                time.sleep(5)


def convert_dtype_dataframe(df, feature_name):
    df = df.astype({i: 'float32' for i in feature_name})
    return df


def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * \
        result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC']
                                         <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(
        buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort


def eval_R20E_E20R_Popt(test_features, gold, prob):
    test_features = test_features[['commit_hash', 'la', 'ld']]
    test_features['label'] = gold
    test_features = convert_dtype_dataframe(test_features, ['la', 'ld'])
    test_features['LOC'] = test_features['la'] + test_features['ld']

    loc_sum = sum(test_features['LOC'])
    test_features['defective_commit_prob'] = prob
    test_features['defect_density'] = test_features['defective_commit_prob'] / \
        test_features['LOC']  # predicted defect density
    test_features['actual_defect_density'] = test_features['label'] / \
        test_features['LOC']  # defect density

    result_df = test_features.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(
        by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(
        by='actual_defect_density', ascending=True)
    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()
    real_buggy_commits = result_df[result_df['label'] == 1]

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * loc_sum
    buggy_line_20_percent = result_df[result_df['cum_LOC']
                                      <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(
        buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(
        math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(
        buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                           real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                        real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df,
                                                                              real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(
            predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(
            actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(
            actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                     auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt


def eval_metrics_epoch(args, eval_dataloader, model, tokenizer, show_bar=False, test=False):
    # Start evaluating model
    logger.info("  " + "***** Running metrics evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    pred, gold, probs, res = [], [], [], {}
    if show_bar is True:
        current_loader = tqdm(eval_dataloader, total=len(eval_dataloader))
    else:
        current_loader = eval_dataloader
    with torch.no_grad():
        for step, examples in enumerate(current_loader, 1):
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_id)
            logits = model(
                cls=True,
                SF=True,
                input_ids=source_ids,
                labels=None,
                attention_mask=source_mask
            )

            prediction = torch.argmax(logits, dim=-1).cpu().numpy()

            pred.extend(prediction)
            gold.extend([ex.y for ex in examples])
            if test is True:
                prob = torch.nn.functional.softmax(
                    logits, dim=1).data.cpu().numpy()[:, 1].tolist()
                probs.extend(prob)

    acc = accuracy_score(gold, pred)
    if test is True:
        auc = roc_auc_score(gold, probs)
        test_features_filename = args.test_filename.replace(
            'changes', 'features').replace('.jsonl', '.pkl')
        test_metrics = pickle.load(open(test_features_filename, 'rb'))
        R20E, E20R, Popt = eval_R20E_E20R_Popt(test_metrics, gold, probs)
        res["auc"] = auc
        res["R20E"] = R20E
        res["E20R"] = E20R
        res["Popt"] = Popt

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average='binary')
    res["precision"] = precision
    res["recall"] = recall
    res["f1"] = f1
    res["accuracy"] = acc

    return res


def test(args, model, tokenizer):
    set_seeds(args.seed)
    pool = multiprocessing.Pool(args.cpu_count)

    if os.path.exists("{}/checkpoint-best-f1/pytorch_model.bin".format(args.output_dir)) and \
            (args.load_model_path is None or args.do_train is True):
        model.load_state_dict(
            torch.load(
                "{}/checkpoint-best-f1/pytorch_model.bin".format(args.output_dir), map_location="cpu")
        )
        logger.info(
            "Load model from {}/checkpoint-best-f1/pytorch_model.bin".format(args.output_dir))
    model.eval()
    _, _, test_dataloader = get_loader(
        args.test_filename, args, tokenizer, pool, eval=False, batch_size=args.eval_batch_size)

    cur_res = eval_metrics_epoch(
        args, test_dataloader, model, tokenizer, show_bar=True, test=True)
    logger.warning(f"Test Res: {cur_res}")


def main(args):
    set_dist(args)
    set_seeds(args.seed)
    args.global_rank = 0
    config, model, tokenizer = build_or_load_gen_model(args)
    if not os.path.exists("{}/checkpoints-last".format(args.output_dir)):
        os.makedirs("{}/checkpoints-last".format(args.output_dir))

    if os.path.exists("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir)) and \
            args.load_model_path is None:
        model.load_state_dict(
            torch.load(
                "{}/checkpoints-last/pytorch_model.bin".format(args.output_dir), map_location="cpu")
        )
        logger.info(
            "Load model from {}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
    scheduler, optimizer = load_scheduler_optimizer(args, model)
    if args.do_train:
        finetune(args, config, model, tokenizer, scheduler, optimizer)
    if args.do_test:
        test(args, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger(
        "transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Training finished.")
