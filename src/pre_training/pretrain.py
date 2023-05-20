import argparse
import os
import random
import sys
import numpy as np
sys.path.append(os.getcwd())
import torch
import multiprocessing
from transformers.utils import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import torch.distributed as dist
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from src.models import build_or_load_gen_model
from src.configs import add_args, set_dist
from src.utils import TextDataset
from tqdm import tqdm
import time

logger = logging.get_logger(__name__)


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def find_langs_in_data_dir(data_dir):
    return list(set(
        ["_".join(f[:-6].split("_")[:-1])
         for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    ))


def num_parameters(model):
    model_parameters = model.parameters()
    return sum([np.prod(p.size()) for p in model_parameters])


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
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)):
        optimizer.load_state_dict(
            torch.load(
                "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
        scheduler.load_state_dict(
            torch.load(
                "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
    return scheduler, optimizer


def get_loaders(data_list, args, tokenizer, pool):
    def fn(features):
        return features
    world_size = 1
    assert len(data_list) > 0, "Empty datalist."
    each_len = len(data_list) // world_size
    data_list = data_list[: each_len * world_size]
    random.shuffle(data_list)       # this will shuffle data chunks
    for data_file in data_list:
        logger.info(f"Start data files {data_file}.")
        # add concat dataset
        datasets = [TextDataset(tokenizer, pool, args, data_file)]
        # truncate to the same length
        data_len = sum(len(dataset) for dataset in datasets)
        # to keep same training size for different gpus
        data_len = torch.tensor(data_len).to(args.device)
        if world_size > 1:
            dist.all_reduce(data_len, op=dist.ReduceOp.MIN)
        data_len = data_len.item()
        prev_len = sum(len(dataset) for dataset in datasets[:-1])
        last_len = data_len - prev_len
        datasets[-1].reset_len(last_len)

        dataset = ConcatDataset(datasets)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=args.cpu_count, collate_fn=fn)
        logger.info(f"Finish data files {data_file}.")
        yield dataset, sampler, dataloader


def pretraining(args, config, model, tokenizer, scheduler, optimizer):
    not_loss_dec_cnt, global_step, best_ppl = 0, 0, 1e6
    save_steps = args.save_steps
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    pool = multiprocessing.Pool(args.cpu_count)
        
    data_list =[os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir) if x.startswith("train") and x.endswith(".jsonl")]
    eval_data_list = [os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir) if x.startswith("valid") and x.endswith(".jsonl")]

    for epoch in range(1, args.num_train_epochs + 1):
  
        save_seed = args.seed
        args.seed += epoch
        set_seeds(args.seed)
        args.seed = save_seed
        random.shuffle(data_list)

        # WARNING: this is a iterator, to save memory
        data_tuples = get_loaders(data_list, args, tokenizer, pool)
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for dataset, sampler, dataloader in data_tuples:
            bar = tqdm(dataloader, total=len(dataloader), desc="Training")
            for step, examples in enumerate(bar, 1):
                source_ids = torch.tensor(
                    [ex.source_ids for ex in examples], dtype=torch.long
                ).to(args.device)
                source_labels = torch.tensor(
                    [ex.source_labels for ex in examples], dtype=torch.long
                ).to(args.device)
                target_ids = torch.tensor(
                    [ex.target_ids for ex in examples], dtype=torch.long
                ).to(args.device)

                source_mask = source_ids.ne(tokenizer.pad_id)
                target_mask = target_ids.ne(tokenizer.pad_id)

                loss = model(
                    input_ids=source_ids,
                    input_labels=source_labels,
                    decoder_input_ids=target_ids,
                    attention_mask=source_mask,
                    decoder_attention_mask=target_mask,
                    encoder_loss=False
                )

                if args.gpu_per_node > 1:
                    loss = loss.mean() 
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
                        bar.set_description("step {}/{}: Train loss {}".format(global_step, args.train_steps, round(train_loss, 3)))
                if global_step == args.train_steps and args.global_rank == 0:
                    # end training
                    output_dir = os.path.join(
                        args.output_dir, "checkpoints-last")
                    if args.save_last_checkpoints:
                        save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(f"Reach max steps {args.train_steps}.")
                    time.sleep(5)
                    return

                if args.global_rank == 0 and \
                        global_step % save_steps == 0 and \
                        nb_tr_steps % args.gradient_accumulation_steps == 0:
                    eval_ppl = round(
                        evaluate(eval_data_list, args, model, tokenizer, pool), 2)
                    logger.info(f"Evaluate perplexity: {eval_ppl}")
                    output_dir = os.path.join(
                        args.output_dir, "checkpoints-" + str(global_step))
                    if args.always_save_model:
                        save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(
                        "Save the {}-step model and optimizer into {}".format(
                            global_step, output_dir
                        )
                    )
                    if eval_ppl < best_ppl:
                        not_loss_dec_cnt = 0
                        best_ppl = eval_ppl
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                        save_model(model, optimizer, scheduler, output_dir, config)
                        logger.info("Save the best ppl model into %s", output_dir)
                    else:
                        not_loss_dec_cnt += 1
                        logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    time.sleep(5)
                    
    # reach max epochs, not max steps
    if args.global_rank == 0:
        # Save the final checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoints-last")
        save_model(model, optimizer, scheduler, output_dir, config)
        logger.info(
            "Save the trained model and optimizer into {}".format(output_dir))
        time.sleep(5)


def evaluate(data_list, args, model, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Eval start  data files {data_list}.")
    
    # add concat dataset
    datasets = [TextDataset(tokenizer, pool, args, data_file, random_sample_num=args.evaluate_sample_size)
                for data_file in data_list]
    
    dataset = ConcatDataset(datasets)
    if args.evaluate_sample_size == -1:
        sampler = SequentialSampler(dataset)
    else:
        sampler = SubsetRandomSampler(np.random.choice(range(len(dataset)), args.evaluate_sample_size))
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
                            num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_list}.")
    logger.info(f"Eval dataset size:{len(dataloader)*args.train_batch_size}")
    eval_loss = 0.0
    eval_steps = 0
    model.eval()
    
    bar = tqdm(dataloader, total=len(dataloader), desc="Eval")
    for step, examples in enumerate(bar, 1):
        with torch.no_grad():
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(args.device)
            source_labels = torch.tensor(
                [ex.source_labels for ex in examples], dtype=torch.long
            ).to(args.device)
            target_ids = torch.tensor(
                [ex.target_ids for ex in examples], dtype=torch.long
            ).to(args.device)
            source_mask = source_ids.ne(tokenizer.pad_id)
            target_mask = target_ids.ne(tokenizer.pad_id)

            loss = model(
                input_ids=source_ids,
                input_labels=source_labels,
                decoder_input_ids=target_ids,
                attention_mask=source_mask,
                decoder_attention_mask=target_mask,
            )

            eval_loss += loss.mean().item()
        eval_steps += 1
        if eval_steps % args.log_steps == 0:
            bar.set_description("Eval loss {}".format(round(eval_loss / eval_steps, 3)))

    eval_loss = eval_loss / eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    model.train()
    return float(perplexity)


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


def main(args):
    set_dist(args)
    set_seeds(args.seed)
    args.global_rank = 0
    config, model, tokenizer = build_or_load_gen_model(args)
    logger.info(f"Starting Training from {args.model_type}")
    logger.info(f"Total parameters : {num_parameters(model)}")
    
    if not os.path.exists("{}/checkpoints-last".format(args.output_dir)):
        os.makedirs("{}/checkpoints-last".format(args.output_dir))
        
    if os.path.exists("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir)):
        model.load_state_dict(
            torch.load(
                "{}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
        )
        logger.info("Load model from {}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
    
    scheduler, optimizer = load_scheduler_optimizer(args, model)
        
    pretraining(args, config, model, tokenizer, scheduler, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    set_seeds(args.seed)
    logger.info(args)
    main(args)
    logger.info("Pre-training finished.")
