import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (ConcatDataset, DataLoader, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from src.utils import SimpleGenDataset

from src.evaluator.smooth_bleu import bleu_fromstr
from src.configs import add_args, set_dist
from src.models import build_or_load_gen_model

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
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)) and not args.load_optimizer_path:
    #     optimizer.load_state_dict(
    #         torch.load(
    #             "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
    #             map_location="cpu",
    #         )
    #     )
    #     scheduler.load_state_dict(
    #         torch.load(
    #             "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
    #             map_location="cpu",
    #         )
    #     )
    #     logger.info("Load optimizer from {}/checkpoints-last/optimizer.pt".format(args.output_dir))
    #     logger.info("Load scheduler from {}/checkpoints-last/scheduler.pt".format(args.output_dir))
    
    if args.load_optimizer_path:
        optimizer.load_state_dict(
            torch.load(
                "{}/optimizer.pt".format(args.load_optimizer_path),
                map_location="cpu",
            )
        )
        scheduler.load_state_dict(
            torch.load(
                "{}/scheduler.pt".format(args.load_optimizer_path),
                map_location="cpu",
            )
        )
        logger.info("Load optimizer from {}/optimizer.pt".format(args.load_optimizer_path))
        logger.info("Load scheduler from {}/scheduler.pt".format(args.load_optimizer_path))
        
    return scheduler, optimizer


def get_loader(data_file, args, tokenizer, pool, eval=False, batch_size=None, rand=False):
    def fn(features):
        return features
    global_rank = args.global_rank
    dataset = SimpleGenDataset(tokenizer, pool, args, data_file)

    data_len = len(dataset)
    if global_rank == 0:
        logger.info(f"Data length: {data_len}.")
    if eval and args.evaluate_sample_size == -1:
        sampler = SequentialSampler(dataset)
    elif eval and args.evaluate_sample_size != -1:
        # random.seed(args.seed)
        sampler = SubsetRandomSampler(random.sample(range(len(dataset)), args.evaluate_sample_size))
    elif args.n_gpu > 1:
        sampler = DistributedSampler(dataset)
    elif eval is False and rand is True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
        
    logger.info(f"Sample size: {len(sampler)}.")
    if batch_size is None:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=args.cpu_count, collate_fn=fn)
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
    global_step, best_bleu_em, best_bleu, best_ppl = 0, -1, -1, 1e6
    save_steps = args.save_steps
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    pool = multiprocessing.Pool(args.cpu_count)
    _, _, train_dataloader = get_loader(args.train_filename, args, tokenizer, pool, rand=True) 
    # exit(0) # test
    model.train()    
    for epoch in range(1, args.num_train_epochs + 1):
        save_seed = args.seed
        args.seed += epoch
        set_seeds(args.seed)
        args.seed = save_seed
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        for step, examples in enumerate(bar, 1):
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(args.device)
            source_labels = None
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
                    bar.set_description("step {}/{}: Train loss {}".format(global_step, args.train_steps, round(train_loss, 3)))
            
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
                
                output_dir = os.path.join(args.output_dir, "result-checkpoints-" + str(global_step))
                _, _, valid_dataloader = get_loader(args.dev_filename, args, tokenizer, pool, eval=True)  
                bleu_result = eval_bleu_epoch(args, valid_dataloader, args.dev_filename, model, tokenizer, output_dir, show_bar=False, eval=True)
                _, _, valid_dataloader = get_loader(args.dev_filename, args, tokenizer, pool, eval=True)  
                eval_ppl = eval_ppl_epoch(args, valid_dataloader, args.dev_filename, model, tokenizer)
                dev_bleu, dev_em = bleu_result['bleu'], bleu_result['em']
                output_dir = os.path.join(args.output_dir, "result-checkpoints-" + str(global_step) + "-" + str(dev_bleu))
                save_model(model, optimizer, scheduler, output_dir, config)
                dev_bleu_em = 0.3*dev_bleu + 0.7*dev_em
                if dev_bleu_em > best_bleu_em:
                    logger.info(
                            "  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    epoch, dev_bleu_em, dev_bleu, dev_em
                        )
                    best_bleu_em = dev_bleu_em
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu-em')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, optimizer, scheduler, output_dir, config)
                    
                if dev_bleu > best_bleu:
                    logger.info(
                            "  [%d] Best bleu: %.2f (bleu: %.2f, em: %.2f)",
                                    epoch, dev_bleu, dev_bleu, dev_em
                        )
                    best_bleu = dev_bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, optimizer, scheduler, output_dir, config)
                    
                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    best_ppl = eval_ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info("Save the best ppl model into %s", output_dir)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    
                logger.info(
                    "Save the {}-step model and optimizer into {}".format(
                        global_step, output_dir
                    )
                )
                time.sleep(5)
                model.train()
            
            
def eval_bleu_epoch(args, eval_dataloader, eval_filename, model, tokenizer, output_dir=None, show_bar=False, eval=False):
    logger.info(f"  ***** Running bleu evaluation on {eval_filename} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if output_dir is None:
        output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pred_ids, ex_ids, indexes = [], [], []
    if show_bar is True:
        current_loader = tqdm(eval_dataloader, total=len(eval_dataloader))
    else:
        current_loader = eval_dataloader
    for step, examples in enumerate(current_loader, 1):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.device)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(source_ids,
                            attention_mask=source_mask,
                            use_cache=True,
                            num_beams=args.beam_size,
                            early_stopping=True,
                            max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
        indexes.extend(ids)
    # [1:] to remove beginning '<msg>'
    pred_nls = [tokenizer.decode(id[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
            
    eval_accs, all_golds = [], []
    
    with open(eval_filename, "r") as f:
        for line in f:
            js = json.loads(line)
            if "msg" in js:
                msg = js["msg"]
            elif "nl" in js:
                msg = js["nl"]
            else:
                continue
            all_golds.append(msg)
    golds = [all_golds[x] for x in indexes]
    output_prefix = os.path.splitext(os.path.basename(eval_filename))[0]
    if eval is False:
        with open(os.path.join(output_dir, output_prefix + "-preds.txt"), "w", encoding="utf-8") as f:
            for pred in pred_nls:
                f.write(pred.strip() + "\n")
        with open(os.path.join(output_dir, output_prefix + "-golds.txt"), "w", encoding="utf-8") as f:
            for gold in golds:
                f.write(gold.strip() + "\n")
        if args.load_model_path:
            output_dir = os.path.dirname(args.load_model_path)
            with open(os.path.join(output_dir, output_prefix + "-preds.txt"), "w", encoding="utf-8") as f:
                for pred in pred_nls:
                    f.write(pred.strip() + "\n")
            with open(os.path.join(output_dir, output_prefix + "-golds.txt"), "w", encoding="utf-8") as f:
                for gold in golds:
                    f.write(gold.strip() + "\n")
                
    for i in range(len(pred_nls)):
        chars = "(_)`."
        for c in chars:
            pred_nls[i] = pred_nls[i].replace(c, " " + c + " ")
            pred_nls[i] = " ".join(pred_nls[i].split())
            golds[i] = golds[i].replace(c, " " + c + " ")
            golds[i] = " ".join(golds[i].split())
        eval_accs.append(pred_nls[i].strip() == golds[i].strip())
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    result = {'em': np.mean(eval_accs) * 100, 'bleu': bleu}
    model.train()
    return result


def eval_ppl_epoch(args, eval_dataloader, eval_filename, model, tokenizer):
    logger.info(f"  ***** Running ppl evaluation on {eval_filename} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    eval_loss, eval_steps, batch_num = 0, 0, 0
    for step, examples in enumerate(eval_dataloader, 1):
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
    eval_loss = eval_loss / eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    model.train()
    return float(perplexity)


def test(args, model, tokenizer):
    set_seeds(args.seed)
    pool = multiprocessing.Pool(args.cpu_count)
    if os.path.exists("{}/checkpoint-best-bleu/pytorch_model.bin".format(args.output_dir)) and \
        args.load_model_path is None:
        model.load_state_dict(
            torch.load(
                "{}/checkpoint-best-bleu/pytorch_model.bin".format(args.output_dir))
        )
        logger.info("Load model from {}/checkpoint-best-bleu/pytorch_model.bin".format(args.output_dir))
    _, _, test_dataloader = get_loader(args.test_filename, args, tokenizer, pool, batch_size=args.eval_batch_size, eval=False)
    model.eval()
    result = eval_bleu_epoch(args, test_dataloader, args.test_filename, model, tokenizer, show_bar=True, eval=False)
    test_bleu, test_em = result['bleu'], result['em']
    logger.warning(f"Test BLEU: {test_bleu}, Test ACC: {test_em}")
    
    
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
                "{}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
        )
        logger.info("Load model from {}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
    
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
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Training finished.")