import json
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import set_seed, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor

from data_helper import get_tensor_dataset, load_raw_dataset, format_input, format_output, Data_Collator_for_Training
from generate_utils import generation, generation_with_prefix

import logging
def get_logger(name, log_path=None):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    if log_path:
        handler = logging.FileHandler(log_path, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def evaluate(dataset, model, args):

    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, 
                    sampler=data_sampler, 
                    batch_size=args.eval_batch_size)
    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Eval Iteration")

    loss_sum = 0.
    ppl_sum = 0.
    tokens_sum = 0.
    for step, batch in enumerate(epoch_iterator):

        input_ids, attention_mask, text_labels = tuple(t.to(args.device) for t in batch) 

        with torch.no_grad():
            outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=text_labels
                    )

            loss = outputs.loss
            num_tokens = (text_labels != -100).sum().item()
            tokens_sum += num_tokens
            ppl_sum += outputs.loss.item() * num_tokens

            loss_sum += loss.item()
        if args.debug and step > 10:
            break

    loss_sum /= (step + 1)
    ppl_sum = math.exp(ppl_sum / tokens_sum)

    return {"loss": loss_sum, "perplexity": ppl_sum}

def inference(dataset, output_path, model, tokenizer, args):
    batch_input = []
    batch_output_prefix = []
    batch_example = []
    example_idx = 0
    if output_path is not None:
        fw = open(output_path, 'w')
    accuracy = 0.
    generated_explanation = []
    if args.add_task_prefix:
        output_prefix = '<pad> [factual]' 
    else:
        output_prefix = '<pad> ' 
    model.eval()
    for example in tqdm(dataset):
        batch_example.append(example)
        input_seq = format_input(example.question, example.choices)

        batch_input.append(input_seq)
        batch_output_prefix.append(output_prefix)

        if len(batch_input) == args.eval_batch_size or example_idx == len(dataset) - 1:
            inputs = tokenizer(batch_input, padding='max_length', max_length=args.max_enc_length, truncation=True, return_tensors='pt').to(args.device)
            decoder_input_ids = tokenizer(batch_output_prefix, add_special_tokens=False, return_tensors='pt').to(args.device).input_ids
            batch_output = generation_with_prefix(inputs, decoder_input_ids, model, tokenizer, args)
            for example, output in zip(batch_example, batch_output):
                answer_prefix = "So the answer is "
                generation_split = output.split(answer_prefix)
                generated_explanation.append(generation_split[0].strip())
                if len(generation_split) == 1:
                    continue
                explanation = generation_split[0].strip()
                prediction = generation_split[1].strip()
                if prediction == example.answer:
                    accuracy += 1
                if output_path is not None:
                    output_example = {"id": example.qid}
                    output_example["question"] = example.question
                    output_example["answer"] = prediction
                    if example.choices is not None:
                        output_example["choices"] = [span.split(') ')[1].strip() for span in example.choices.split('(')[1:]]
                    else:
                        if example.is_statement:
                            output_example["choices"] = ["false", "true"]
                        else:
                            output_example["choices"] = ["no", "yes"]
                    if not args.without_explanation:
                        output_example["explanation"] = explanation

                    fw.write(json.dumps(output_example)+'\n')
            batch_input = []
            batch_example = []
            batch_output_prefix = []
        example_idx += 1
        if args.debug and example_idx > 50:
            break

    if output_path is not None:
        fw.close()
    return accuracy * 100. / len(dataset), generated_explanation

def inference_with_oracle(dataset, model, tokenizer, args):
    example_idx = 0
    accuracy = 0.
    model.eval()
    for example in tqdm(dataset):
        input_seq = format_input(example.question, example.choices)
        answer_prefix = " So the answer is"

        inputs = tokenizer(input_seq, padding='max_length', max_length=args.max_enc_length, truncation=True, return_tensors='pt').to(args.device)
        if args.add_task_prefix:
            output_prefix = '<pad> [factual]' + example.explanation + answer_prefix 
        else:
            output_prefix = '<pad> ' + example.explanation + answer_prefix 
        decoder_input_ids = tokenizer(output_prefix, add_special_tokens=False, return_tensors='pt').to(args.device).input_ids
        prediction = generation_with_prefix(inputs, decoder_input_ids, model, tokenizer, args)[0].strip()
        if prediction == example.answer:
            accuracy += 1
        example_idx += 1
        if args.debug and example_idx > 50:
            break

    return accuracy * 100. / len(dataset)

def inference_with_perturb(dataset, explanations, model, tokenizer, args, replace_ratio=0.5):
    example_idx = 0
    accuracy = 0.
    model.eval()
    for example in tqdm(dataset):
        input_seq = format_input(example.question, example.choices)
        answer_prefix = " So the answer is"
        answer_prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)

        inputs = tokenizer(input_seq, padding='max_length', max_length=args.max_enc_length, truncation=True, return_tensors='pt').to(args.device)
        explanation_ids = tokenizer.encode(explanations[example_idx], add_special_tokens=False)
        explanation_length = len(explanation_ids)
        mask_idx = random.sample(range(explanation_length), int(explanation_length * replace_ratio))
        pert_explanation_ids = [random.choice(range(len(tokenizer))) if _idx in mask_idx else explanation_ids[_idx] for _idx in range(explanation_length)]
        if args.add_task_prefix:
            decoder_input_ids = [tokenizer.pad_token_id] + tokenizer.encode('[factual]', add_special_tokens=False) + pert_explanation_ids + answer_prefix_ids
        else:
            decoder_input_ids = [tokenizer.pad_token_id] + pert_explanation_ids + answer_prefix_ids
        decoder_input_ids = torch.tensor([decoder_input_ids]).to(args.device)
        prediction = generation_with_prefix(inputs, decoder_input_ids, model, tokenizer, args)[0].strip()
        if prediction == example.answer:
            accuracy += 1
        example_idx += 1
        if args.debug and example_idx > 50:
            break

    return accuracy * 100. / len(dataset)

def main(args, seed):
    # ----------------------------------------------------- #
    # prepare logger
    log_path = os.path.join(args.save_dir, 'train_seed{}.log'.format(seed))
    logger = get_logger("model", log_path)
    logger.info('args: {}'.format(args))

    # ----------------------------------------------------- #
    # model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='../cache/')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir='../cache/')
    model.to(args.device)

    # ----------------------------------------------------- #
    # data

    trainset = get_tensor_dataset('train', tokenizer, args)
    train_sampler = RandomSampler(trainset)
    train_dataloader = DataLoader(trainset,
                collate_fn=None,
                sampler=train_sampler, 
                batch_size=args.train_batch_size,
    )

    if args.counterfactual_alpha > 0:
        trainset1 = get_tensor_dataset('train.counterfactual' , tokenizer, args, counterfactual=True)
        train_sampler1 = RandomSampler(trainset1)
        train_dataloader_counterfactual = DataLoader(trainset1, collate_fn=None, sampler=train_sampler1, batch_size=args.train_batch_size)

    devset = get_tensor_dataset('dev', tokenizer, args)

    # ----------------------------------------------------- #
    # optimization
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False
                )

    num_update_steps_per_epoch = len(train_dataloader)
    t_total = num_update_steps_per_epoch // args.grad_step * args.num_epoch
    warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # ----------------------------------------------------- #
    # training loop
    model_ckpt = os.path.join(args.save_dir, 'model_seed{}.ckpt'.format(seed))
    output_path = os.path.join(args.save_dir, 'validation_seed{}.jsonl'.format(seed))
    global_step = 0
    best_dev_loss = 1e19
    step_nogress = 0
    optimizer.zero_grad()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.smoothing_factor)
    if args.debug:
        args.num_epoch = 1
    for epoch in trange(int(args.num_epoch), desc="Epoch"):
        train_loss = 0.
        counterfactual_loss = 0.
        model.train()
        epoch_iterator = tqdm(train_dataloader, desc="Train Iteration at Epoch {}".format(epoch), total=num_update_steps_per_epoch)
        if args.counterfactual_alpha > 0:
            counterfactual_iterator = iter(train_dataloader_counterfactual)
        for step, batch in enumerate(epoch_iterator):

            input_ids, attention_mask, labels = tuple(t.to(args.device) for t in batch) 

            outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            outputs_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

            loss = (1 - args.counterfactual_alpha) * outputs_loss

            if args.counterfactual_alpha > 0:
                try:
                    counterfactual_batch = next(counterfactual_iterator)
                except StopIteration:
                    counterfactual_iterator = iter(train_dataloader_counterfactual)
                    counterfactual_batch = next(counterfactual_iterator)
                input_ids, attention_mask, labels, decoder_input_ids = tuple(t.to(args.device) for t in counterfactual_batch) 

                counterfactual_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            # labels=labels,
                        )
                counterfactual_outputs_loss = loss_fct(counterfactual_outputs.logits.view(-1, counterfactual_outputs.logits.size(-1)), labels.view(-1))
                loss += args.counterfactual_alpha * counterfactual_outputs_loss

            loss /= args.grad_step
            loss.backward()
            if (global_step + 1) % args.grad_step == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            train_loss += outputs_loss.item() # * args.grad_step
            if args.counterfactual_alpha > 0:
                counterfactual_loss += counterfactual_outputs_loss.item()
            global_step += 1
            epoch_iterator.set_description("Epoch {} loss {:.4f} counter {:.4f}".format(epoch, train_loss / (step + 1), counterfactual_loss / (step + 1)))
            if args.debug and global_step > 10:
                break

        train_loss /= (step + 1)
        counterfactual_loss /= (step + 1)
        log = 'Epoch: {:03d} Train loss: {:.4f} Counterfacual loss: {:.4f}'
        logger.info(log.format(epoch, train_loss, counterfactual_loss))

        dev_result = evaluate(devset, model, args)
        log = 'Epoch: {:03d}, dev loss {:.4f}, perplexity {:.4f}'
        if dev_result["loss"] < best_dev_loss:
            torch.save({'ckpt': model.state_dict(), 'args': args}, model_ckpt)
            log += ' best'
            best_dev_loss = dev_result["loss"]
            step_nogress = 0
        else:
            step_nogress += 1
        logger.info(log.format(epoch, dev_result["loss"], dev_result["perplexity"]))
        if step_nogress > args.num_epoch_early_stopping and global_step > warmup_steps:
            break

    return_result = {}
    model.load_state_dict(torch.load(model_ckpt)['ckpt'])
    for split in ['test']:
        testset = load_raw_dataset(split, args)
        output_path = os.path.join(args.save_dir, '{}_seed{}.jsonl'.format(split, seed))

        accuracy, explanations = inference(testset, output_path, model, tokenizer, args)
        if split == 'test':
            return_result["accuracy_inference"] = accuracy
            log = 'Epoch: {:03d}, inference accuracy: {:.4f}'
            logger.info(log.format(-1, accuracy))
            if not args.without_explanation:
                return_result["accuracy_oracle"] = inference_with_oracle(testset, model, tokenizer, args)
                log = 'Epoch: {:03d}, oracle accuracy: {:.4f}'
                logger.info(log.format(-1, return_result["accuracy_oracle"]))
                return_result["accuracy_perturb"] = inference_with_perturb(testset, explanations, model, tokenizer, args)
                log = 'Epoch: {:03d}, perturb accuracy: {:.4f}'
                logger.info(log.format(-1, return_result["accuracy_perturb"]))

    if not args.save_ckpt:
        os.remove(model_ckpt)
    return return_result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--save_dir', '-o', type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--save_ckpt", action='store_true')
    parser.add_argument("--add_task_prefix", action='store_true')

    # model
    parser.add_argument('--model_name', '-m', type=str)
    parser.add_argument('--max_enc_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=128)

    # training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--num_epoch', type=float, default=1000)
    parser.add_argument('--num_epoch_early_stopping', type=int, default=10)

    # method
    parser.add_argument("--without_explanation", action='store_true')
    parser.add_argument('--counterfactual_alpha', type=float, default=0)
    parser.add_argument('--smoothing_factor', type=float, default=0)

    # inference
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument("--overwrite_output", action='store_true')

    # gpu and workers option
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    eval_result_all_split = {}
    for seed in range(5):
        set_seed(seed)
        eval_result = main(args, seed)
        for split in eval_result:
            if split not in eval_result_all_split:
                eval_result_all_split[split] = []
            eval_result_all_split[split].append(eval_result[split])
    output_result = {}
    for split in eval_result_all_split:
        output_result[split] = {
                "accuracy_mean": np.mean(eval_result_all_split[split]),
                "accuracy_std": np.std(eval_result_all_split[split]),
        }
    with open(os.path.join(args.save_dir, 'evaluation_results.json'), 'w') as fw:
        json.dump(output_result, fw, indent=4)
