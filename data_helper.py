import json
import os 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from torch.utils.data import Dataset, TensorDataset

@dataclass(frozen=True)
class InputExample:

    qid: str
    question: str
    explanation: List[str]
    choices: str
    answer: str
    is_statement: bool

class TrainingDataset(Dataset):

    features: List[InputExample]

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputExample:
        return self.features[i]

def load_raw_dataset(split, args):
    data_path = os.path.join('./outputs', args.dataset, '{}.jsonl'.format(split))
    dataset = []

    with open(data_path, 'r') as fr:
        for line_idx, line in tqdm(enumerate(fr), desc='processing {}'.format(data_path)):
            example = json.loads(line)
            dataset.append(
                    InputExample(
                            qid=example["id"],
                            question=example["statement"] if "statement" in example else example["question"],
                            explanation=example["explanation"],
                            choices=example["choices"] if "choices" in example else None,
                            answer=example["answer"],
                            is_statement="statement" in example,
                    )
            )


    for example in dataset[:2]:
        print("*** Example ***")
        print(example)

    return dataset
    
def get_label_tensor(raw_label, tokenizer, args):
    label_ids = tokenizer.encode(raw_label, add_special_tokens=False)
    label_ids += [tokenizer.eos_token_id]
    label_ids = label_ids[:args.max_dec_length] 
    label_ids += [-100] * (args.max_dec_length - len(label_ids))
    return label_ids

def get_label_tensor_answer_only(raw_label, raw_label_without_answer, tokenizer, args):
    label_ids = tokenizer.encode(raw_label, add_special_tokens=False)
    label_ids += [tokenizer.eos_token_id]
    label_ids = label_ids[:args.max_dec_length] 
    label_ids += [-100] * (args.max_dec_length - len(label_ids))

    label_ids_without_answer = tokenizer.encode(raw_label_without_answer, add_special_tokens=False)
    label_ids_without_answer = label_ids_without_answer[:args.max_dec_length]

    label_ids_answer_only = label_ids.copy()
    for idx in range(len(label_ids_without_answer)):
        label_ids_answer_only[idx] = -100

    decoder_input_ids = [tokenizer.pad_token_id] + [tokenizer.pad_token_id if _id == -100 else _id for _id in label_ids[:-1]]

    return decoder_input_ids, label_ids_answer_only

def format_input(context, choices=None, counterfactual=False, add_task_prefix=True):
    input_seq = ""
    if add_task_prefix:
        if counterfactual:
            input_seq += "[counterfactual] "
        else:
            input_seq += "[factual] "
    input_seq += context.strip()
    if choices is not None:
        input_seq += " \\n {}".format(choices.strip())
    return input_seq

def format_output(explanation, answer, counterfactual=False, without_explanation=False, add_task_prefix=True):
    output_seq = ""
    if add_task_prefix:
        if counterfactual:
            output_seq += "[counterfactual] "
        else:
            output_seq += "[factual] "

    if not without_explanation:
        output_seq += explanation.strip()
    output_seq += ' So the answer is '
    output_seq_with_answer = output_seq + answer.strip()
    return output_seq_with_answer, output_seq.strip()

class Data_Collator_for_Training(object):
    def __init__(self, tokenizer, args, counterfactual=False):
        self.tokenizer = tokenizer
        self.args = args
        self.counterfactual = counterfactual

    def __call__(self, examples):
        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        decoder_label_tensor = []
        decoder_input_ids_tensor = []

        for example_idx, example in enumerate(examples):
            input_seq = format_input(example.question, example.choices, counterfactual=self.counterfactual, add_task_prefix=self.args.add_task_prefix)
            inputs = self.tokenizer(input_seq, padding='max_length', max_length=self.args.max_enc_length, truncation=True)

            if isinstance(example.explanation, list):
                explanation = random.choice(example.explanation)
            else:
                explanation = example.explanation
            output_seq, output_seq_without_answer = format_output(explanation, example.answer, counterfactual=self.counterfactual, without_explanation=self.args.without_explanation, add_task_prefix=self.args.add_task_prefix)

            encoder_input_tensor.append(inputs['input_ids'])
            encoder_attention_mask_tensor.append(inputs['attention_mask'])

            if self.counterfactual:
                decoder_input_ids, decoder_label = get_label_tensor_answer_only(output_seq, output_seq_without_answer, self.tokenizer, self.args)
                decoder_input_ids_tensor.append(decoder_input_ids)
                decoder_label_tensor.append(decoder_label)
            else:
                decoder_label_tensor.append(get_label_tensor(output_seq, self.tokenizer, self.args))

        if self.counterfactual:
            return tuple(torch.tensor(t) for t in [encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor, decoder_input_ids_tensor])
        else:
            return tuple(torch.tensor(t) for t in [encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor])

def get_tensor_dataset(split, tokenizer, args, counterfactual=False):
    data_path = os.path.join('./data', args.dataset, '{}.jsonl'.format(split))

    encoder_input_tensor = []
    encoder_attention_mask_tensor = []
    decoder_label_tensor = []
    decoder_input_ids_tensor = []

    with open(data_path, 'r') as fr:
        for line_idx, line in tqdm(enumerate(fr), desc='processing {}'.format(data_path)):
            example = json.loads(line)

            if "question" in example:
                if "choices" in example:
                    input_seq = format_input(example["question"], example["choices"], counterfactual=counterfactual, add_task_prefix=args.add_task_prefix)
                else:
                    input_seq = format_input(example["question"], counterfactual=counterfactual, add_task_prefix=args.add_task_prefix)
            else:
                input_seq = format_input(example["statement"], counterfactual=counterfactual, add_task_prefix=args.add_task_prefix)

            inputs = tokenizer(input_seq, padding='max_length', max_length=args.max_enc_length, truncation=True)

            if isinstance(example["explanation"], list):
                for explanation in example["explanation"][:5]:
                    output_seq, output_seq_without_answer = format_output(explanation, example["answer"], counterfactual=counterfactual, without_explanation=args.without_explanation, add_task_prefix=args.add_task_prefix)

                    encoder_input_tensor.append(inputs['input_ids'])
                    encoder_attention_mask_tensor.append(inputs['attention_mask'])

                    if counterfactual:
                        decoder_input_ids, decoder_label = get_label_tensor_answer_only(output_seq, output_seq_without_answer, tokenizer, args)
                        decoder_input_ids_tensor.append(decoder_input_ids)
                        decoder_label_tensor.append(decoder_label)
                    else:
                        decoder_label_tensor.append(get_label_tensor(output_seq, tokenizer, args))

            else:
                output_seq, output_seq_without_answer = format_output(example["explanation"], example["answer"], counterfactual=counterfactual, without_explanation=args.without_explanation, add_task_prefix=args.add_task_prefix)

                encoder_input_tensor.append(inputs['input_ids'])
                encoder_attention_mask_tensor.append(inputs['attention_mask'])

                if counterfactual:
                    decoder_input_ids, decoder_label = get_label_tensor_answer_only(output_seq, output_seq_without_answer, tokenizer, args)
                    decoder_input_ids_tensor.append(decoder_input_ids)
                    decoder_label_tensor.append(decoder_label)
                else:
                    decoder_label_tensor.append(get_label_tensor(output_seq, tokenizer, args))

    encoder_input_tensor = torch.tensor(encoder_input_tensor, dtype=torch.long)
    encoder_attention_mask_tensor= torch.tensor(encoder_attention_mask_tensor, dtype=torch.long)
    decoder_label_tensor = torch.tensor(decoder_label_tensor, dtype=torch.long)
    if counterfactual:
        decoder_input_ids_tensor = torch.tensor(decoder_input_ids_tensor, dtype=torch.long) 
    for f1, f2, f3 in zip(encoder_input_tensor[:2], encoder_attention_mask_tensor[:2], decoder_label_tensor[:2]):
        print("*** Example ***")
        print("encoder input: %s" % tokenizer.decode(f1))
        print("encoder attention mask: %s" % f2)
        print("decoder output: %s" % tokenizer.decode([tid for tid in f3 if not tid == -100]))
    if counterfactual:
        for f4 in decoder_input_ids_tensor[:2]:
            print("decoder input: %s" % tokenizer.decode(f4))

        return TensorDataset(encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor, decoder_input_ids_tensor)
    else:
        return TensorDataset(encoder_input_tensor, encoder_attention_mask_tensor, decoder_label_tensor)

