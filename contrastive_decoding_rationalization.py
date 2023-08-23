import json
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm

torch.set_num_threads(4)

# for REPRODUCIBILITY
set_seed(42)

# ----------------------------------------------------- #
# hyper-parameters
num_return_sequences = 1
generation_length = 128

# ----------------------------------------------------- #

def contrastive_decoding(input_seq1, input_seq2, model, tokenizer, indicator_token_ids, args):
    inputs1 = tokenizer(input_seq1, truncation=True, return_tensors='pt').to(args.device)
    input_length1 = len(inputs1.input_ids[0])
    generated1 = inputs1.input_ids
    past_key_values1 = None

    inputs2 = tokenizer(input_seq2, truncation=True, return_tensors='pt').to(args.device)
    input_length2 = len(inputs2.input_ids[0])
    generated2 = inputs2.input_ids
    past_key_values2 = None

    with torch.no_grad():
        for step in range(generation_length):
            # get probs given by the original teacher
            attention_mask1 = generated1.new_ones(generated1.shape)
            outputs1 = model(
              input_ids=generated1 if past_key_values1 is None else generated1[:, -1:],
              past_key_values=past_key_values1,
              attention_mask=attention_mask1,
            )
            logits1 = outputs1.logits[:, -1, :]
            past_key_values1 = outputs1.past_key_values
            prob1 = F.log_softmax(logits1 / args.temperature, dim=-1)

            candidate_next_token = prob1.argmax(dim=-1, keepdim=True)
            if candidate_next_token[0].item() == indicator_token_ids["stop"]:
                break

            # get probs given by the hallucinating teacher
            attention_mask2 = generated2.new_ones(generated2.shape)
            outputs2 = model(
              input_ids=generated2 if past_key_values2 is None else generated2[:, -1:],
              past_key_values=past_key_values2,
              attention_mask=attention_mask2,
            )
            logits2 = outputs2.logits[:, -1, :]
            past_key_values2 = outputs2.past_key_values
            prob2 = F.log_softmax(logits2, dim=-1)

            # contrastive decoding
            debiased_prob = prob1 - args.interpolation * prob2
            next_token = debiased_prob.argmax(dim=-1, keepdim=True)

            if next_token[0] == indicator_token_ids["stop"]:
                break

            generated1 = torch.cat((generated1, next_token), dim=1)
            generated2 = torch.cat((generated2, next_token), dim=1)

    generation = tokenizer.decode(generated1[0][input_length1:], skip_special_tokens=True)
    return generation

def main(args):
    # ----------------------------------------------------- #
    # load LM
    model_path = 'EleutherAI/gpt-neox-20b'
    model_name = "GPT-neox"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='../cache') #, use_fast=False)
    n_gpus = 1
    free_in_GB = 49
    max_memory = {i: "{}GB".format(free_in_GB) for i in range(args.gpu, args.gpu + n_gpus)}

    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
                model_path, 
                device_map='auto',
                max_memory = max_memory,
                cache_dir='../cache',
                torch_dtype='auto'
            )

    indicator_token_ids = {
            "stop": tokenizer.encode("\n\nQ")[-2],
    }

    model.eval()

    # ----------------------------------------------------- #
    # prepare data
    with open('./prompts/{}.{}.txt'.format(args.dataset, args.prompt), 'r') as fr:
        prompt = json.load(fr)["prompt"]

    print(prompt)
    prompt_without_question = '\n\n'.join(prompt.split('\n\n')[:-1])+'\n\n'

    for split in args.eval_split.split(','):
        with open('./data/{}/{}.jsonl'.format(args.dataset, split), 'r') as fr:
            examples = [json.loads(line) for line in fr.readlines()]

        # ----------------------------------------------------- #
        # inference

        output_path = os.path.join(args.output_prefix, '{}.jsonl'.format(split))

        fw = open(output_path, 'w', buffering=1)
        for example in tqdm(examples):
            if "context" in example:
                formatted_question = example["context"] 
                choices = ["false", "true"]
                if "counterfactual" in split:
                    answer = "false" if example["answer"] == 1 else "true"
                    wrong_answer = "false" if example["answer"] == 0 else "true"
                else:
                    answer = "false" if example["answer"] == 0 else "true"
                    wrong_answer = "false" if example["answer"] == 1 else "true"
                question = example["context"]
            else:
                formatted_question = example["question"] 
                choices = example["choices"] if "choices" in example else ["no", "yes"]
                question = example["question"]
                if "choices" in example and len(example["choices"]) > 2:
                    if "counterfactual" in split:
                        answer = random.choice(example["choices"][:example["answer"]] + example["choices"][example["answer"]+1:])
                        wrong_answer = example["choices"][example["answer"]]
                    else:
                        answer = example["choices"][example["answer"]]
                        wrong_answer = random.choice(example["choices"][:example["answer"]] + example["choices"][example["answer"]+1:])
                else:
                    if "counterfactual" in split:
                        answer = "yes" if example["answer"] == 0 else "no"
                        wrong_answer = "yes" if example["answer"] == 1 else "no"
                    else:
                        answer = "yes" if example["answer"] == 1 else "no"
                        wrong_answer = "yes" if example["answer"] == 0 else "no"

            if "choices" in example and len(example["choices"]) > 2:
                choices_seq = ""
                formatted_question += "\nAnswer Choices:"
                for choice_id, choice in enumerate(example["choices"]):
                    formatted_question += "\n({}) {}".format(chr(ord('a')+choice_id), choice)
                    choices_seq += " ({}) {}".format(chr(ord('A')+choice_id), choice)

            input_seq1 = prompt.format(formatted_question, answer)
            input_seq2 = prompt.format(formatted_question, wrong_answer) # replace wrong_answer with "" if using empty string as the perturbed answer
            if args.debug:
                print(input_seq2)
                print(input_seq3)
            generation = contrastive_decoding(input_seq1, input_seq2, model, tokenizer, indicator_token_ids, args)

            if "context" in example:
                fw.write(json.dumps({"id": example["id"], "answer": answer, "statement": question, "explanation": generation_list}) + "\n")
            else:
                if "choices" in example and len(example["choices"]) > 2:
                    fw.write(json.dumps({"id": example["id"], "answer": answer, "question": question, "choices": choices_seq.strip(), "explanation": generation.strip()}) + "\n")
                else:
                    fw.write(json.dumps({"id": example["id"], "answer": answer, "question": question, "explanation": generation.strip()}) + "\n")
        fw.close()

    # ----------------------------------------------------- #

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--output_prefix', '-o', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--eval_split', type=str, default='test,dev,train,train.counterfactual')
    parser.add_argument("--debug", action='store_true')

    # debiased factor
    parser.add_argument('--interpolation', type=float, default=0.5)

    # decoding strategy
    parser.add_argument('--temperature', type=float, default=1.0)

    # gpu and workers option
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    main(args)

