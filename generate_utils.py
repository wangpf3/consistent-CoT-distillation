import json
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import set_seed, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor

from utils import get_logger

def generation(inputs, model, tokenizer, args):

    with torch.no_grad():
        pred_ids = model.generate(
              input_ids=inputs.input_ids, 
              attention_mask=inputs.attention_mask,
              max_length=args.max_dec_length,
              decoder_start_token_id=model.config.decoder_start_token_id,
              eos_token_id=tokenizer.eos_token_id, 
              pad_token_id=tokenizer.pad_token_id,
              early_stopping=True, 
              num_return_sequences=1, #args.num_return_sequences,
              num_beams=args.num_beams,
              do_sample=args.sample,
              top_p=args.top_p,
              top_k=args.top_k,
              use_cache=True
             ) 

    batch_output = [tokenizer.decode(beam, skip_special_tokens=True) for beam in pred_ids]

    return batch_output

def generation_with_prefix(inputs, decoder_input_ids, model, tokenizer, args):

    input_length = len(decoder_input_ids[0])
    with torch.no_grad():
        pred_ids = model.generate(
              input_ids=inputs.input_ids, 
              attention_mask=inputs.attention_mask,
              max_length=args.max_dec_length,
              decoder_start_token_id=model.config.decoder_start_token_id,
              decoder_input_ids=decoder_input_ids,
              eos_token_id=tokenizer.eos_token_id, 
              pad_token_id=tokenizer.pad_token_id,
              early_stopping=True, 
              num_return_sequences=1, #args.num_return_sequences,
              num_beams=args.num_beams,
              do_sample=args.sample,
              top_p=args.top_p,
              top_k=args.top_k,
              use_cache=True
             ) 

    batch_output = [tokenizer.decode(beam[input_length:], skip_special_tokens=True) for beam in pred_ids]

    return batch_output


