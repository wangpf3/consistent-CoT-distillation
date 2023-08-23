#!/bin/bash

dataset="strategyQA"
prompt="explanation"

output_prefix="outputs/${dataset}/"
mkdir -p $output_prefix

python -u \
    contrastive_decoding_rationalization.py \
    --output_prefix $output_prefix \
    --dataset $dataset \
    --prompt $prompt \
    >${output_prefix}/rationalization.log 2>&1 &
