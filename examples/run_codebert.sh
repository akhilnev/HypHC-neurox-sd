#!/bin/bash
python train.py --dataset codebert \
                --epochs 50 \
                --batch_size 256 \
                --learning_rate 1e-3 \
                --temperature 1e-1 \
                --eval_every 1 \
                --patience 30 \
                --optimizer RAdam \
                --anneal_every 50 \
                --anneal_factor 0.5 \
                --init_size 5e-2 \
                --num_samples 100000 \
                --seed 0
