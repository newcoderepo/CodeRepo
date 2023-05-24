#!/bin/sh
python -u main.py --dataset UrbanSound \
               --split_method quantity \
			   --split_para 3 \
			   --split_num 20 \
			   --class_num 10 \
			   --client_num 5 \
			   --loss CE \
			   --local_epochs 10 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname MOBNET \
			   --algorithm FedAvg \
			   --seed 0
