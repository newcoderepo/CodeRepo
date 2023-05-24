#!/bin/sh
python -u main.py --dataset UrbanSound \
               --split_method quantity \
			   --split_para 3 \
			   --split_num 20 \
			   --class_num 10 \
			   --client_num 5 \
			   --loss CE \
			   --local_epochs 20 \
			   --batch_size 64 \
			   --num_global_iters 1 \
			   --personal_learning_rate 0.0005 \
			   --modelname AudioNet \
			   --algorithm CCVR \
			   --layer -1 \
			   --fea_percent 0.1 
