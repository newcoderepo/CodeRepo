#!/bin/sh
for l in -1 0 1 2 3
	do
	python -u main.py --dataset UrbanSound \
				   --split_method quantity \
				   --split_para 3 \
				   --split_num 20 \
				   --client_num 5 \
				   --class_num 10 \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedFea \
				   --loss CE_CE_KL \
				   --layer $l \
				   --fea_percent 0.05 \
				   --seed 0 
done


