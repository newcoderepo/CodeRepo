#!/bin/sh

for a in 1.0
	do
	
	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --class_num 10 \
				   --client_num 5 \
				   --loss CE \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedAvg \
				   --seed 0 
				   
	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --class_num 10 \
				   --client_num 5 \
				   --loss CE_Prox \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedProx \
				   --seed 0 \
				   --beta 0.001
				   
	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --class_num 10 \
				   --client_num 5 \
				   --loss CE_LC \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedLC \
				   --seed 0 \
				   --beta 1		   

	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --class_num 10 \
				   --client_num 5 \
				   --loss NT_CE \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedNTD \
				   --seed 0 

	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
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
				   --layer 0 \
				   --fea_percent 0.1 \
				   --seed 0

	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
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
				   --layer -1 \
				   --fea_percent 0.1 \
				   --seed 0
				   
				   
	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --client_num 5 \
				   --class_num 10 \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedFea \
				   --loss CE_CE_KL_Prox \
				   --layer 0 \
				   --fea_percent 0.1 \
				   --seed 0

	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --client_num 5 \
				   --class_num 10 \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedFea \
				   --loss CE_CE_KL_LC \
				   --layer 0 \
				   --fea_percent 0.1 \
				   --seed 0
				   
	python -u main.py --dataset UrbanSound \
				   --split_method distribution \
				   --split_para $a \
				   --split_num 20 \
				   --client_num 5 \
				   --class_num 10 \
				   --local_epochs 10 \
				   --batch_size 64 \
				   --num_global_iters 100 \
				   --personal_learning_rate 0.001 \
				   --modelname AudioNet \
				   --algorithm FedFea \
				   --loss CE_CE_NT \
				   --layer 0 \
				   --fea_percent 0.1 \
				   --seed 0						   
				   

done				   