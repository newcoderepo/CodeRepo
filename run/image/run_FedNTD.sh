#!/bin/sh
python -u main.py --dataset Cifar10 \
               --split_method distribution \
			   --split_para 0.1 \
			   --split_num 100 \
			   --client_num 10 \
			   --class_num 10 \
			   --local_epochs 10 \
			   --batch_size 64 \
			   --num_global_iters 100 \
			   --personal_learning_rate 0.001 \
			   --modelname MOBNET \
			   --algorithm FedNTD \
               --loss NT_CE 
    
  
