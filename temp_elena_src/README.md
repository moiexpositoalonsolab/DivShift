Models to benchmark divshift dataset

On Carnegie BSE cluster, run like:
srun --mem 1200000 --time 168:00:00 --ntasks 1 --cpus-per-task 7 --partition bse --gres=gpu:tesla-a100:1 python3 ./DivShift/temp_elena_src/supervised_train.py --device 0 --data_dir ./west_coast_dataset --checkpoint_freq 5 --exp_id Year_Supervised_10epochs --num_epochs 10 --batch_size 64 --test_batch_size 64 --learning_rate 0.064 --processes 7 --train_type full_finetune --train_split 2019-2021 --test_split 2022 --to_classify name