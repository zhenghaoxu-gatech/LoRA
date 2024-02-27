export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./mnli_128_1"
export output_dir_col="./mnli_col_128_1"
export output_dir_svd="./mnli_svd_128_1"
export seed=1

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--learning_rate 1e-4 \
--num_train_epochs 5 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--warmup_steps 1000 \
--cls_dropout 0.15 \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed $seed \
--weight_decay 0 \
--use_deterministic_algorithms

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--learning_rate 1e-4 \
--num_train_epochs 5 \
--output_dir $output_dir_col/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir_col/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--warmup_steps 1000 \
--cls_dropout 0.15 \
--apply_lora \
--column_init column \
--lora_r 8 \
--lora_alpha 16 \
--seed $seed \
--weight_decay 0 \
--use_deterministic_algorithms

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--learning_rate 1e-4 \
--num_train_epochs 5 \
--output_dir $output_dir_svd/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir_svd/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--warmup_steps 1000 \
--cls_dropout 0.15 \
--apply_lora \
--column_init svd \
--lora_r 8 \
--lora_alpha 16 \
--seed $seed \
--weight_decay 0 \
--use_deterministic_algorithms
