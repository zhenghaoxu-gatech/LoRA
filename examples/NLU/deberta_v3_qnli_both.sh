export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir_col="./qnli_col_32"
export output_dir="./qnli_32"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name qnli \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-4 \
--num_train_epochs 8 \
--output_dir $output_dir_col/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir_col/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--warmup_steps 500 \
--cls_dropout 0.1 \
--apply_lora \
--column_init \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--use_deterministic_algorithms

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name qnli \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-4 \
--num_train_epochs 8 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--warmup_steps 500 \
--cls_dropout 0.1 \
--apply_lora \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--use_deterministic_algorithms
