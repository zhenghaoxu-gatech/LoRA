export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir_col="./cola_col_64"
export output_dir_svd="./cola_svd_64"
export output_dir="./cola_64"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 64 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--learning_rate 1.3e-4 \
--num_train_epochs 10 \
--output_dir $output_dir_svd/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir_svd/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 20 \
--save_strategy steps \
--save_steps 20 \
--warmup_steps 100 \
--cls_dropout 0.1 \
--apply_lora \
--column_init svd \
--lora_r 16 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0 \
--use_deterministic_algorithms

