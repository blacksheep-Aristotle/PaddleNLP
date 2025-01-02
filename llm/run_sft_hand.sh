# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# just for debug

set -x

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export NNODES=1
export PADDLE_TRAINERS_NUM=1

export GLOG_v=0

export FLAGS_cudnn_deterministic=0
export FLAGS_embedding_deterministic=0
# export FLAGS_max_inplace_grad_add=65536
export FLAGS_enable_auto_parallel_align_mode=0

task_name="llama_3.1_sft"
rm -rf output/$task_name/
rm -rf "log/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../:$PYTHONPATH

#ulimit -c unlimited

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  "log/$task_name""_log" \
    run_finetune.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_name_or_path "fintune_data/data" \
    --output_dir "./checkpoints/llama_sft_ckpts" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 3e-05 \
    --max_steps 10 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --src_length 1024 \
    --max_length 2048 \
    --bf16 true \
    --fp16_opt_level "O2" \
    --amp_master_grad true \
    --do_train true \
    --do_eval false \
    --disable_tqdm true \
    --load_best_model_at_end true \
    --eval_with_do_generation false \
    --metric_for_best_model "accuracy" \
    --recompute false \
    --save_total_limit 1 \
    --tensor_parallel_degree 2 \
<<<<<<< HEAD:llm/run_sft_hand.sh
    --pipeline_parallel_degree 2 \
=======
    --pipeline_parallel_degree 1 \
>>>>>>> update auto_lora_model:llm/auto_parallel/llama/llama_finetune_with_api.sh
    --sharding "stage1" \
    --zero_padding false \
    --unified_checkpoint false \
    --flash_mask false \
    --use_flash_attention true \
    --sharding "stage1" \
    --auto_parallel_resume_form_hybrid_parallel true \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --pipeline_parallel_config "enable_sharding_comm_overlap enable_dp_comm_overlap enable_overlap_p2p_comm disable_p2p_cache_shape" \
    # --num_hidden_layers 4 \
