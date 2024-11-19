# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set -x

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export NNODES=1
export PADDLE_TRAINERS_NUM=1

export GLOG_v=0

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1
export FLAGS_max_inplace_grad_add=65536
export FLAGS_enable_auto_parallel_align_mode=1
export FLAGS_enable_pir_api=1

task_name="llama_auto"
rm -rf output
rm -rf log

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

#ulimit -c unlimited

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  "log" \
    ./run_pretrain_auto.py \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --to_static false \
    --pipeline_parallel_degree 2 \
    --sharding_parallel_degree 2 \
    --tensor_parallel_degree 2 \
    --virtual_pp_degree 1 \
    --pipeline_schedule_mode "VPP" \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_steps 10 \
    --logging_steps 1 \
    --eval_steps 10000 \
    --save_steps 1000 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --save_total_limit 2 \
    --device gpu \
    --model_type "llama_network" \
    --use_intermediate_api true \
    --dataloader_num_workers 4 \
    --distributed_dataloader 0 \
    --enable_auto_parallel 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \
    --recompute false \
    --recompute_use_reentrant true \
    --skip_profile_timer true \
    --recompute_granularity full \
    --pp_recompute_interval 0 \
    --bf16 true \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --fuse_attention_ffn false \
    --fuse_attention_qkv true \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm false \
    --max_seq_length 4096 \
    --sequence_parallel true \
    --sharding "stage1" \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --num_hidden_layers 4 \
    --auto_parallel_resume_form_hybrid_parallel true \
