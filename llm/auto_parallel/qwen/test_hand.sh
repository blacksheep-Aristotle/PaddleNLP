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
export FLAGS_call_stack_level=3
export FLAGS_use_cuda_managed_memory=false

task_name="llama_auto"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH


rm -rf ./log/auto_3d_hand

export FLAGS_embedding_deterministic=0        
export FLAGS_cudnn_deterministic=0
export FLAGS_max_inplace_grad_add=65536
export FLAGS_enable_auto_parallel_align_mode=0
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_enable_pir_in_executor=1


python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "log/auto_3d_hand" \
    ../../run_pretrain.py \
    --model_name_or_path "qwen/qwen-14b" \
    --tokenizer_name_or_path "qwen/qwen-14b" \
    --input_dir "../data" \
    --output_dir "./checkpoints/qwen_pretrain_ckpts" \
    --per_device_train_batch_size 1\
    --gradient_accumulation_steps 32\
    --per_device_eval_batch_size 16\
    --sharding "stage1" \
    --sharding_parallel_degree 1\
    --tensor_parallel_degree 2\
    --pipeline_parallel_degree 4\
    --virtual_pp_degree 5\
    --sequence_parallel 0\
    --use_flash_attention true\
    --use_fused_rms_norm false\
    --use_fused_rope true\
    --max_seq_length 4096\
    --learning_rate 3e-05\
    --min_learning_rate 3e-06\
    --scale_loss 1024\
    --warmup_steps 30\
    --logging_steps 1\
    --max_steps 10000\
    --save_steps 1000\
    --eval_steps 10000\
    --weight_decay 0.01\
    --bf16 true\
    --fp16_opt_level "O2"\
    --amp_master_grad true \
    --warmup_ratio 0.01\
    --max_grad_norm 0.0\
    --dataloader_num_workers 4\
    --continue_training 0\
    --do_train true\
    --do_eval false\
    --do_predict false\
    --disable_tqdm true\
    --recompute false\
    --recompute_granularity "core_attn"\
    --recompute_use_reentrant true\
    --distributed_dataloader 0\
    --save_total_limit 2\
    --enable_auto_parallel 0\
    --to_static 0 \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --pipeline_parallel_config "enable_sharding_comm_overlap enable_dp_comm_overlap enable_overlap_p2p_comm" \