#!/bin/bash

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
unset http_proxy && unset https_proxy && unset no_proxy
ps aux | grep "/root/paddlejob/workspace/env_run/output/zhangwl/*" |  awk '{print $2}' | xargs kill
WORK_ROOT=/root/paddlejob/workspace/env_run/output/zhangwl

export PYTHONPATH=$PYTHONPATH:/root/paddlejob/workspace/env_run/output/zhangwl/Paddle_origin/PaddleNLP
# export PATH=${PYTHONPATH}/bin:${PATH}
export GLOG_v=0

WORLD_SIZE=32
GBS=128
MBS=1
MP=4
SP=0  # 0 or 1
PP=4
VPP=5
SD=$(($WORLD_SIZE / ($MP * $PP)))
ACC_STEPS=$(($GBS / ($SD * $MBS)))

COMPANY_NAME="meta-llama"
 
MODEL_TYPE="Llama-2-70b"
OUTPUT_FILENAME=paddle_${MODEL_TYPE}.gbs${GBS}_mp${MP}pp${PP}sd${SD}_mbs${MBS}_acc${ACC_STEPS}.20231031

autoconfig_args=

if [ "$autoconfig_args" = "" ]; then
  if [ "$MP" != "1" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
  fi
fi
 
rm -rf log_${MODEL_TYPE}
rm -rf output
 
if [ "$WORLD_SIZE" = "8" ]; then
  unset PADDLE_TRAINER_ENDPOINTS
  unset DISTRIBUTED_TRAINER_ENDPOINTS
  export PADDLE_NNODES=1
  distributed_args="--nnodes 1 --nproc_per_node 8"
else
  # unset PADDLE_ELASTIC_JOB_ID
  # unset PADDLE_TRAINER_ENDPOINTS
  # unset DISTRIBUTED_TRAINER_ENDPOINTS
  # unset FLAGS_START_PORT
  # unset PADDLE_ELASTIC_TIMEOUT
 
  # unset PADDLE_ELASTIC_JOB_ID
  # unset PADDLE_TRAINER_ENDPOINTS
  # unset DISTRIBUTED_TRAINER_ENDPOINTS
  # unset FLAGS_START_PORT
  # unset PADDLE_ELASTIC_TIMEOUT
  # unset PADDLE_TRAINERS_NUM
  # unset PADDLE_TRAINER_ID
  # unset PADDLE_WORKERS_IP_PORT_LIST
  # unset PADDLE_TRAINERS
  # unset PADDLE_NUM_GRADIENT_SERVERS
 
  if [ "$autoconfig_args" != "" ]; then
    distributed_args="--master=etcd://10.54.84.211:9090 --nnodes=4:4"
  else
    iplist="10.54.84.229,10.54.84.211,10.54.85.210,10.54.64.145"
    distributed_args="--nnodes 4 --nproc_per_node 8 --ips ${iplist} --run_mode=collective"
  fi
fi

python -u -m paddle.distributed.launch \
        --gpus "0,1,2,3,4,5,6,7" \
        --log_dir log_${MODEL_TYPE} \
        ${WORK_ROOT}/Paddle_origin/PaddleNLP/llm/run_pretrain.py \
        --model_name_or_path "${COMPANY_NAME}/${MODEL_TYPE}" \
        --tokenizer_name_or_path "${COMPANY_NAME}/${MODEL_TYPE}" \
        --input_dir "${WORK_ROOT}/Paddle_origin/PaddleNLP/llm/auto_parallel/llama_data" \
        --output_dir "output" \
        --split 949,50,1 \
        --max_seq_length 4096 \
        --per_device_train_batch_size ${MBS} \
        --gradient_accumulation_steps ${ACC_STEPS} \
        --per_device_eval_batch_size 4 \
        --bf16 true\
        --fp16_opt_level "O2"  \
        --amp_master_grad true \
        --tensor_parallel_degree ${MP} \
        --pipeline_parallel_degree ${PP} \
        --virtual_pp_degree ${VPP} \
        --sequence_parallel ${SP} \
        --learning_rate 0.00001 \
        --min_learning_rate 0.000001 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --do_train \
        --continue_training 0 \
        --max_steps 50 \
        --eval_steps 1000 \
        --save_steps 5000 \
        --logging_steps 1 \
        --dataloader_num_workers 4 \
        --recompute 0 \
        --recompute_use_reentrant true \
        --recompute_granularity "full" \
        --use_flash_attention true \
        --use_fused_rms_norm false \
        --fuse_attention_qkv true \
        --fuse_attention_ffn true \
        --use_fused_rope false \
        --enable_linear_fused_grad_add true \
        --sharding "stage1" \
        --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
        --pipeline_parallel_config "enable_sharding_comm_overlap disable_partial_send_recv" \
        --disable_tqdm true \
        --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
        # --model_type "llama" \