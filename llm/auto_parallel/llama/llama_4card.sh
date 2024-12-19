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
set -e

COMPANY_NAME="meta-llama"
MODEL_TYPE="Llama-2-70b"

WORK_ROOT=/root/paddlejob/workspace/env_run/output/zhangwl
source ${WORK_ROOT}/py310_zwl/bin/activate
export PYTHONPATH=${WORK_ROOT}/PaddleNLP/:$PYTHONPATH
export FLAGS_enable_pir_api=1

# export FLAGS_cudnn_deterministic=1
# export FLAGS_embedding_deterministic=1
# unset CUDA_VISIBLE_DEVICES
# export PARALLEL_CROSS_ENTROPY=true
# export CUDA_DEVICE_MAX_CONNECTIONS=1 # 必须要开，开了后性能提高4%

# unset PADDLE_ELASTIC_JOB_ID
# unset PADDLE_TRAINER_ENDPOINTS
# unset DISTRIBUTED_TRAINER_ENDPOINTS
# unset FLAGS_START_PORT
# unset PADDLE_ELASTIC_TIMEOUT

# unset PADDLE_TRAINERS_NUM
# unset PADDLE_TRAINER_ID
# unset PADDLE_TRAINERS
# unset PADDLE_NUM_GRADIENT_SERVERS

# unset PADDLE_WORKERS_IP_PORT_LIST

export NNODES=1
export PADDLE_TRAINERS_NUM=1
export FLAGS_log_memory_stats=1
export GLOG_v=1
# export CUDA_MODULE_LOADING=LAZY
# export CUDA_DEVICE_MAX_CONNECTIONS=1
WORLD_SIZE=32
GBS=32
MBS=1
MP=2
SP=1  # 0 or 1
PP=8
VPP=5
SD=$(($WORLD_SIZE / ($MP * $PP)))
ACC_STEPS=$(($GBS / ($SD * $MBS)))

task_name="llama70b_auto_MP${MP}_PP${PP}_SP${SP}_GLOG${GLOG_v}"
log_dir="log/$task_name"
output_dir="output/$task_name"
rm -rf $log_dir
rm -rf $output_dir


python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  ${log_dir} \
    ${WORK_ROOT}/PaddleNLP/llm/auto_parallel/llama//run_pretrain_auto.py \
    --model_name_or_path "meta-llama/Llama-2-70b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-70b" \
    --model_type "llama" \
    --output_dir "output/$task_name" \
    --input_dir "${WORK_ROOT}/PaddleNLP/llm/auto_parallel/llama_data" \
    --split 949,50,1 \
    --to_static true \
    --tensor_parallel_degree ${MP} \
    --pipeline_parallel_degree ${PP} \
    --virtual_pp_degree ${VPP} \
    --pipeline_schedule_mode "VPP" \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 0.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_steps 10\
    --warmup_steps 30 \
    --logging_steps 2 \
    --eval_steps 10000 \
    --save_steps 1000 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --save_total_limit 2 \
    --device gpu \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --enable_auto_parallel 1 \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${ACC_STEPS}\
    --per_device_eval_batch_size 1 \
    --recompute false \
    --recompute_use_reentrant true \
    --recompute_granularity full \
    --pp_recompute_interval 0 \
    --bf16 true \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --use_flash_attention true \
    --fused_linear_param_grad_add true \
    --use_fused_rope true \
    --use_fused_rms_norm false \
    --max_seq_length 4096 \
    --sequence_parallel ${SP} \
    --sharding "stage1" \
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate " \
    --sharding_parallel_config "enable_stage1_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce replace_with_parallel_cross_entropy" \
    --pipeline_parallel_config "enable_send_recv_overlap enable_split_backward" \
    --auto_parallel_resume_form_hybrid_parallel true \
    # --master=10.54.84.211:9090 \
    # --nnodes 4 \
    # --nproc_per_node 8 \