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
WORK_ROOT=/root/paddlejob/workspace/env_run/output/zhangwl
# ps aux | grep "/root/paddlejob/workspace/env_run/output/*" | awk '{print $2}' | xargs kill
# export PYTHONPATH=/root/paddlejob/workspace/env_run/output/zhangwl/Paddle_origin/PaddleNLP:$PYTHONPATH
source ${WORK_ROOT}/py39_zwl/bin/activate
export PYTHONPATH=${WORK_ROOT}/Paddle_origin/PaddleNLP/:$PYTHONPATH
# export FLAGS_cudnn_deterministic=0
# export FLAGS_embedding_deterministic=0
# unset CUDA_VISIBLE_DEVICES
# export PARALLEL_CROSS_ENTROPY=true
# export CUDA_DEVICE_MAX_CONNECTIONS=1 # 必须要开，开了后性能提高4%

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
unset PADDLE_TRAINERS
unset PADDLE_NUM_GRADIENT_SERVERS

unset PADDLE_WORKERS_IP_PORT_LIST

export NNODES=1
export PADDLE_TRAINERS_NUM=1

# export CUDA_DEVICE_MAX_CONNECTIONS=1
WORLD_SIZE=8
GBS=32
MBS=1
MP=4
SP=1  # 0 or 1
PP=2
VPP=1
SD=$(($WORLD_SIZE / ($MP * $PP)))
ACC_STEPS=$(($GBS / ($SD * $MBS)))

task_name="llama70b_hand_MP${MP}_PP${PP}_SP${SP}"
log_dir="log/$task_name"
output_dir="output/$task_name"
rm -rf $log_dir
rm -rf $output_dir

python -u -m paddle.distributed.launch \
    --gpus=0,1,2,3,4,5,6,7 \
    --log_dir ${log_dir} \
    ${WORK_ROOT}/Paddle_origin/PaddleNLP/llm/run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-70b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-70b" \
    --skip_memory_metrics 0 \
    --output_dir "output/$task_name" \
    --input_dir "${WORK_ROOT}/Paddle_origin/PaddleNLP/llm/auto_parallel/llama_data" \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --per_device_eval_batch_size 2 \
    --hybrid_parallel_topo_order "sharding_first" \
    --tensor_parallel_degree ${MP} \
    --pipeline_parallel_degree ${PP} \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv" \
    --virtual_pp_degree ${VPP} \
    --sequence_parallel ${SP} \
    --bf16 true \
    --fp16_opt_level "O2" \
    --amp_master_grad true \
    --max_grad_norm 0.0 \
    --use_flash_attention true \
    --use_fused_rms_norm false \
    --use_fast_layer_norm false \
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --use_fused_rope true \
    --enable_linear_fused_grad_add true \
    --max_seq_length 4096 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --max_steps 20 \
    --save_steps 5000 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute false \
    --recompute_use_reentrant true \
    --distributed_dataloader 0 \
    --recompute_granularity "full" \
    --save_total_limit 2 \
    --device "gpu" \
    --skip_memory_metrics 0 \
    --sharding "stage1" \
    --sharding_parallel_config "split_param enable_stage1_allgather_overlap" \
    --num_hidden_layers 20 \
    # --master=10.54.84.211:9090 \
    # --nnodes 4 \
    # --nproc_per_node 8 \