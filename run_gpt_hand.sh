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

# unset PADDLE_ELASTIC_JOB_ID
# unset PADDLE_TRAINER_ENDPOINTS
# unset DISTRIBUTED_TRAINER_ENDPOINTS
# unset FLAGS_START_PORT
# unset PADDLE_ELASTIC_TIMEOUT
# unset CUDA_VISIBLE_DEVICES
# export NNODES=1
# export PADDLE_TRAINERS_NUM=1
export GLOG_v=0
export FLAGS_set_to_1d=0;
# export CUDA_MODULE_LOADING=LAZY
# 模型环境变量
# model_dir_list: benchmark/frame_benchmark/paddle/PaddleNLP/tests/test_tipc/static/auto_parallel/gpt3/N4C32/meta-llama-Llama-2-13b_pretrain_dy2st_bs32_bf16_DP1_MP1_PP4_1F1B_Sharding4_Stage1.sh
# docker: iregistry.baidu-int.com/tiangexiao/base-images:paddlecloud-ubuntu20.04-gcc12.2-cuda12.3-cudnn9.0-nccl2.21.5-openmpi4.1.5-FleetY10.0-release
# 拉取日志解析工具
wget https://paddle-qa.bj.bcebos.com/benchmark/tools.tar.gz && tar xvf tools.tar.gz && export BENCHMARK_ROOT=$PWD/tools/;
# git clone http://github.com/PaddlePaddle/PaddleNLP.git -b develop && cd PaddleNLP && git checkout -b dc0ca03b525028b47b54fe80bcf24f501574bc76 dc0ca03b525028b47b54fe80bcf24f501574bc76;
export PROFILING_TIMER_ONLY=no;

# pip install -U https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc122-Cuda123-Cudnn90-Trt86-Py310-CINN-Compile/b67c88f952163f44b273da6adba4393a480b20d8/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl; 

python -m pip install -U setuptools==58.0.4   # 60版本会报AttributeError: module distutils has no attribute version; 
python -m pip install setuptools-scm==6.4.2   # 7.0版本下安装jiaba报错导致NLP不能安装;
python -m pip install -U pillow;        # 升级到最高版本,否则NLP安装失败

cd tests/;
bash test_tipc/dygraph/hybrid_parallelism/gpt3/N4C32/gpt3-13b_pretrain_bs128_bf16_DP1_MP2_PP4_VPP1_Sharding4_Stage1.sh;
