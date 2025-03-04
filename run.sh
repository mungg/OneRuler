#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME LANGUAGE

export HF_HOME=''
export STANZA_RESOURCES_DIR=''

if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name> $2 <language>" 
    exit 1
fi

# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="dataset" # the path that stores generated task samples and model predictions.
MODEL_DIR="../.." # the path that contains individual model folders from HUggingface.
CONFIG_DIR='./config/'
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization


# Model and Tokenizer
source ${CONFIG_DIR}/config_models.sh

MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


# Benchmark and Tasks
source ${CONFIG_DIR}/config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

LANGUAGE=${3}

total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${LANGUAGE}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        python -u OneRuler/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            --lang ${LANGUAGE}\

            ${REMOVE_NEWLINE_TAB}
        
        start_time=$(date +%s)ㅌ
    done
done

echo "Total time spent on call_api: $total_time seconds"
