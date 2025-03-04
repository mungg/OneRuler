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

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    8000
    # 32000
    # 64000
    # 128000
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    case "$MODEL_NAME" in
        Llama3)
            MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="$MODEL_PATH"
            TOKENIZER_TYPE="hf"
            ;;
        Qwen2.5)
            MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="$MODEL_PATH"
            TOKENIZER_TYPE="hf"
            ;;
        deepseek-r1)
            MODEL_PATH="deepseek-ai/DeepSeek-R1"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH="$MODEL_PATH"
            TOKENIZER_TYPE="hf"
            ;;
        gemini-1.5-flash)
            MODEL_PATH="gemini-1.5-flash"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH="$MODEL_PATH"
            TOKENIZER_TYPE="gemini"
            ;;
        o3-mini)
            MODEL_PATH="o3-mini"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="o200k_base"
            TOKENIZER_TYPE="openai"
            ;;
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
