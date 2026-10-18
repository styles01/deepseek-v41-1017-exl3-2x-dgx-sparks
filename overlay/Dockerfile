# The 1017 overlay — layers on top of the release recipe image (scripts/image/build_serve_image.sh).
# Build on a Spark (arm64, native CUDA):
#   cd dgx-spark-deepseek-v41/overlay && docker build -t ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks:v1 .
# (or start FROM the published release image of the base recipe once it is on ghcr).
FROM dsv41-serve:v4
COPY compressor.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/compressor.py
COPY config.py /usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/config.py
COPY attention.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/attention.py
COPY dense.py /usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/dense.py
COPY dspark_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py
COPY dspark.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py
RUN python3 -m py_compile \
  /usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/config.py \
  /usr/local/lib/python3.12/dist-packages/vllm_exl3_moe/dense.py \
  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/compressor.py \
  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/attention.py \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py