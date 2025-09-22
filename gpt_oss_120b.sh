apt update && apt upgrade -y
apt-get install vim -y
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# virtual env
mkdir workspace/gpt_oss
cd workspace/gpt_oss
uv init --python 3.12
uv venv
source .venv/bin/activate

uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match


# H100
# Required env
export VLLM_USE_TRTLLM_ATTENTION=1

# Choose one of the following
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

export CUDA_HOME=path/to/cuda_home # Must be the same path as used during installation

# We are ready to serve:



# エラーが出たninjaがない（以下GPT）
# 1) ninja を入れる
apt-get update && apt-get install -y ninja-build
ln -sf /usr/bin/ninja /usr/local/bin/ninja

# 2) CUDA Toolkit (nvcc) を入れる
#   * その環境のドライバと互換のCUDAを選ぶ（例: 12.x）
#   * 既存イメージにインストールするか、最初から nvidia/cuda:*-devel ベースの
#     vLLM イメージを使うのが安定
#   ここは環境によりコマンドが異なるため、RunPodのテンプレやベースOSに従って導入

# 3) 環境変数を設定
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

# （任意・ビルド時間短縮）GPUに合わせてアーキを固定
# H100 → 9.0 / A100 → 8.0 など
export TORCH_CUDA_ARCH_LIST="9.0"

# 4) vLLMを再起動
vllm serve openai/gpt-oss-120b ...


# エラー
以下のエラーが出た。同様に対処方法を教えて　そのseq lengthだとkvキャッシュのメモリー不足だよ。
ValueError: To serve at least one request with the models's max seq len (131072), (4.79 GiB KV cache is needed, which is larger than the available KV cache memory (4.06 GiB). Based on the available memory, the estimated maximum model length is 109776. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.

# kvキャッシュに合わせてseq lengthを少し減らす
vllm serve openai/gpt-oss-120b --max-model-len 100000 --host 0.0.0.0 --port 8000

