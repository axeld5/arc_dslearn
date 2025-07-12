# ──────────────────────────────────────────────────────────────────────────────
#  Dockerfile – ARC-DSL learner (H100, CUDA-12.4+, Flash-Attn, Deepspeed)
# ──────────────────────────────────────────────────────────────────────────────
#  • Base: NVIDIA PyTorch 24.05  ➜  Python 3.10, Torch 2.3.0, CUDA 12.4
#  • Extras: Flash-Attention 2.6, Triton 3.2, Deepspeed 0.14, bitsandbytes
#  • Meant to live at repo root next to requirements.txt
# ──────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/pytorch:24.05-py3

# -----------------------------------------------------------------------------
# 1.  System deps (for Flash-Attn builds & bnb compile fallback)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build cmake curl && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2.  Python deps – explicit pins known to play nicely on Hopper/H100
#     Flash-Attn wheel ≥2.6 has sm90 kernels; Triton is optional but speeds up
# -----------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="90"          \
    NCCL_P2P_LEVEL="NVL"               \
    NCCL_IB_DISABLE=1                  \
    PYTHONUNBUFFERED=1

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        flash-attn==2.6.3          \
        triton==3.2.0              \
        transformers==4.43.2       \
        datasets==2.18.0           \
        peft==0.11.1               \
        accelerate==0.31.0         \
        trl[peft]==0.20.1          \
        deepspeed==0.14.2          \
        tf-keras==2.16.*           \
        ninja

# bitsandbytes: pre-built wheels don’t yet target sm90 (H100) → build locally
RUN git clone --depth=1 https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && \
    CUDA_VERSION=12.4 make cuda12x  && \
    pip install . && \
    cd .. && rm -rf bitsandbytes

# -----------------------------------------------------------------------------
# 3.  Copy project & install it as an editable package (keeps host ↔ container in sync)
# -----------------------------------------------------------------------------
WORKDIR /workspace/arc_dslearn
COPY . .
RUN pip install -e .

# -----------------------------------------------------------------------------
# 4.  Small sanity test – prints Torch, Flash-Attn & GPU info (shown at build time)
# -----------------------------------------------------------------------------
RUN python - <<'PY'\n\
import torch, flash_attn\n\
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)\n\
print('GPU(s):', torch.cuda.device_count(), torch.cuda.get_device_name(0))\n\
print('Flash-Attn loaded →', hasattr(flash_attn, '__version__'))\n\
PY

# -----------------------------------------------------------------------------
# 5.  Default entry – override with `docker run … python your_script.py`
# -----------------------------------------------------------------------------
CMD ["/bin/bash"]
