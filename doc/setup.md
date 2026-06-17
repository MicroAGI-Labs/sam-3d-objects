# Setup

> **MicroAGI Blackwell fork note.** This branch (`microagi-blackwell`) is patched to run on
> NVIDIA Blackwell GPUs (RTX PRO 6000, `sm_120`) with **CUDA 13.0**, instead of the upstream
> torch 2.5.1 / CUDA 12.1 pin (whose wheels have no `sm_120` kernels). Changes are
> dependency-only — no model/inference logic was touched:
> - **spconv** (a hard inference dep — `SparseConv3d` runs in `structured_latent_flow.py` and
>   `structured_latent_vae/decoder_mesh.py`) is swapped from `spconv-cu121==2.3.8` to the
>   **rathaROG** prebuilt `cumm-cu130` + `spconv-cu130` wheels, which ship `sm_120+PTX`
>   (`--extra-index-url https://ratharog.github.io/cumm-spconv/`). Fallback: source-build with
>   `CUMM_CUDA_ARCH_LIST='...;12.0;12.1' CUMM_DISABLE_JIT=1 SPCONV_DISABLE_JIT=1`.
> - **flash-attn** and **xformers** are dropped — they are optional attention backends, and
>   `set_attention_backend()` only selects `flash_attn` on A100/H100/H200; on `sm_120` the
>   default `sdpa` backend is used. Set `ATTN_BACKEND=sdpa` / `SPARSE_ATTN_BACKEND=sdpa` to be
>   explicit.
> - torch/torchvision/torchaudio install from the **cu130** index; `pytorch3d` and `gsplat` are
>   source/JIT-built with `TORCH_CUDA_ARCH_LIST=12.0`.
> - **kaolin** is a hard dep too — `representations/mesh/flexicubes/flexicubes.py` imports
>   `kaolin.utils.testing` on the pipeline import path, so the pipeline won't load without it.
>   The PyPI `kaolin` is only a placeholder wheel that raises on import, and NVIDIA ships no
>   cu130 wheel (newest prebuilt is `torch-2.8.0_cu129`), so it is **built from `master`**
>   (already cu13/Blackwell-aware: `TORCH_MAX_VER=2.10.0`, default arch includes `12.0+PTX`).
>   The build needs `setuptools<80` (its `setup.py` imports `pkg_resources`) and
>   `CUB_HOME=/usr/local/cuda/include/cccl` (CUDA 13 relocated cub/thrust under `cccl/`).
> - The server installs the slim **`requirements.serve.txt`** (this fork) rather than the full
>   `requirements.txt`, which is the upstream training/dev set (bpy, sagemaker, wandb,
>   auto_gptq, …) — most of it irrelevant to inference and some uncompilable on cu130.
>
> The runnable artifact is the container in MicroAGI-Labs/research-infra at
> `apps/cluster/workloads/sam-3d-objects/` — this repo is consumed there as a submodule.

## Prerequisites

* A linux 64-bits architecture (i.e. `linux-64` platform in `mamba info`).
* A NVIDIA GPU with at least 32 Gb of VRAM.

## 1. Setup Python Environment

The following will install the default environment. If you use `conda` instead of `mamba`, replace its name in the first two lines. Note that you may have to build the environment on a compute node with GPU (e.g., you may get a `RuntimeError: Not compiled with GPU support` error when running certain parts of the code that use Pytorch3D).

```bash
# create sam3d-objects environment
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

## 2. Getting Checkpoints

### From HuggingFace

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects
Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

⚠️ SAM 3D Objects is available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions.
Sanctioned jurisdiction will result in requests being **rejected**.

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```


