# Magenta RealTime 2

[![CLI Tests](https://github.com/magenta/magenta-realtime/actions/workflows/cli_tests.yml/badge.svg)](https://github.com/magenta/magenta-realtime/actions/workflows/cli_tests.yml)

> [!NOTE]
> **Go [here](https://magenta.withgoogle.com/mrt2) for pre-built Apps & Plugins.**

Magenta RealTime 2 (MRT2) is a state-of-the-art open-weights model for real-time music generation. It contains several key components
* An [open-weights model](https://huggingface.co/google/magenta-realtime-2)
* A [Python library](README.md) `magenta-rt` for inference with JAX and MLX backends
* A [C++ inference engine](core/README.md) `magentart::core` for efficient streaming audio generation on Apple Silicon MacBooks
* A suite of [example applications](examples/README.md) built on the inference engine.

Use this project to run inference, build a DAW plugin, or embed the model into new applications of your imagination.
Future updates will support supervised fine-tuning.

📖 **Full documentation:** https://magenta.github.io/magenta-realtime/
(or build it locally — see [`docs/README.md`](docs/README.md)).

> [!NOTE]
> **Looking for Magenta RealTime v1?** The original model and code have been moved to the [`v1_legacy`](https://github.com/magenta/magenta-realtime/tree/v1_legacy) branch.

## Repo Highlights

- `magenta_rt/` — Python inference library (JAX / MLX backends).
- `core/` — C++ inference library (`magentart::core`).
- `examples/mrt2/auv3` — All-in-one AUv3 plugin for DAWs.
- `examples/mrt2/standalone` — All-in-one standalone macOS app.
- `examples/jam/` — App for exploring note control.
- `examples/collider/` — App for exploring prompt space.
- `notebooks/` - Notebook for trying Python API.

## Hardware requirements

**Real-time streaming** requires **Apple Silicon** (M-series). We offer two model sizes:

- **`mrt2_small`** (230M parameters) — runs real-time on any Apple Silicon Mac, including Air models.
- **`mrt2_base`** (2.4B parameters) — higher quality; requires a Pro Max chip for real-time streaming.

The table below shows which devices support **real-time streaming** (generating audio faster than playback):

| Device | `mrt2_small` (230M) | `mrt2_base` (2.4B) |
|---|---|---|
| M5 Max | ✅ | ✅ |
| M3 Max | ✅ | ✅ |
| M2 Max | ✅ | ✅ |
| M4 Pro | ✅ | ✅ |
| M2 Pro | ✅ | ❌ |
| M1 Pro | ✅ | ❌ |
| M4 Air | ✅ | ❌ |
| M3 Air | ✅ | ❌ |
| M1 Air | ✅ | ❌ |

> **Note:** Both models can also run **offline (non-real-time) inference** on any Apple Silicon Mac or NVIDIA GPU via the Python library. See more details on [`docs/models.md`](docs/models.md).

## Quickstart on Kaggle / Colab (NVIDIA GPU, incl. 2x T4)

Run this in a single Kaggle/Colab cell (GPU accelerator enabled). It clones this
fork, installs a **pinned, non-drifting JAX/CUDA stack**, downloads the model
resources + the `mrt2_base` checkpoint, and quantizes it to bf16 so it fits on a
T4. It is idempotent and safe to re-run.

```bash
curl -fsSL https://raw.githubusercontent.com/ctunix/magenta-realtime/main/scripts/install_kaggle.sh | bash
```

When it finishes it prints the exact Python cell to run for generation. That cell
sets the JAX memory env vars **before** importing JAX, asserts a GPU is present
(no silent CPU fallback), shards the 2.4B model across 2 GPUs when 2 are present
(tensor parallelism — ~5 GB/GPU fp32 or ~2.6 GB/GPU bf16), and generates 8 s of
audio. The full notebook is in [`notebooks/kaggle_2xt4_mrt2.ipynb`](notebooks/kaggle_2xt4_mrt2.ipynb).

<details><summary>Or, the equivalent generation cell (run it after the install above)</summary>

```python
import os
# Must be set BEFORE the first `import jax` so jaxlib reads them.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')
os.environ.setdefault('MAGENTA_HOME', '/kaggle/working/magenta-rt-v2')  # /content/... on Colab

import logging, jax
logging.basicConfig(level=logging.INFO, force=True)
from magenta_rt.jax import _gpu_check
print(_gpu_check.diagnose_devices())
_gpu_check.assert_gpu_available()          # fail loudly, no silent CPU fallback
n_gpu = _gpu_check.num_local_cuda_devices()

from magenta_rt import MagentaRT2Jax
from magenta_rt.config import MUSICCOCA
mrt = MagentaRT2Jax(
    size='mrt2_base',
    checkpoint='mrt2_base_bf16.safetensors',
    shard=(n_gpu >= 2),                     # tensor-parallel across 2 GPUs if present
    require_gpu=True,
)
print('Sharded:', mrt._sharded, '| mesh:', mrt._mesh)

embedding = mrt.embed_style('disco funk', use_mapper=True)
wav, state = mrt.generate(conditioning={MUSICCOCA.key: embedding}, frames=200)  # 8 s @ 25 Hz

import IPython.display as ipd
out = '/kaggle/working/mrt2_base_output.wav'
wav.write(out)
print('Saved', out)
ipd.display(ipd.Audio(wav.samples.T, rate=wav.sample_rate))
```

</details>

Notes:
- The install pins `jax==jaxlib==jax-cuda12-plugin==jax-cuda12-pjrt==0.10.1`,
  `numpy==2.3.5`, `numba==0.65.1` so the CUDA plugin can't drift out of sync.
- To use your own fork instead of `ctunix`, set `MRT_REPO` before the curl, e.g.
  `MRT_REPO=https://github.com/<you>/magenta-realtime.git curl ... | bash`.
- `mrt2_base` is 2.4B params; on a single T4 use the **bf16** checkpoint
  (`mrt checkpoints quantize mrt2_base`), on 2x T4 it also shards across GPUs.

## Quickstart on Apple Silicon

```bash
# Install uv if you haven't and create a venv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies (Python dev)
uv pip install "magenta-rt[mlx]"

# Download resources: style model and codec model
# (i.e., MusicCoCa and SpectroStream)
mrt models init
# Download the streaming model you want to use
mrt models download
# Generate 4 seconds of music (change to `mrt2_small` for small model)
mrt mlx generate --prompt "disco funk" --duration 4.0 --model=mrt2_base
```

### Python Development

For local development, clone the repo instead of installing from PyPI:

```bash
git clone --recurse-submodules https://github.com/magenta/magenta-realtime.git
cd magenta-realtime
uv pip install -e ".[mlx]"
```

### C++ App Development

To get started building C++ apps, perform the following setup:

```bash
# Install dependencies
uv pip install "cmake<3.28"

# Build hello_mrt2 (a basic command-line interface)
cmake . -B build
cmake --build build --target hello_mrt2 -j10

# Generate 4 seconds of music
./build/examples/hello_mrt2/hello_mrt2 \
    ~/Documents/Magenta/magenta-rt-v2/models/mrt2_base/mrt2_base.mlxfn \
    ~/Documents/Magenta/magenta-rt-v2/resources \
    100 \
    --prompt "ambient pads with sub bass"
```

See the full documentation:

- [Installation](docs/installation.md)
- [Models & checkpoints](docs/models.md)
- [Inference](docs/inference.md)
- [Exporting models](docs/exporting.md)
- [Latency benchmark](docs/benchmark.md)
- [Testing](docs/testing.md)

## Other resources

- [Get Started](https://magenta.withgoogle.com/mrt2)
- [Blog Post](https://magenta.withgoogle.com/magenta-realtime-2)
- [Hugging Face](https://huggingface.co/google/magenta-realtime-2)

## Citing this work

Please cite our previous [technical report](https://arxiv.org/abs/2508.04651):

**BibTeX:**

```
@article{gdmlyria2025live,
    title={Live Music Models},
    author={Caillon, Antoine and McWilliams, Brian and Tarakajian, Cassie and Simon, Ian and Manco, Ilaria and Engel, Jesse and Constant, Noah and Li, Pen and Denk, Timo I. and Lalama, Alberto and Agostinelli, Andrea and Huang, Anna and Manilow, Ethan and Brower, George and Erdogan, Hakan and Lei, Heidi and Rolnick, Itai and Grishchenko, Ivan and Orsini, Manu and Kastelic, Matej and Zuluaga, Mauricio and Verzetti, Mauro and Dooley, Michael and Skopek, Ondrej and Ferrer, Rafael and Borsos, Zal{\'a}n and van den Oord, {\"A}aron and Eck, Douglas and Collins, Eli and Baldridge, Jason and Hume, Tom and Donahue, Chris and Han, Kehang and Roberts, Adam},
    journal={arXiv:2508.04651},
    year={2025}
}
```
