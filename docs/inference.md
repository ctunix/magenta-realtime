# Inference

**JAX:**
```bash
# Generate 4 seconds of audio (single GPU)
mrt jax generate

# Generate 8s with the 2.4B model sharded across 2 GPUs (e.g. Kaggle 2x T4)
mrt jax generate --model mrt2_base --duration 8.0 --shard

# Explicitly request N GPUs (implies --shard; capped to local CUDA device count)
mrt jax generate --num-devices 2 --duration 8.0

# Use a bf16-quantized checkpoint to fit mrt2_base on a single T4 (16 GB):
#   mrt checkpoints quantize mrt2_base        # -> mrt2_base_bf16.safetensors
mrt jax generate --checkpoint mrt2_base_bf16.safetensors --duration 8.0
```

> **Note:** `mrt jax generate` fails loudly if no CUDA GPU is detected (no
> silent CPU fallback). Pass `--no-require-gpu` only for CPU testing.

**MLX:**
```bash
# Generate 4 seconds of audio
mrt mlx generate --bits=8
```

To print MusicCoCa tokens for a prompt directly without generating audio:

```python
from magenta_rt.musiccoca import MusicCoCa
m = MusicCoCa()
print(m.tokenize(m.embed('a jazz piano trio')).tolist())

# Get tokens from audio
from magenta_rt.audio import Waveform
wav = Waveform.from_file("jazz_piano_trio.wav")
print(m.tokenize(m.embed(wav)).tolist())
```

## Python API (JAX)

```python
from magenta_rt import MagentaRT2Jax
from magenta_rt.config import MUSICCOCA

# shard=True shards across all local CUDA GPUs (falls back to single-GPU
# if fewer than 2 are available).
mrt = MagentaRT2Jax(size="mrt2_base", shard=True)
embedding = mrt.embed_style("disco funk")
wav, state = mrt.generate(
    conditioning={MUSICCOCA.key: embedding}, frames=200,
)
wav.write("out.wav")
```

Generation is **streaming**: `generate()` runs one frame per step and carries
the KV cache in `state`. Peak memory is O(1) per step and does **not** scale
with `frames`/`--duration`, so long clips cost no more memory than short ones.
Pass the returned `state` back to `generate(..., state=state)` to continue.

## Bulk generation

Bulk-generate 60s audio clips from MusicCoCa prompts for listener evaluation:

```bash
python scripts/bulk_generate.py --size=mrt2_base
```

Outputs are saved to `outputs/eval_audio/<size>/`.
