# Copyright 2026 Google LLC
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

"""Checkpoint quantization utility (fp32 -> bf16/fp16).

Halves the on-disk and on-GPU footprint of a safetensors checkpoint, as an
alternative or complement to multi-GPU sharding for users on a single GPU.

mrt2_base is 2.4B params ~= 9.84 GB in fp32 but ~= 4.92 GB in bf16, which fits
on a single T4 (15 GB usable) with headroom for activations. The model already
computes in bfloat16 (`compute_dtype = jnp.bfloat16`), so loading bf16 params
changes only storage, not numerics (the upcast to fp32 for RMSNorm reductions
still happens via `reductions_in_at_least_fp32`).

Runs on CPU (uses numpy + ml_dtypes, no GPU required), so you can quantize on a
laptop and upload the smaller checkpoint to Kaggle.
"""

import os
from pathlib import Path

import numpy as np

try:
    import ml_dtypes  # provides bfloat16 dtype for numpy/safetensors
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ml_dtypes is required for bf16 quantization. "
        "Install it with: pip install ml_dtypes"
    ) from exc


# Numpy dtype objects for the supported target dtypes.
_DTYPES = {
    "bf16": ml_dtypes.bfloat16,
    "bfloat16": ml_dtypes.bfloat16,
    "fp16": np.float16,
    "float16": np.float16,
    "fp32": np.float32,
    "float32": np.float32,
}


def _load_flat(path: str) -> dict:
    """Load a safetensors checkpoint as a flat {name: np.ndarray} dict."""
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def _save_flat(tensors: dict, path: str, metadata: dict | None = None) -> None:
    """Save a flat {name: np.ndarray} dict to a safetensors file.

    Tries the numpy backend first (no JAX needed). Falls back to the flax
    backend (which natively handles bfloat16 via JAX) if the numpy backend
    rejects the bf16 dtype.
    """
    try:
        from safetensors.numpy import save_file

        save_file(tensors, path, metadata=metadata)
    except Exception as numpy_err:  # noqa: BLE001 - fall back to flax
        import jax.numpy as jnp
        from safetensors.flax import save_file as flax_save_file

        jax_tensors = {k: jnp.asarray(v) for k, v in tensors.items()}
        flax_save_file(jax_tensors, path, metadata=metadata)
        del numpy_err


def quantize_checkpoint(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    target_dtype: str = "bf16",
    keep_fp32_patterns: list[str] | None = None,
) -> dict:
    """Convert a safetensors checkpoint to a lower-precision dtype.

    Args:
        src: Source checkpoint path (.safetensors).
        dst: Destination checkpoint path (.safetensors).
        target_dtype: One of 'bf16'/'bfloat16', 'fp16'/'float16', 'fp32'.
        keep_fp32_patterns: Substrings; any param whose name contains one of
            these is left in fp32 (e.g. ['norm', 'scale'] to keep RMSNorm
            scales in fp32 for stability). Default: quantize everything.

    Returns:
        A dict with 'src_bytes', 'dst_bytes', 'num_params', 'num_kept_fp32'.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")
    if target_dtype not in _DTYPES:
        raise ValueError(
            f"Unsupported target_dtype {target_dtype!r}. "
            f"Choose one of: {sorted(_DTYPES)}"
        )
    np_dtype = _DTYPES[target_dtype]
    keep = keep_fp32_patterns or []

    flat = _load_flat(str(src))
    src_bytes = src.stat().st_size

    out = {}
    num_params = 0
    num_kept_fp32 = 0
    for name, arr in flat.items():
        num_params += int(np.prod(arr.shape)) if arr.shape else int(arr.size)
        if arr.dtype == np.float32:
            if any(p and p in name for p in keep):
                out[name] = arr
                num_kept_fp32 += 1
            else:
                out[name] = arr.astype(np_dtype)
        else:
            # Non-float tensors (int indices, bool masks): keep as-is.
            out[name] = arr

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Remove any stale destination so save_file doesn't error on overwrite.
    if dst.exists():
        dst.unlink()
    _save_flat(out, str(dst), metadata={"quantized_to": target_dtype})
    dst_bytes = dst.stat().st_size

    return {
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "num_params": num_params,
        "num_kept_fp32": num_kept_fp32,
        "target_dtype": target_dtype,
    }
