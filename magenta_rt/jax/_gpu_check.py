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

"""GPU detection, loud-fail startup assertion, and JAX memory defaults.

Why this exists
----------------
JAX silently falls back to CPU when the CUDA plugin is missing or mismatched
(see requirements-kaggle.txt). A CPU fallback for a 2.4B-param model is
useless and wastes the user's time, so we assert loudly instead.

It also sets sensible XLA memory env vars *before* JAX is imported. The mrt2_base
OOM on a single T4 is a BFC fragmentation problem: with preallocate=true XLA
grabs ~90% of GPU memory up front and then cannot find contiguous 2-8GB chunks
for the streaming allocator alongside the MusicCoCa/SpectroStream resources.
Defaulting to preallocate=false lets the allocator grow on demand.

These are *defaults* only; any value already set in the environment wins.
"""

import os

# Sensible defaults for memory-constrained GPUs (T4 16GB, ~15GB usable).
# Override by setting the env var before importing this module / starting Python.
_DEFAULTS = {
    # Do not grab 90% of GPU memory up front; grow the BFC allocator on demand.
    # This avoids the "failed to allocate N GB contiguous chunk" fragmentation
    # failures when MusicCoCa/SpectroStream/JAX share one T4.
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # Even with preallocate=false, cap XLA at 85% so other processes (and the
    # TFLite MusicCoCa interpreter) have room.
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
}


def configure_jax_memory_defaults() -> None:
    """Set default XLA/JAX memory env vars if the user has not set them.

    Call this BEFORE `import jax` for the vars to take effect. Idempotent.
    """
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)


class GPUNotAvailableError(RuntimeError):
    """Raised when JAX cannot find a usable CUDA device."""


def assert_gpu_available() -> None:
    """Fail loudly if JAX is not running on at least one CUDA GPU.

    Raises:
        GPUNotAvailableError: if no CudaDevice is found (silent CPU fallback).
    """
    import jax  # local import; this module is imported before jax elsewhere

    devices = jax.devices()
    cuda_devices = [d for d in devices if d.platform == "gpu"]

    if not cuda_devices:
        raise GPUNotAvailableError(
            "No CUDA GPU devices found via jax.devices() -> "
            f"{[str(d) for d in devices]}. "
            "JAX silently fell back to CPU. This usually means the CUDA plugin "
            "is missing or version-mismatched. Verify that jax, jaxlib, "
            "jax-cuda12-plugin and jax-cuda12-pjrt are ALL the exact same "
            "version (see requirements-kaggle.txt). "
            f"jax.__version__={jax.__version__}; "
            f"devices={devices}."
        )


def diagnose_devices() -> str:
    """Return a human-readable summary of JAX devices for logging."""
    import jax

    lines = [f"jax.__version__ = {jax.__version__}"]
    devices = jax.devices()
    lines.append(f"jax.devices() ({len(devices)}):")
    for d in devices:
        lines.append(f"  {d} (platform={d.platform})")
    cuda = [d for d in devices if d.platform == "gpu"]
    lines.append(f"CUDA devices: {len(cuda)}")
    if cuda:
        try:
            # Per-device memory stats (best-effort; works on CUDA).
            for d in cuda:
                stats = d.memory_stats() or {}
                limit = stats.get("limit") or stats.get("bytes_limit")
                used = stats.get("bytes_in_use", 0)
                if limit:
                    lines.append(
                        f"  {d}: {used / 1e9:.2f} GB / {limit / 1e9:.2f} GB in use"
                    )
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(lines)


def num_local_cuda_devices() -> int:
    """Count local CUDA devices. Safe to call after jax import."""
    import jax

    return sum(1 for d in jax.devices() if d.platform == "gpu")
