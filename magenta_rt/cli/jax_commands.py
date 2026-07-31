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

"""CLI commands for the JAX backend: mrt jax {generate}."""

import click

from magenta_rt.cli import main
from magenta_rt import paths


@main.group()
def jax():
    """JAX backend commands."""


@jax.command()
@click.option("--prompt", default="disco funk", help="Text conditioning for MusicCoCa.")
@click.option("--model", default=paths.DEFAULT_MODEL_NAME, type=str, help="Model variant name (e.g. 'mrt2_base', 'mrt2_small').")
@click.option("--duration", default=4.0, type=float, help="Duration in seconds.")
@click.option("--temperature", default=1.3, type=float)
@click.option("--top-k", default=40, type=int)
@click.option("--cfg-musiccoca", default=3.0, type=float)
@click.option("--cfg-notes", default=1.0, type=float)
@click.option("--checkpoint", default=None, type=str, help="Checkpoint filename in checkpoints/ directory.")
@click.option(
    "--shard",
    is_flag=True,
    default=False,
    help="Shard the model across all local CUDA GPUs (tensor parallelism). "
         "Falls back to single-GPU if fewer than 2 GPUs are detected.",
)
@click.option(
    "--num-devices",
    "num_devices",
    default=None,
    type=int,
    help="Shard across this many GPUs (implies --shard). Capped to the number "
         "of local CUDA devices.",
)
@click.option(
    "--no-require-gpu",
    "no_require_gpu",
    is_flag=True,
    default=False,
    help="Allow CPU execution (for testing). By default the command fails loudly "
         "if no CUDA GPU is available instead of silently falling back to CPU.",
)
def generate(prompt, model, duration, temperature, top_k,
              cfg_musiccoca, cfg_notes, checkpoint, shard, num_devices,
              no_require_gpu):
    """Generate audio with the JAX backend."""
    from magenta_rt.jax.generate import main as run

    kwargs = dict(
        prompt=prompt,
        model_name=model,
        checkpoint=checkpoint,
        temperature=temperature,
        top_k=top_k,
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        duration=duration,
        shard=shard,
        num_devices=num_devices,
        require_gpu=not no_require_gpu,
    )
    run(**kwargs)
