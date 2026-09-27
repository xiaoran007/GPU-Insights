# GPU-Insights

Multi-model GPU/NPU training performance benchmark suite. Measures compute throughput across diverse deep learning workloads and hardware platforms.

## Features

- **Smart Launcher** — One command auto-selects device, precision, ABS, and CUDA DDP
- **5 Benchmark Models** — CNN, ResNet-50, ViT, UNet, DDPM covering classification, segmentation, and diffusion
- **6 Device Backends** — CUDA, MPS, NPU (Huawei Ascend), MUSA (Moore Threads), TPU, auto-detection
- **DDP Multi-GPU** — Distributed data-parallel training via `torchrun`
- **Auto Batch Size** — Calibration-table-based automatic batch size selection (NVML)
- **Unified Scoring** — Throughput-based scoring system consistent across all models
- **LLM Inference Track** — Standalone full-GPU llama.cpp benchmark for coding-agent workloads

## Quick Start

```shell
# Install dependencies
pip install torch torchvision

# Run one model with the smart launcher
python3 main_auto.py -mt resnet50

# Omit --model to run the major model set: resnet50, vit, unet, ddpm
python3 main_auto.py

# Preview what the smart launcher will do
python3 main_auto.py -mt vit --dry-run

# Force a single precision or batch size when needed
python3 main_auto.py -mt unet --dtype FP32
python3 main_auto.py -mt ddpm -bs 32
```

Legacy expert entrypoints remain available:

```shell
python main.py -mt resnet50 -s 512 -e 2 -dt FP32
torchrun --nproc_per_node=2 main_ddp.py -mt resnet50 -s 512 -e 2 -abs -dt FP16
python main_tpu.py -mt resnet50 -s 512 -e 2 -dt BF16
```

## LLM Inference Benchmark

The LLM inference track is separate from the training benchmark path. It uses
`llama-bench` from an external llama.cpp installation and records prompt
processing (PP) plus token generation (TG) throughput for coding-agent-shaped
cases.

Prepare llama.cpp with the helper below or an external installation. GPU-Insights
does not install GPU drivers or backend toolchains such as CUDA, ROCm, Vulkan,
or SYCL. The benchmark launcher accepts an explicit binary path with `--llama-bench`.

### Prepare llama.cpp

GPU-Insights calls the `llama-bench` binary produced by llama.cpp. Follow the
upstream llama.cpp build guide for your platform:

- Official build guide: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- Intel GPU / SYCL guide: <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md>

GPU-Insights provides a helper that prepares `llama-bench`. For CUDA on Linux
amd64, it first checks the visible NVIDIA GPU and CUDA major version, then tries
to install the matching GPU-Insights prebuilt release asset. If that fails in an
interactive shell, it asks whether to fall back to a source build. Other
backends go directly to source build. Source builds clone llama.cpp, check out
upstream `origin/HEAD` by default, and build both `llama-bench` and `llama-server`
with their CMake dependencies. Pass
`--ref <git-ref>` only when you want to pin a specific llama.cpp commit, branch,
or tag. The helper does not install GPU drivers, CUDA, ROCm, Vulkan SDK, oneAPI,
compilers, or CMake.

```shell
# Interactive backend selection
bash scripts/bootstrap-llama-cpp.sh

# Non-interactive CUDA example
bash scripts/bootstrap-llama-cpp.sh --backend cuda --jobs 16

# Force source build, skipping prebuilt release assets
bash scripts/bootstrap-llama-cpp.sh --backend cuda --prebuilt off

# Require a prebuilt asset and stop if it is unavailable
bash scripts/bootstrap-llama-cpp.sh --backend cuda --prebuilt on

# Optional pinned-ref build
bash scripts/bootstrap-llama-cpp.sh --ref <llama.cpp-commit> --backend cuda --jobs 16
```

Prebuilt CUDA assets are installed under `third_party/llama-bench/`, with
`third_party/llama-bench/current/bin/llama-bench` pointing at the selected
release. Override the GitHub release source with `--release-repo <owner/repo>`
or `--release-tag <tag>`. On CUDA 12 hosts with visible pre-Ampere GPUs, the
helper selects the `cuda12-legacy` asset automatically. The LLM launcher checks
that prebuilt path before the source-build path and `PATH`.

For Linux amd64 CUDA release assets, build reproducible tarballs in Docker and
upload them manually to a GitHub Release:

```shell
# Build linux-amd64-cuda12, linux-amd64-cuda12-legacy, and linux-amd64-cuda13 assets
bash scripts/build-llama-bench-release.sh --ref <llama.cpp-commit> --jobs 16

# Or build one variant
bash scripts/build-llama-bench-release.sh --variant cuda12 --ref <llama.cpp-commit>
bash scripts/build-llama-bench-release.sh --variant cuda12-legacy --ref <llama.cpp-commit>
bash scripts/build-llama-bench-release.sh --variant cuda13 --ref <llama.cpp-commit>
```

The release builder mounts a named Docker volume at `/work` by default
(`gpu-insights-llama-bench-work`) so the llama.cpp checkout and CMake build tree
survive the temporary `docker run --rm` container. Repeated builds reuse that
cache per variant. Use `--clean-work` to clear the selected variant work tree,
or `--work-volume <name>` / `GPU_INSIGHTS_LLAMA_BENCH_WORK_VOLUME` to isolate a
different cache. Release files written under `dist/` are chowned back to the
host UID/GID after packaging so they can be uploaded, overwritten, or deleted
without `sudo`.

The CUDA 12 asset is built with `GGML_NATIVE=OFF` and
`CMAKE_CUDA_ARCHITECTURES=80;86;87;89;90`, covering Ampere, Ada, and Hopper.
The CUDA 12 legacy asset uses `60;61;62;70;72;75`, covering Pascal, Volta, and
Turing while excluding Ampere. Pascal is kept in the CUDA 12 legacy package;
CUDA 13 is not used for this target because CUDA 13 library support dropped
older pre-Turing architectures.
The CUDA 13 asset uses `80;86;87;88;89;90;100;103;110;120;121`, covering
Ampere and later CUDA targets including Blackwell-generation SMs supported by
CUDA 13 nvcc. Release packages include a `llama-bench` wrapper, the compiled
`llama-bench.bin`, llama.cpp/ggml shared libraries, the GCC runtime libraries
needed by the build (`libstdc++.so.*`, `libgcc_s.so.*`, `libgomp.so.*`, and
`libatomic.so.*` when linked),
`LICENSE.llama.cpp`, `BUILD-MANIFEST.json`, and SHA256 checksums. Bundling the
C++ runtime keeps cluster module environments with older `libstdc++` from
shadowing the ABI required by the release build. The package script strips
release binaries and writes `.tar.zst` archives. They intentionally do not
bundle NVIDIA driver, CUDA runtime, cuBLAS, glibc, or model files. The Docker
build uses CUDA stub libraries only for link-time `libcuda.so.1` resolution;
target machines still provide the real NVIDIA driver at runtime.

On CUDA systems, if the active GCC is newer than the CUDA toolkit supports,
load a compatible compiler module first or pass it explicitly:

```shell
bash scripts/bootstrap-llama-cpp.sh --backend cuda --cuda-host-compiler /path/to/g++-14
```

By default the helper uses `third_party/llama.cpp`, which is ignored by git.
Override paths with `--dir`, `--build-dir`, or the corresponding
`GPU_INSIGHTS_LLAMA_CPP_*` environment variables printed by `--help`.

General source checkout:

```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Common local builds:

```shell
# CPU-only sanity build
cmake -B build
cmake --build build --config Release -j

# NVIDIA CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# AMD ROCm / HIP on Linux
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Vulkan, useful for cross-vendor Windows/Linux setups
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j
```

On Windows, use a Visual Studio 2022 Developer Command Prompt or the toolchain
prompt required by the selected GPU backend, then run the same CMake build shape
with the relevant `GGML_*` backend flag. On macOS, Metal is enabled by default
in llama.cpp, so the regular CMake build is the expected local GPU build.

For Intel GPU, llama.cpp recommends the SYCL backend with Intel oneAPI. After
installing Intel GPU drivers and oneAPI, source the oneAPI environment and
build with SYCL:

```shell
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build --config Release -j
```

After building, either add `llama.cpp/build/bin` to `PATH` or pass the binary
explicitly:

```shell
python3 -m llm_bench.cli --llama-bench /path/to/llama.cpp/build/bin/llama-bench
```

Keep GPU memory behavior strict for benchmark submissions. Do not enable CUDA
unified-memory/system-RAM fallback for submitted full-GPU results; if a case
does not fit in VRAM, let it fail and import the failed case status.

### Fixed model and cases

The default contract lives in `llm_bench/configs/default.json`.

Model:

- Repo: `unsloth/Qwen3.6-27B-GGUF`
- Revision: `82d411acf4a06cfb8d9b073a5211bf410bfc29bf`
- File: `Qwen3.6-27B-Q4_K_M.gguf`
- Default local path: `models/llm/Qwen3.6-27B-Q4_K_M.gguf`

Cases:

| Case | Prompt Tokens | Generation Tokens | Purpose |
|------|--------------:|------------------:|---------|
| `agent_step_small` | 2,048 | 128 | Short tool result or agent loop step |
| `single_file_edit` | 8,192 | 512 | One file plus focused context |
| `multi_file_patch` | 16,384 | 1,024 | Multi-file edit with longer patch output |
| `repo_context_plan` | 32,768 | 512 | Large repo context with concise planning |
| `long_context_debug` | 65,536 | 1,024 | Long logs/diffs/context with substantial response |

Results are full-GPU only: the default config uses `nGpuLayers: -1` and does
not fall back to partial CPU offload. If a GPU cannot fit a case, that case is
recorded as `status: failed` with the error text.

The default runtime profile is a single representative coding-agent setup, not
a tuning matrix: full GPU offload, F16 KV cache, flash attention enabled,
`batchSize: 2048`, `ubatchSize: 512`, and an explicit per-case context size
rounded up from `prompt + generation + 256` tokens. MTP/speculative decoding is
not part of the canonical baseline.

### Gemma small-GPU auxiliary path

For smaller GPUs, GPU-Insights also provides non-dashboard Gemma presets:

```shell
# Preview or download a Gemma GGUF
python3 scripts/download-llm-model.py --gemma --12b --dry-run
python3 scripts/download-llm-model.py --gemma --12b
python3 scripts/download-llm-model.py --gemma --e2b --dry-run
python3 scripts/download-llm-model.py --gemma --e2b

# Run the small-GPU cases
python3 -m llm_bench.cli --gemma --12b
python3 -m llm_bench.cli --gemma --e2b

# List only the Gemma auxiliary cases
python3 -m llm_bench.cli --gemma --12b --list-cases
python3 -m llm_bench.cli --gemma --e2b --list-cases
```

The 12B path uses `unsloth/gemma-4-12B-it-qat-GGUF` with
`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`. The E2B path uses
`unsloth/gemma-4-E2B-it-qat-GGUF` with
`gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`. Both follow the repositories'
`UD-Q4_K_XL` llama.cpp recommendation. They are intended for local hardware
checking only: by default Gemma auxiliary runs print per-case results and the
summary but do not write dashboard payload files, debug sidecars, Base64
payloads, or an import command. Pass `--output-file` or `--emit-base64` only
when you need a local inspection payload; do not import Gemma auxiliary results
into the dashboard.

Gemma cases:

| Case | Prompt Tokens | Generation Tokens | Purpose |
|------|--------------:|------------------:|---------|
| `small_agent_step` | 1,024 | 128 | Compact tool result or short agent loop response |
| `focused_file_edit` | 4,096 | 384 | One file plus instructions with a small patch |
| `two_file_patch` | 8,192 | 512 | Two focused files or diff chunks |

### Download the fixed GGUF

```shell
# Oscar cluster: link models/llm to persistent storage before downloading
bash scripts/bootstrap-llm-oscar.sh

# AutoDL: link models/llm to persistent storage before downloading
bash scripts/bootstrap-llm-autodl.sh

# Colab: copy an existing Drive GGUF into local runtime storage before running
bash scripts/bootstrap-llm-colab.sh

# Vast.ai: prepare an ephemeral instance workspace before downloading
bash scripts/bootstrap-llm-vast.sh

# Preview the exact Hugging Face URL and output path
python3 scripts/download-llm-model.py --dry-run

# Download to models/llm/Qwen3.6-27B-Q4_K_M.gguf
python3 scripts/download-llm-model.py
```

The downloader is only a model-file helper. It does not install llama.cpp or
configure GPU runtime libraries.

If the final GGUF already exists and matches the configured expected byte size,
the downloader exits without downloading. Interrupted downloads are kept as a
`.part` file and resumed on the next run when the server honors HTTP Range
requests. Progress shows human-readable downloaded and total sizes; when
resuming, downloaded includes the bytes already present in the `.part` file.

On Oscar, `scripts/bootstrap-llm-oscar.sh` links the repo-local `models/llm`
path to `/users/tfang11/tfang/llm` so the GGUF survives compute-node teardown
and does not need to be downloaded again. Override the target with
`GPU_INSIGHTS_OSCAR_LLM_DIR=/path/to/llm` if needed.

On AutoDL, `scripts/bootstrap-llm-autodl.sh` links `models/llm` to
`/root/autodl-fs/llm`. Override the target with
`GPU_INSIGHTS_AUTODL_LLM_DIR=/path/to/llm` if needed.

On Colab, `scripts/bootstrap-llm-colab.sh` expects the fixed GGUF to already
exist in Google Drive at `/content/drive/MyDrive/GPU-Insights/llm`, copies it
into `/content/gpu-insights-llm`, then links `models/llm` to that local runtime
cache. This keeps Google Drive as persistent storage while avoiding Drive I/O on
the benchmark hot path. Mount Drive in the notebook first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Override the Drive cache with `GPU_INSIGHTS_COLAB_LLM_DIR=/path/to/llm`, or the
local runtime cache with `GPU_INSIGHTS_COLAB_LOCAL_LLM_DIR=/content/path`. If the
Drive GGUF is missing or has the wrong byte size, the script exits without
copying or downloading.

On Vast.ai, use a CUDA base image such as
`nvidia/cuda:12.6.3-runtime-ubuntu22.04`, clone the repo into the instance, then
run `scripts/bootstrap-llm-vast.sh`. The Vast helper only checks the local
one-shot instance environment, creates `models/llm`, verifies basic tools and
disk space, and prints the explicit download/build/run commands. It does not
download the GGUF, install or build llama.cpp, or configure persistent model
storage. Because Vast instances are usually rented as ephemeral containers, run
`python3 scripts/download-llm-model.py` on the instance after the bootstrap.
Override the free-space guard with `GPU_INSIGHTS_VAST_MIN_FREE_GIB=60` if you
want a larger local disk margin.

### Run the llama.cpp API server (source build)

The API service uses `llama-server` directly, with the same model presets as the
downloader. **Use `--prebuilt off`: existing prebuilt releases and Docker benchmark
wrappers only provide `llama-bench`; server release packaging is not included.**
Activate your Python environment and run these commands from the project root:

```shell
# Source-build both binaries for an NVIDIA GPU, including L40S
bash scripts/bootstrap-llama-cpp.sh --backend cuda --prebuilt off --jobs 16

# Download Qwen3.8-27B UD-Q6_K (~22 GB), then serve it on one L40S
python scripts/download-llm-model.py --qwen38
python -m llm_bench.serve --qwen38 --device CUDA0
```

The launcher requires only the Python standard library. CMake builds the server's
native dependencies; the host still needs the compiler, CMake, CUDA toolkit (for
CUDA), and development libraries required by the selected upstream revision
(including OpenSSL development files for current upstream defaults). The helper
does not install missing system packages. Use `--ref <commit-or-tag>` to pin
llama.cpp. See the [upstream server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

The listener defaults to `127.0.0.1:18080`. With `--qwen38`, the serving preset
targets a dedicated 48 GB L40S: **131,072 tokens per request and 4 concurrent
slots**, with Q8_0 K/V caches, continuous batching, and separate KV buffers per
slot. The launcher passes 524,288 total tokens to llama-server; each slot's budget
includes both input and output. Raising concurrency preserves the per-slot context
and increases memory use. Jinja chat templates are enabled.

The Qwen3.8 serving preset loads `llm_bench/templates/qwen3.8-agent.jinja`, based
on Unsloth's template at revision `3ea932cee0a432ae86e9c7826cbe8aef52323a28`.
It preserves later system/developer messages in place as system blocks instead
of raising `System message must be at the beginning.` This addresses HTTP 500
from agent requests containing such messages while retaining the upstream tool
and reasoning formats. It is a compatibility modification, not a guarantee of
model adherence to instructions. Use `--chat-template-file /path/to/template.jinja`
to supply another template. No GGUF download or native rebuild is required for
this change; restart the server after updating the checkout.

If the server reports `Invalid API Key`, rerun the local connection helper and
use its newly printed Claude Code command: each compute-helper restart generates
a new key. Authentication errors are separate from Jinja/template HTTP 500 errors.

Capacity estimate for Qwen3.8 UD-Q6_K: weights are about 20.47 GiB and attention
KV caches about 17 GiB at 4 x 128K. The KV estimate is
`16 attention layers x 2 (K,V) x 4 KV heads x 256 dimensions x (34/32 bytes for Q8_0) x 524288 tokens`.
This follows the [model architecture](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/config.json);
it excludes recurrent states, compute buffers, and allocator overhead. This is a
capacity-based starting point, not a measured throughput optimum or an OOM
guarantee. Four long requests share compute and can have high prefill latency.
Q8 KV caching also introduces quantization error relative to F16. Inspect actual
server memory and latency on the target node before increasing capacity.

For an 80 GB GPU, 4 x 256K or 8 x 128K with Q8 KV uses about 34 GiB of attention
KV, before other overheads; choose longer contexts or more concurrent agents to
match the workload. Faster GPUs with the same 48 GB do not gain memory capacity.
Other model presets retain 32K/one slot unless their config has a `serving` section.
Serving overrides do not change benchmark KV settings. Model loading and request
logs stream directly from llama-server. Stop the foreground service with Ctrl+C.

```shell
# Health check (HTTP 200 once the model is ready)
curl http://127.0.0.1:18080/health

# OpenAI-compatible chat endpoint; base URL for clients: http://127.0.0.1:18080/v1
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3_8_27b_ud_q6_k","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'

# Explicit L40S capacity: 4 x 128K (same as the Qwen3.8 serving preset)
python -m llm_bench.serve --qwen38 --ctx-per-slot 131072 --parallel 4

# 80 GB GPU: prioritize longer contexts, or more concurrent agents
python -m llm_bench.serve --qwen38 --ctx-per-slot 262144 --parallel 4
python -m llm_bench.serve --qwen38 --ctx-per-slot 131072 --parallel 8

# --ctx-size retains native TOTAL-context semantics; do not confuse it with per-slot context
python -m llm_bench.serve --qwen38 --ctx-size 524288 --parallel 4 --port 18081

# Other model presets or an explicit source-built binary
python -m llm_bench.serve --gemma --12b
python -m llm_bench.serve --config /path/to/config.json \
  --llama-server /path/to/build/bin/llama-server

# Forward native server options after -- (e.g. authentication for remote access)
python -m llm_bench.serve --qwen38 --host 0.0.0.0 -- --api-key-file /path/to/api-keys.txt
```

Without a model selector, the existing default Qwen3.6 preset is used. The API
model name defaults to the config's `model.key`. `--model-path` overrides the GGUF
path. The binary defaults to `third_party/llama.cpp/build/bin/llama-server`;
`GPU_INSIGHTS_LLAMA_CPP_DIR` and `GPU_INSIGHTS_LLAMA_CPP_BUILD_DIR` also apply.
For custom or multi-configuration build layouts, pass `--llama-server` explicitly.
Additional native arguments after `--` are appended unchanged. Serving does not
run benchmark cases or generate dashboard payloads. The downloaded Qwen GGUF is
used for text serving here; image input additionally requires an appropriate
multimodal projector and native server options.

#### Access a compute-node service through a login node

For the existing Oscar/Jupyter-style workflow, use the two repository helpers.
They reuse your `oscar` SSH alias, including any outer tunnel it already needs:

```shell
# Compute node: inside an interact GPU allocation, with project venv/conda activated
# Run from the GPU-Insights checkout. Defaults to Qwen3.8 and remote port 18080.
bash scripts/llm-interact.sh --device CUDA0

# Local Mac, terminal 1: existing outer tunnel, if required by your oscar alias
bash ~/brown_oscar.sh

# Local Mac, terminal 2: run from your local GPU-Insights checkout
bash scripts/oscar-llm-connect.sh

# Or use the direct login alias without the outer tunnel
OSCAR_HOST=oscar-direct bash scripts/oscar-llm-connect.sh

# Optional port overrides (local and remote ports need not match)
# Compute node:
LLM_PORT=18081 bash scripts/llm-interact.sh --device CUDA0
# Local Mac:
LOCAL_PORT=18082 bash scripts/oscar-llm-connect.sh
```

The compute helper requires `SLURM_JOB_ID`, an activated Python environment,
and `openssl` for key generation. It binds to the internal IPv4 address returned
by `hostname -i`, generates a fresh API key, and atomically publishes job/node/port
and credentials under `~/.cache/oscar-llm/active/current.env`. This requires the
compute and login nodes to share your home directory. Files are private (0600),
and the active directory is 0700. The helper stays in the foreground, forwards
termination to the server, and removes its connection information on normal exit,
Ctrl+C, or SIGTERM. A second registered session is rejected. After an uncatchable
termination or node failure, check that the old job/service has ended before
manually removing the stale `~/.cache/oscar-llm/active` directory.

The local helper reads metadata as data, checks that the Slurm job is RUNNING,
prints the client base URL and API key, and keeps the SSH forward in the foreground.
It also prints a copyable `claude --model '<model.key>' --settings '{...}'` command for
another terminal. The inline settings contain the current tunnel address and key
under Claude Code's `env` settings field; no shell exports or persistent settings
files are needed. Claude Code uses the server root URL (without `/v1`) for its
Anthropic-compatible Messages API. The compute helper records the configured
`model.key` (the same name passed to llama-server's `--alias`) as `MODEL_KEY` in
the connection information. Claude Code's model and Haiku/Sonnet/Opus mappings
all use that exact name, e.g. `qwen3_8_27b_ud_q6_k`. Restart older compute-node
sessions to publish this field before using the updated local helper.
It neither opens a browser nor starts an allocation. Local port conflicts fail
through `ExitOnForwardFailure`; use `LOCAL_PORT` to choose another port. Closing
the local tunnel does not stop the remote model. Metadata is published during
startup, so wait for the server's model-ready log before sending agent requests.
For a readiness check, request `/health` through the local tunnel with the printed
key in `Authorization: Bearer <key>`. The helpers do not alter your existing
Jupyter scripts, SSH configuration, or home-directory command installations.

The compute helper defaults to `--qwen38`; `--gemma --12b`, `--gemma --e2b`, or
`--config` selects another model. Launcher options such as `--parallel`,
`--ctx-per-slot`, and `--llama-server` pass through. Host, port, and authentication
are managed by the helper; native arguments after `--` are not accepted. Use
`python -m llm_bench.serve` directly for manual networking or native options.
This login-to-compute layout uses the cluster network for its final HTTP hop,
as in the Jupyter helper. The manual jump-host alternative below keeps that hop
inside SSH when the cluster permits SSH to compute nodes.

Start the service inside your allocated GPU job on the compute node. If SSH to
that node is allowed, keep the default loopback listener and run this on your
local computer (replace user and host placeholders):

```shell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -J USER@LOGIN_HOST \
  -L 127.0.0.1:18080:127.0.0.1:18080 USER@COMPUTE_HOST
```

The SSH connection terminates on the compute node; `127.0.0.1` at the far end
therefore reaches the compute-node server. The login node is only a jump host.
Use `http://127.0.0.1:18080/v1` locally. The local port can be changed independently
if occupied, e.g. `-L 127.0.0.1:18081:127.0.0.1:18080`.

If you can SSH only to the login node, and it can reach compute-node TCP ports,
bind the server to the compute node's internal IP instead:

```shell
# On the allocated compute node (COMPUTE_INTERNAL_IP must belong to this node)
python -m llm_bench.serve --qwen38 --device CUDA0 \
  --host COMPUTE_INTERNAL_IP --port 18080 -- --api-key-file /path/to/api-keys.txt

# On your local computer
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18080:COMPUTE_INTERNAL_IP:18080 USER@LOGIN_HOST
```

In this second layout the login node connects to the compute-node IP, so a
compute-node loopback listener cannot work. `--host 0.0.0.0` can also be used to
listen on all compute-node interfaces. Configure the same API key in your client
(`Authorization: Bearer <key>` for curl). This layout exposes the listener on the
cluster network, and the login-to-compute HTTP hop is not protected by the local
SSH tunnel. Prefer the jump-host layout when available. Neither layout runs the
model on the login node. See [OpenSSH forwarding options](https://man.openbsd.org/ssh).

### Run the LLM benchmark

```shell
# Run all configured coding-agent cases
python3 -m llm_bench.cli

# Use a specific llama-bench binary
python3 -m llm_bench.cli --llama-bench /path/to/llama-bench

# Run prebuilt llama-bench inside an NVIDIA CUDA Docker runtime image
python3 -m llm_bench.cli --docker

# Run one case
python3 -m llm_bench.cli --case repo_context_plan

# Pin execution and metadata to selected CUDA GPUs
python3 -m llm_bench.cli --gpu-id 0,1

# List configured cases
python3 -m llm_bench.cli --list-cases
```

By default the launcher looks for `llama-bench` in
`third_party/llama-bench/current/bin/`, then the source bootstrap build output
under `third_party/llama.cpp/build/bin/`, then `PATH`. Set
`GPU_INSIGHTS_LLAMA_BENCH` or pass `--llama-bench` to override this order.
During a run it prints the selected runtime, model path, per-case progress,
PP/TG throughput results, a summary, and finally a short import command.
`llama-bench` stderr is streamed live with a `llama-bench |` prefix so
backend/debug messages remain visible while stdout is still parsed as JSON.

Pass `--docker` to keep the small prebuilt release package while running it in
an NVIDIA CUDA runtime container that provides CUDA user-space libraries such as
`libcudart` and cuBLAS. The launcher checks Docker, the prebuilt install, and
the selected runtime image before starting benchmark cases. By default the
Docker image is selected from `third_party/llama-bench/current/BUILD-MANIFEST.json`
(`nvidia/cuda:12.6.3-runtime-ubuntu22.04` for CUDA 12 assets and
`nvidia/cuda:13.0.2-runtime-ubuntu24.04` for CUDA 13 assets). Override it with
`GPU_INSIGHTS_LLAMA_BENCH_DOCKER_IMAGE`, and override the Docker `--gpus` value
with `GPU_INSIGHTS_LLAMA_BENCH_DOCKER_GPUS`.

On CUDA hosts, the LLM launcher uses layer split by default. If `--gpu-id` is
omitted and multiple NVIDIA GPUs are visible through `nvidia-smi`, it runs
llama.cpp with all detected CUDA devices, for example `-dev CUDA0/CUDA1 -sm
layer`. Pass `--gpu-id 0` or `--gpu-id 0,1` to select devices manually; pass
`--device` only when you want to override the raw llama.cpp `-dev` value.

By default the dashboard import payload is written to `outputs/llm-bench/`, and
a sidecar `*.debug.json` file keeps the full raw llama-bench rows for debugging:

```text
LLM_RESULT_PAYLOAD_FILE=outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json
LLM_DEBUG_PAYLOAD_FILE=outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.debug.json

Import:
  python3 scripts/manage-data.py l outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json
```

Import it into the dashboard data:

```shell
python3 scripts/manage-data.py l outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json
```

Pass `--emit-base64` if you need the legacy `LLM_RESULT_PAYLOAD_B64=...` line
for copy-paste workflows.

For development without running llama.cpp:

```shell
python3 -m llm_bench.cli \
  --mock-result-file llm_bench/mock/llama-bench-qwen3_6_27b-q4.json \
  --pretty
```

## LLM API benchmark

This separate track measures the API service seen by a client, including network and
queue time. It supports OpenAI Chat Completions, OpenAI Responses, Claude Messages,
Gemini native streaming, DeepSeek, vLLM, llama.cpp, and a custom OpenAI-compatible
base URL. The existing `llm_bench.cli` track still measures local `llama-bench`
engine throughput.

Preview the request count and output token cap without contacting a service:

```shell
python -m llm_bench.api.cli --provider openai-chat --model YOUR_MODEL --dry-run
```

For a real run, select the provider and model and omit `--dry-run`. Credentials are
resolved in this order: `--api-key`, an environment variable, then a repository-local
`.env` file (or `--env-file PATH`). The default variable names are
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, and
`LLM_API_KEY` for custom and local endpoints. Local servers may run without a key.
The `.env` file is Git-ignored. A command-line key can be visible in shell history
and process listings, so the environment or `.env` form is preferable.
Sending a key to a non-loopback plain HTTP endpoint requires
`--allow-insecure-http`; loopback servers work with HTTP by default.

```shell
# Hosted API
OPENAI_API_KEY=... python -m llm_bench.api.cli --provider openai-chat --model YOUR_MODEL

# Local OpenAI-compatible API
python -m llm_bench.api.cli --provider vllm --model YOUR_MODEL \
  --base-url http://127.0.0.1:8000/v1

# Custom service that rejects stream_options
python -m llm_bench.api.cli --provider compatible --model YOUR_MODEL \
  --base-url https://example.com/v1 --no-stream-usage

# Separate concurrent load mode (decode workload only)
python -m llm_bench.api.cli --provider vllm --model YOUR_MODEL \
  --concurrency 4 --repetitions 5
```

The command writes a JSON result to `outputs/llm-api-bench/`. It stores timing,
usage, and generic failure status, without API keys, prompt text, or response text.
After reviewing the result, import its summary for the `#/llm-api` GitHub Pages
view:

```shell
python scripts/manage-data.py import-api-payload outputs/llm-api-bench/RESULT.json --dry-run
python scripts/manage-data.py import-api-payload outputs/llm-api-bench/RESULT.json
```

The default workload uses three input sizes and repeated requests, plus a separate
longer output case. Decode metrics come from server timings when available, or from
provider token usage and the visible streaming interval. Effective prefill is
estimated from the slope of token count against time to first visible text. It is
left blank when token counts, enough successful samples, or a stable slope are
unavailable. Concurrent mode runs only the longer output case and reports successful
requests per second; it does not estimate prefill. The dashboard labels server values
and client estimates separately.

## Models

| Model | Parameters | Input Size | Task | Aliases |
|-------|-----------|------------|------|---------|
| CNN | 62K | 3×32×32 | Classification | `cnn` |
| ResNet-50 | 23.5M | 3×32×32 | Classification | `resnet50`, `ResNet-50`, `ResNet50` |
| ViT-Base/16 | 85.8M | 3×224×224 | Classification | `vit`, `vit-base` |
| UNet | 31.0M | 3×256×256 | Segmentation | `unet` |
| DDPM | 62.3M | 3×64×64 | Diffusion (noise prediction) | `ddpm` |

## CLI Arguments

### Smart launcher (`main_auto.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `-mt`, `--model` | Model to benchmark. Omit to run `resnet50`, `vit`, `unet`, `ddpm` | all four major models |
| `-s`, `--size` | Data size in MB | `1024` |
| `-e`, `--epochs` | Training epochs | `5` |
| `-d`, `--device` | Device: `auto`, `cuda`, `mps`, `npu`, `musa`, `tpu` | `auto` |
| `-gpu`, `--gpu_id` | CUDA GPU ids, e.g. `all` or `0,1` | `all` |
| `-dt`, `--dtype` | Run a single precision instead of auto-selection | auto |
| `--no-abs` | Disable auto batch size | off |
| `-bs`, `--batch` | Batch size override | `0` |
| `--single-process` | Disable automatic CUDA DDP | off |
| `--dry-run` | Print launch plan without running | off |

Default smart-launch behavior:

- Auto-detects the backend using the existing backend priority.
- Enables ABS by default unless `-bs` or `--no-abs` is provided.
- Runs `BF16 + FP32` on BF16-capable devices, otherwise `FP16 + FP32` when AMP is supported.
- Automatically switches to CUDA DDP when multiple CUDA GPUs are visible and the model supports DDP.
- Runs `resnet50`, `vit`, `unet`, and `ddpm` in order when `--model` is omitted.
- Prints a final `RESULT_PAYLOAD_B64=...` line after the human summary so benchmark results can be copied into an update script.

### Smart launcher output payload

At the end of a real `main_auto.py` run, the launcher prints one machine-readable line:

```text
RESULT_PAYLOAD_B64=<base64-json>
```

The decoded JSON includes:

- `schema_version` and `generated_at`
- `source` launcher metadata
- `host` metadata such as `vendor`, `architecture`, `device`, `memory`, `platform`, and `driver_runtime`
- `benchmarks`, one normalized entry per model

Payload compatibility notes:

- `FP32` fills `fp32` / `fp32bs`
- `FP16` fills `fp16` / `fp16bs`
- `BF16` is intentionally exported into `fp16` / `fp16bs` for compatibility with the current dashboard data schema
- per-precision status metadata is preserved in the payload for downstream tooling

### Legacy entrypoints (`main.py`, `main_ddp.py`, `main_tpu.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `-mt`, `--model` | Model to benchmark | `resnet50` |
| `-s`, `--size` | Data size in MB | `1024` |
| `-e`, `--epochs` | Training epochs | `5` |
| `-dt`, `--data_type` | Precision: `FP32`, `FP16`, `BF16` | `FP32` |
| `-bs`, `--batch` | Batch size (0 = model default) | `0` |
| `-abs`, `--auto_batch_size` | Auto batch size via calibration table | off |
| `-d`, `--device` | Device: `auto`, `cuda`, `mps`, `npu`, `musa`, `tpu` | `auto` |
| `-gpu`, `--gpu_id` | GPU ID(s), e.g. `0` or `0,1` | `0` |
| `-cudnn`, `--cudnn_benchmark` | Enable cuDNN benchmark mode | off |

## Makefile Targets

```shell
make smart      # Smart launcher (set MODEL=vit, MODEL=unet, etc.)
make tpu        # ResNet50 on TPU single-core
make tpu-multi  # ResNet50 on TPU 8-core
make calibrate  # Run memory calibration
make docs       # Build visualization website
make docs-dev   # Start docs dev server
make help       # Show current targets and variables
```

## Device Backends

| Backend | Hardware | Requirements |
|---------|----------|-------------|
| `cuda` | NVIDIA GPUs | PyTorch with CUDA |
| `mps` | Apple Silicon | PyTorch ≥ 1.12, macOS |
| `npu` | Huawei Ascend | `torch_npu` |
| `musa` | Moore Threads | `torch_musa` |
| `tpu` | Google TPU | `torch_xla` |
| `auto` | Auto-detect | Tries CUDA → NPU → MUSA → MPS |

Use `--device` to select a specific backend, or leave as `auto` (default).

## Environment Probe Script

Use the dedicated probe helper to inspect the normalized host metadata used by the smart launcher payload:

```shell
python3 scripts/probe_benchmark_env.py --pretty
python3 scripts/probe_benchmark_env.py -d cuda -gpu 0
```

On NVIDIA systems, the script prefers NVML (`pynvml`) for device name, total memory, and driver version, then combines that with CUDA compute capability from PyTorch to map the GPU architecture name.

## DDP Multi-GPU Training

```shell
# Smart launcher
make smart MODEL=vit
python3 main_auto.py -mt resnet50

# 2 GPUs (default)
torchrun --nproc_per_node=2 main_ddp.py -mt resnet50 -s 512 -e 2 -abs -dt FP16

# DDP with ViT
torchrun --nproc_per_node=4 main_ddp.py -mt vit -s 512 -e 2 -bs 32 -dt FP16
```

## Auto Batch Size (ABS)

When `-abs` is enabled, the benchmark automatically selects a batch size based on an NVML-calibrated memory profile table. The selection logic:

1. Queries the device's total VRAM
2. Applies a 10% safety margin (90% usable)
3. Looks up pre-measured `(model, dtype)` peak memory data from the calibration table
4. Picks the largest batch size whose peak memory fits within the usable budget
5. Falls back to the model's default batch size if no calibration data exists

The calibration table lives in `benchmark/calibration.py`. To generate calibration data for your GPU, see [Memory Calibration](#memory-calibration) below.

**Backend support:** CUDA is the primary target. NPU/MUSA use CUDA calibration data as a proxy. MPS/TPU fall back to model defaults.

## Memory Calibration

The calibration tool measures real peak VRAM via NVML (`pynvml`) during short training runs:

```shell
# Install dependency
pip install pynvml

# Full calibration (all models × all precisions)
python calibrate_memory.py

# Specific model/precision
python calibrate_memory.py -mt resnet50 -dt FP16

# Custom batch sizes
python calibrate_memory.py -mt vit -dt FP32 -bs 8,16,32,64

# JSON output (for programmatic use)
python calibrate_memory.py --json

# Specify GPU
python calibrate_memory.py -gpu 1
```

After running, paste the output into `benchmark/calibration.py` `CALIBRATION_TABLE`.

## How to Understand Results

The benchmark evaluates hardware training throughput under a fixed workload. Output is a **score** representing compute performance — higher is better. Scores are affected by compute capability, memory bandwidth, and PCIe/interconnect bandwidth.

## Results

For a visual dashboard, visit the [GPU-Insights Dashboard](https://xiaoran007.github.io/GPU-Insights/).

## Data Management

```shell
# Import the final smart-launcher payload directly into the dashboard data
python3 scripts/manage-data.py import-payload 'RESULT_PAYLOAD_B64=...'

# Decode a payload for inspection
python3 scripts/manage-data.py decode-payload 'RESULT_PAYLOAD_B64=...' --pretty

# Import an LLM inference payload
python3 scripts/manage-data.py l outputs/llm-bench/llm-bench-20260618-211527-qwen3_6_27b_q4.json

# Validate benchmark data
python3 scripts/manage-data.py validate

# Show statistics
python3 scripts/manage-data.py stats

# Add a benchmark entry
python3 scripts/manage-data.py add \
  --vendor nvidia --architecture Ada \
  --device "RTX 4090" --memory "24GB" \
  --platform "Linux" --fp32 24000 --fp32bs 512 \
  --fp16 43000 --fp16bs 1024

# Migrate version field
python3 scripts/manage-data.py migrate-version
```

The smart launcher payload is designed to be script-friendly. The recommended update flow is:

```shell
python3 main_auto.py
python3 scripts/manage-data.py import-payload '<paste RESULT_PAYLOAD_B64 value here>'
```

`import-payload` accepts either the full `RESULT_PAYLOAD_B64=...` line or the raw Base64 value. Failed models with no successful precision result are skipped automatically. When `model + vendor + architecture + device + memory` all match an existing entry, that entry is updated in place; otherwise a new entry is appended. Exact duplicate payload rows are treated as no-op updates and skipped.

## Project Structure

```
├── main_auto.py         # Smart launcher with auto device / precision / DDP planning
├── main.py              # Single-device entry point
├── main_ddp.py          # DDP multi-GPU entry point
├── main_tpu.py          # TPU entry point
├── calibrate_memory.py  # NVML memory calibration tool
├── scripts/
│   ├── probe_benchmark_env.py  # Host/device metadata probe used by main_auto.py payload export
│   ├── bootstrap-llm-oscar.sh  # Link models/llm to Oscar persistent storage
│   ├── bootstrap-llm-autodl.sh # Link models/llm to AutoDL persistent storage
│   ├── bootstrap-llm-colab.sh  # Copy Drive GGUF into Colab local runtime cache
│   ├── bootstrap-llm-vast.sh   # Prepare a Vast.ai ephemeral instance workspace
│   ├── build-llama-bench-release.sh
│   ├── package-llama-bench-release.sh
│   ├── download-llm-model.py   # Fixed Qwen3.6-27B Q4_K_M GGUF downloader
│   └── manage-data.py          # Benchmark data management
├── Makefile             # Smart launcher and developer convenience targets
├── llm_bench/           # Standalone llama.cpp LLM inference benchmark adapter
├── benchmark/
│   ├── Bench.py         # Orchestrator
│   ├── cli.py           # Unified CLI parsing
│   ├── scoring.py       # Scoring system
│   ├── calibration.py   # Calibration table + auto batch size logic
│   ├── models/          # BenchModel implementations
│   │   ├── base.py      # BenchModel ABC
│   │   ├── cnn.py       # Simple CNN (62K params)
│   │   ├── resnet50.py  # ResNet50 (23.5M params)
│   │   ├── vit.py       # ViT-Base/16 (85.8M params)
│   │   ├── unet.py      # UNet segmentation (31.0M params)
│   │   └── ddpm.py      # DDPM diffusion (62.3M params)
│   ├── devices/         # DeviceBackend implementations
│   │   ├── base.py      # DeviceBackend ABC
│   │   ├── cuda_device.py
│   │   ├── macos_info.py
│   │   ├── mps_device.py
│   │   ├── npu_device.py
│   │   ├── musa_device.py
│   │   └── tpu_device.py
│   ├── runners/         # Training runners
│   │   ├── common.py    # Shared training utilities
│   │   ├── single_runner.py
│   │   └── ddp_runner.py
│   └── data/            # Dataset utilities
└── docs/                # GitHub Pages dashboard
```

## License

See [LICENSE](LICENSE).
