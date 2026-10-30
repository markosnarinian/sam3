# Fork documentation

This is [markosnarinian/sam3](https://github.com/markosnarinian/sam3), a fork of
[Meta's SAM 3](https://github.com/facebookresearch/sam3). This section describes
the fork's current differences; the SAM 3 overview and upstream capabilities
remain below. The comparison was checked against upstream `main` at
[`f1647e8`](https://github.com/facebookresearch/sam3/commit/f1647e8)
and this fork's implementation, not just commit titles.

Contents:

- [Scope and upstream compatibility](#scope-and-upstream-compatibility)
- [Portable single-image inference](#portable-single-image-inference)
- [Locked installation and optional dependencies](#locked-installation-and-optional-dependencies)
- [Notebooks, training, and evaluation workflows](#notebooks-training-and-evaluation-workflows)
- [Orb development environment](#orb-development-environment)
- [Development checks and documentation maintenance](#development-checks-and-documentation-maintenance)

## Scope and upstream compatibility

The fork adds portable single-image execution, reproducible `uv` environments,
updated setup instructions, and snapshot-ready Amp orb development. It does not
introduce new model weights, segmentation algorithms, a hosted service, a new
CLI, or keyboard shortcuts. Upstream checkpoint access and licensing still
apply; see [Getting Started](#getting-started) and [License](LICENSE).

Text/box image prompts, prompt reset, confidence filtering, embedding reuse,
video tracking, SAM 3.1 multiplexing, training, and evaluation are upstream
capabilities, not fork inventions. Several commits have different hashes but
the same patches upstream: caller-mask video conditioning (`add_mask`), forced
tracker propagation, streamed frame-output eviction, programmatic object-removal
flags, refined-object broadcasting, exception-handler fixes, and type-suppression
maintenance. Session cleanup and autocast restoration are also already upstream.
They are not additional fork-only APIs. Likewise, references in shared commit
messages to instance caps, seed-frame helpers, reverse box passes, or
`--prompts-json` describe external integration code, not new entry points shipped
by this repository. For upstream video usage, see [Video Usage](#video-usage-cuda)
and [SAM 3.1 release notes](RELEASE_SAM3p1.md).

## Portable single-image inference

`build_sam3_image_model(device=None)` now selects the available device at call
time in **CUDA → Apple MPS → CPU** order. Explicit strings (including indexed
CUDA devices such as `"cuda:1"`) and `torch.device` values are honored.
`Sam3Processor(model)` defaults to the model's device instead of hard-coded CUDA;
if explicitly overriding its `device`, keep it consistent with the model.

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()  # Or device="mps", "cpu", "cuda:0"
processor = Sam3Processor(model)  # resolution=1008, confidence_threshold=0.5
```

Use the existing [Basic Usage](#basic-usage) examples for RGB image loading,
text prompts, independent labels, output shapes, and offline checkpoint loading.
Positive/negative boxes use the existing
`processor.add_geometric_prompt(box, label, state)` API: `box` is normalized
`[center_x, center_y, width, height]`, with `label=True` for positive and `False`
for negative. The checkpoint format and output contract are unchanged; tensors
remain on the model's device, so move them to CPU before NumPy conversion.

Portability fixes remove image-path dependencies on CUDA-only video/Triton and
training imports, use the standard-library resource loader for the bundled BPE
vocabulary, allocate geometry tensors on their actual device, and move cached
position encodings to the input device. MPS handles empty geometric prompts and
padding checks without unsupported asynchronous assertions. The MLP uses normal
PyTorch operations for CPU/MPS, float32, and gradient-enabled execution, retaining
the fused CUDA path for inference with bfloat16 inputs or CUDA autocast.

**Limits:** MPS requires a compatible Apple Silicon Mac and available PyTorch
MPS backend. CPU/MPS use float32 by default; do not wrap them in CUDA autocast.
`PYTORCH_ENABLE_MPS_FALLBACK` is not required. `compile=False` is the default;
`compile=True` raises `ValueError` on non-CUDA image devices. This support is
scoped to single-image inference, not a promise that all batched, interactive,
video, evaluation, or training paths work on MPS/CPU. Video and training still
require CUDA. No new accuracy or speed guarantees are implied.

## Locked installation and optional dependencies

Follow [Installation](#installation) to clone this fork and run
`uv sync --locked`. The fork replaces the documented Conda/pip workflow with
an editable `.venv`, committed `uv.lock`, and `.python-version` pin. Python is
restricted to **3.12** (`>=3.12,<3.13`), rather than upstream's broader package
metadata. The lock constrains **PyTorch 2.10.0 / torchvision 0.25.0**; Linux and
Windows select CUDA 12.8 wheels, while macOS selects PyPI wheels. CUDA wheels
can also run CPU inference without an NVIDIA GPU; CUDA execution requires a
compatible NVIDIA driver. Existing Conda environments are not migrated in place.

Core image dependencies now explicitly include torch, torchvision, einops,
Pillow, psutil, and pycocotools. Optional environments are selected as follows:

| Selection | Purpose |
| --- | --- |
| No extra/group | Basic image API; no dependency groups enabled by default |
| `--extra notebooks` | Jupyter, visualization, and example dependencies |
| `--extra train` | Training dependencies |
| `--group dev` | pytest, coverage, formatters, and development utilities; replaces the old `dev` extra |
| `--group eval` | Dataset preparation and evaluation tools |
| `--group fa3` | FlashAttention-3 3.0.0 for supported Hopper CUDA hardware |
| `--group cc-torch` | Locally built CUDA connected-components extension |

Combine all required flags in one sync command: `uv sync` removes packages not
selected by that command. `cc-torch` is pinned to source revision
`446e9360ed10d3c8d03925ccb5f220ccb62009de`, with isolated build dependencies
including torch 2.10.0; it still requires a system CUDA 12.8 toolkit (`nvcc`) and
compatible C++ compiler. Drivers, toolkits, FFmpeg, weights, datasets, and access
credentials are not provided by the Python lockfile. These environments do not
constitute a deployment service or production GPU provisioning.

## Notebooks, training, and evaluation workflows

Run local notebooks from this checkout using the environment in
[Examples](#examples). The single-image predictor notebook uses automatic device
selection, CUDA-only autocast when available, and the bundled vocabulary rather
than a hand-built path. Other example/Colab installation cells now use `uv`, but
many still install **upstream** from GitHub and assume CUDA 12.8. To exercise this
fork, use this checkout's locked environment instead; the Colab cells are not a
locked or portable fork installation. The `notebooks` extra excludes `decord` on
macOS, so video notebooks that import it are not covered by Mac image support.

The [SAM 3 Agent notebook](examples/sam3_agent.ipynb) documents optional vLLM
serving in a separate `.venv-vllm` using `uv venv --python 3.12` and
`uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto`.
Keep that tightly coupled torch/CUDA stack separate from SAM 3's lockfile;
external LLM APIs do not need a local vLLM server. Follow the notebook for its
server/model options. The separate environment is ignored by Git.

The existing [training guide](README_TRAIN.md) and evaluation guides for
[Gold](scripts/eval/gold/README.md), [Silver](scripts/eval/silver/README.md), and
[VEval](scripts/eval/veval/README.md) now use `uv sync`/`uv run`. They remain the
source for dataset preparation, paths, configuration, and local/SLURM commands;
the fork changes environment setup, not those algorithms or benchmark contracts.

## Orb development environment

On Debian/Ubuntu-based Amp orbs, `.agents/setup` installs missing system packages
(`build-essential`, `ca-certificates`, `curl`, `ffmpeg`, `libgl1`, `libglib2.0-0`)
with sudo, installs uv 0.12.9 if uv is absent, and runs
`uv sync --locked --group dev --extra train`. Existing uv installations are
reused. Amp can snapshot the prepared environment; `.agents/resume` only checks
that `.venv/bin/python` exists and reports its version, without reinstalling.
If it is missing or dependencies need repair, rerun `.agents/setup`.

Use `uv run --no-sync` to preserve the prepared feature set. Notebooks, evaluation
groups, GPU extensions/toolchains, drivers, checkpoints, and Hugging Face
credentials are not provisioned by these hooks. CPU tests can run without GPU
access, but GPU inference/training still needs suitable hardware. Generated orb
portal metadata in `.amp/portals/` is ignored; no application service or portal
is configured by these hooks. See [Contributing](CONTRIBUTING.md#amp-orbs).

## Development checks and documentation maintenance

See [Contributing](CONTRIBUTING.md#development-environment) for dependency updates
(`uv lock`), lock consistency (`uv lock --check`), and formatting. With the locked
development/training environment installed, the full repository test command is:

```bash
uv run --no-sync pytest -q test
```

Pass `test` explicitly: pytest's configured `testpaths` says `tests`, but the
repository suite lives in `test/`. The fork adds full-model image execution,
MLP float32 correctness, and position-cache device regression tests. See
[Checking image inference](#checking-image-inference) for targeted checks and
`SAM3_TEST_CHECKPOINT`. Without that optional authorized checkpoint, image tests
use initialized random weights: they validate execution and output contracts,
not segmentation quality. Unavailable MPS/CUDA devices are skipped.

Root [AGENTS.md](AGENTS.md) requires documentation changes alongside changes to
documented behavior or workflows, synchronized contents links, and removal of
obsolete claims. Implementation-only changes do not require README churn.

# SAM 3: Segment Anything with Concepts

Meta Superintelligence Labs

[Nicolas Carion](https://www.nicolascarion.com/)\*,
[Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en)\*,
[Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en)\*,
[Shoubhik Debnath](https://scholar.google.com/citations?user=fb6FOfsAAAAJ&hl=en)\*,
[Ronghang Hu](https://ronghanghu.com/)\*,
[Didac Suris](https://www.didacsuris.com/)\*,
[Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en)\*,
[Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en)\*,
[Haitham Khedr](https://hkhedr.com/)\*, Andrew Huang,
[Jie Lei](https://jayleicn.github.io/),
[Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en),
[Baishan Guo](https://scholar.google.com/citations?user=BC5wDu8AAAAJ&hl=en),
Arpit Kalla, [Markus Marks](https://damaggu.github.io/),
[Joseph Greer](https://scholar.google.com/citations?user=guL96CkAAAAJ&hl=en),
Meng Wang, [Peize Sun](https://peizesun.github.io/),
[Roman Rädle](https://scholar.google.com/citations?user=Tpt57v0AAAAJ&hl=en),
[Triantafyllos Afouras](https://www.robots.ox.ac.uk/~afourast/),
[Effrosyni Mavroudi](https://scholar.google.com/citations?user=vYRzGGEAAAAJ&hl=en),
[Katherine Xu](https://k8xu.github.io/)°,
[Tsung-Han Wu](https://patrickthwu.com/)°,
[Yu Zhou](https://yu-bryan-zhou.github.io/)°,
[Liliane Momeni](https://scholar.google.com/citations?user=Lb-KgVYAAAAJ&hl=en)°,
[Rishi Hazra](https://rishihazra.github.io/)°,
[Shuangrui Ding](https://mark12ding.github.io/)°,
[Sagar Vaze](https://sgvaze.github.io/)°,
[Francois Porcher](https://scholar.google.com/citations?user=LgHZ8hUAAAAJ&hl=en)°,
[Feng Li](https://fengli-ust.github.io/)°,
[Siyuan Li](https://siyuanliii.github.io/)°,
[Aishwarya Kamath](https://ashkamath.github.io/)°,
[Ho Kei Cheng](https://hkchengrex.com/)°,
[Piotr Dollar](https://pdollar.github.io/)†,
[Nikhila Ravi](https://nikhilaravi.com/)†,
[Kate Saenko](https://ai.bu.edu/ksaenko.html)†,
[Pengchuan Zhang](https://pzzhang.github.io/pzzhang/)†,
[Christoph Feichtenhofer](https://feichtenhofer.github.io/)†

\* core contributor, ° intern, † project lead, order is random within groups

[[`Paper`](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)]
[[`Project`](https://ai.meta.com/sam3)]
[[`Demo`](https://segment-anything.com/)]
[[`Blog`](https://ai.meta.com/blog/segment-anything-model-3/)]
[[`BibTeX`](#citing-sam-3)]

![SAM 3 architecture](assets/model_diagram.png?raw=true) SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor [SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. It achieves 75-80% of human performance on our new [SA-CO benchmark](https://github.com/facebookresearch/sam3?tab=readme-ov-file#sa-co-dataset) which contains 270K unique concepts, over 50 times more than existing benchmarks.

This breakthrough is driven by an innovative data engine that has automatically annotated over 4 million unique concepts, creating the largest high-quality open-vocabulary segmentation dataset to date. In addition, SAM 3 introduces a new model architecture featuring a presence token that improves discrimination between closely related text prompts (e.g., “a player in white” vs. “a player in red”), as well as a decoupled detector–tracker design that minimizes task interference and scales efficiently with data.

<p align="center">
  <img src="assets/dog.gif" width=380 />
  <img src="assets/player.gif" width=380 />
</p>

## Latest updates

**03/27/2026 -- SAM 3.1 Object Multiplex is released. It introduces a shared-memory approach for joint multi-object tracking that is significantly faster without sacrificing accuracy.**

- A new suite of improved model checkpoints (denoted as **SAM 3.1**) are released on [Hugging Face](https://huggingface.co/facebook/sam3.1). See [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md) for full details.
  * To use the new SAM 3.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.

> [!IMPORTANT]
> **Environment management has moved from Conda and pip to
> [`uv`](https://docs.astral.sh/uv/).** The committed `uv.lock` is now the
> source of truth for reproducible development environments. It pins CPython
> 3.12, PyTorch 2.10.0, and torchvision 0.25.0. Linux/Windows use CUDA 12.8
> wheels; macOS uses PyPI wheels with Metal/MPS support.
> Existing Conda environments are not updated in place: install `uv` and run
> `uv sync --locked` to create the repository's `.venv`. Notebook, training,
> evaluation, development, and optional acceleration dependencies are selected
> with the extras and groups documented below. NVIDIA drivers, the CUDA toolkit
> required to compile `cc-torch`, and other system tools remain outside `uv`.

## Installation

### Prerequisites

- For single-image inference: an NVIDIA CUDA GPU, an Apple Silicon Mac with
  Metal/MPS support, or CPU. The same checkpoints and image API work on all three.
- NVIDIA GPU users need a driver supporting CUDA 12.8. Video inference and
  training still require CUDA; MPS support is scoped to single-image inference.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

1. **Clone the repository:**

```bash
git clone https://github.com/markosnarinian/sam3.git
cd sam3
```

2. **Create the environment and install SAM 3:**

```bash
uv sync --locked
```

`uv` installs Python 3.12 when needed, creates `.venv`, and installs the package
in editable mode. PyTorch and torchvision use the CUDA 12.8 index on Linux/Windows
and PyPI on macOS, where MPS is included without CUDA packages. CUDA wheels also
support CPU inference on machines without an NVIDIA GPU. No video, training, or
optional acceleration groups are needed for the basic image API.
On a Mac, check MPS with
`uv run python -c "import torch; print(torch.backends.mps.is_available())"`.
Run commands in the environment with `uv run`, or activate it:

```bash
source .venv/bin/activate
```

3. **Install additional dependencies for example notebooks or development:**

```bash
# For running example notebooks
uv sync --locked --extra notebooks

# For evaluation and dataset preparation
uv sync --locked --group eval

# For development
uv sync --locked --group dev --extra train
```

4. **Install optional CUDA dependencies for faster inference (not for Mac/CPU):**
```bash
# FlashAttention-3 (for supported Hopper GPUs)
uv sync --locked --group fa3

# CUDA connected-components extension
uv sync --locked --group cc-torch
```

`cc-torch` is built locally and requires a CUDA 12.8 toolkit (including `nvcc`)
and a compatible C++ compiler. Its source revision and Python build dependencies
are locked, but the system CUDA toolchain is not managed by `uv`. Combine group
and extra flags in one `uv sync` command when you need multiple feature sets.
Because `uv sync` makes `.venv` exactly match the selected features, repeat all
needed flags when changing the environment, for example:

```bash
uv sync --locked --extra notebooks --extra train --group dev --group eval
```

## Getting Started

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Usage

The image builder automatically selects **CUDA → MPS → CPU**. The processor
uses the model's device, so existing image code needs no device-specific changes.
To override selection, pass `device="mps"`, `device="cpu"`, `device="cuda:0"`, or a
`torch.device` to `build_sam3_image_model`. MPS and CPU use float32 by default;
do not enter a CUDA autocast context on those backends. `compile=True` is currently
supported only on CUDA. `PYTORCH_ENABLE_MPS_FALLBACK` is not required.

```python
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>").convert("RGB")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

Outputs stay on the model's device: masks have shape `(N, 1, H, W)`, boxes are
pixel coordinates `[x0, y0, x1, y1]` with shape `(N, 4)`, and scores have shape
`(N,)`. Use `.cpu()` before converting to NumPy or exporting annotations.

For automatic annotation with multiple independent labels, compute the image
embedding once and reuse it:

```python
annotations = {}
for label in ["person", "car", "bicycle"]:
    processor.reset_all_prompts(inference_state)
    output = processor.set_text_prompt(state=inference_state, prompt=label)
    annotations[label] = {
        "masks": output["masks"].cpu().numpy(),
        "boxes": output["boxes"].float().cpu().numpy(),
        "scores": output["scores"].float().cpu().numpy(),
    }
```

Call `set_image` again for the next image. Adjust filtering with
`Sam3Processor(model, confidence_threshold=0.5)`. Empty results have `N = 0`.
To use an already-downloaded checkpoint without contacting Hugging Face:

```python
model = build_sam3_image_model(checkpoint_path="/path/to/sam3.pt", load_from_HF=False)
processor = Sam3Processor(model)
```

### Video Usage (CUDA)

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

### Checking image inference

```bash
uv run --locked --group dev pytest test/test_image_inference.py
# Optionally exercise the same tests with your authorized checkpoint:
SAM3_TEST_CHECKPOINT=/path/to/sam3.pt uv run --locked --group dev pytest test/test_image_inference.py
```

The tests exercise the full image model on CPU and available accelerators,
including text prompts, positive/negative box prompts, and empty results. Without
a checkpoint they use initialized random weights to check execution, not
segmentation accuracy. CUDA cases are skipped when no NVIDIA GPU is available.

## Examples

The `examples` directory contains notebooks demonstrating how to use SAM3 with
various types of prompts:

- [`sam3_image_predictor_example.ipynb`](examples/sam3_image_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text and visual box prompts on images.
- [`sam3_video_predictor_example.ipynb`](examples/sam3_video_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text prompts on videos, and doing
  further interactive refinements with points.
- [`sam3_image_batched_inference.ipynb`](examples/sam3_image_batched_inference.ipynb)
  : Demonstrates how to run batched inference with SAM 3 on images.
- [`sam3_agent.ipynb`](examples/sam3_agent.ipynb): Demonsterates the use of SAM
  3 Agent to segment complex text prompt on images.
- [`saco_gold_silver_vis_example.ipynb`](examples/saco_gold_silver_vis_example.ipynb)
  : Shows a few examples from SA-Co image evaluation set.
- [`saco_veval_vis_example.ipynb`](examples/saco_veval_vis_example.ipynb) :
  Shows a few examples from SA-Co video evaluation set.

There are additional notebooks in the examples directory that demonstrate how to
use SAM 3 for interactive instance segmentation in images and videos (SAM 1/2
tasks), or as a tool for an MLLM, and how to run evaluations on the SA-Co
dataset.

To run the Jupyter notebook examples:

```bash
# Make sure you have the notebooks dependencies installed
uv sync --locked --extra notebooks

# Start Jupyter notebook
uv run jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## Model

SAM 3 consists of a detector and a tracker that share a vision encoder. It has 848M parameters. The
detector is a DETR-based model conditioned on text, geometry, and image
exemplars. The tracker inherits the SAM 2 transformer encoder-decoder
architecture, supporting video segmentation and interactive refinement.

## Image Results

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="3" style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th colspan="3" style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">Instance Segmentation</th>
      <th colspan="5" style="text-align: center; padding: 12px 20px">Box Detection</th>
    </tr>
    <tr>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">SA-Co/Gold</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">COCO</th>
      <th style="text-align: center; padding: 12px 20px">SA-Co/Gold</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP<sub>o</sub>
</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Human</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">72.8</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">74.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">OWLv2*</td>
      <td style="text-align: center; padding: 10px 20px; color: #999">29.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px; color: #999">43.4</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">24.6</td>
      <td style="text-align: center; padding: 10px 20px; color: #999">30.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px; color: #999">45.5</td>
      <td style="text-align: center; padding: 10px 20px">46.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">23.9</td>
      <td style="text-align: center; padding: 10px 20px">24.5</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">DINO-X</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">38.5</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">21.3</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">52.4</td>
      <td style="text-align: center; padding: 10px 20px">56.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">22.5</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Gemini 2.5</td>
      <td style="text-align: center; padding: 10px 20px">13.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">13.0</td>
      <td style="text-align: center; padding: 10px 20px">16.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">14.4</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">37.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">48.5</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">54.1</td>
      <td style="text-align: center; padding: 10px 20px">40.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">53.6</td>
      <td style="text-align: center; padding: 10px 20px">56.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">55.7</td>
      <td style="text-align: center; padding: 10px 20px">55.7</td>
    </tr>
  </tbody>
</table>

<p style="text-align: center; margin-top: 10px; font-size: 0.9em; color: #ddd;">* Partially trained on LVIS, AP<sub>o</sub> refers to COCO-O accuracy</p>

</div>

## Video Results

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SA-V test</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">YT-Temporal-1B test</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SmartGlasses test</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVVIS test</th>
      <th style="text-align: center; padding: 12px 20px">BURST test</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">mAP</th>
      <th style="text-align: center; padding: 12px 20px">HOTA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Human</td>
      <td style="text-align: center; padding: 10px 20px">53.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">70.5</td>
      <td style="text-align: center; padding: 10px 20px">71.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">78.4</td>
      <td style="text-align: center; padding: 10px 20px">58.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">72.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">30.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">58.0</td>
      <td style="text-align: center; padding: 10px 20px">50.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">69.9</td>
      <td style="text-align: center; padding: 10px 20px">36.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">63.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">36.3</td>
      <td style="text-align: center; padding: 10px 20px">44.5</td>
    </tr>
  </tbody>
</table>
</div>

## SA-Co Dataset

We release 2 image benchmarks, [SA-Co/Gold](scripts/eval/gold/README.md) and
[SA-Co/Silver](scripts/eval/silver/README.md), and a video benchmark
[SA-Co/VEval](scripts/eval/veval/README.md). The datasets contain images (or videos) with annotated noun phrases. Each image/video and noun phrase pair is annotated with instance masks and unique IDs of each object matching the phrase. Phrases that have no matching objects (negative prompts) have no masks, shown in red font in the figure. See the linked READMEs for more details on how to download and run evaluations on the datasets.

* HuggingFace host: [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver) and [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)
* Roboflow host: [SA-Co/Gold](https://universe.roboflow.com/sa-co-gold), [SA-Co/Silver](https://universe.roboflow.com/sa-co-silver) and [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

![SA-Co dataset](assets/sa_co_dataset.jpg?raw=true)

## Development

To set up the development environment:

```bash
uv sync --locked --group dev --extra train
```

To format the code:

```bash
uv run ufmt format .
```

## Contributing

See [contributing](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgements

We would like to thank the following people for their contributions to the SAM 3 project: Alex He, Alexander Kirillov,
Alyssa Newcomb, Ana Paula Kirschner Mofarrej, Andrea Madotto, Andrew Westbury, Ashley Gabriel, Azita Shokpour,
Ben Samples, Bernie Huang, Carleigh Wood, Ching-Feng Yeh, Christian Puhrsch, Claudette Ward, Daniel Bolya,
Daniel Li, Facundo Figueroa, Fazila Vhora, George Orlin, Hanzi Mao, Helen Klein, Hu Xu, Ida Cheng, Jake Kinney,
Jiale Zhi, Jo Sampaio, Joel Schlosser, Justin Johnson, Kai Brown, Karen Bergan, Karla Martucci, Kenny Lehmann,
Maddie Mintz, Mallika Malhotra, Matt Ward, Michelle Chan, Michelle Restrepo, Miranda Hartley, Muhammad Maaz,
Nisha Deo, Peter Park, Phillip Thomas, Raghu Nayani, Rene Martinez Doehner, Robbie Adkins, Ross Girshik, Sasha
Mitts, Shashank Jain, Spencer Whitehead, Ty Toledano, Valentin Gabeur, Vincent Cho, Vivian Lee, William Ngan,
Xuehai He, Yael Yungster, Ziqi Pang, Ziyi Dou, Zoe Quake.

## Citing SAM 3

If you use SAM 3 or the SA-Co dataset in your research, please use the following BibTeX entry.

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```
