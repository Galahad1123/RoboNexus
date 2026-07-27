# RoboNexus environments

RoboNexus uses separate environments for components with incompatible Python
and model dependencies. Run all commands from the repository root. The
environment files contain portable runtime dependencies; third-party source
trees and model weights are installed separately so that local editable paths
and large files never leak into a Conda specification.

## Host requirements

- Linux x86_64
- NVIDIA GPU with at least 24 GB VRAM for the tested model combination
- NVIDIA driver compatible with CUDA 12.8
- Conda and Git
- `uv` for `multitask_dit_policy`

The verified development machine uses an RTX 3090. Before creating an
environment, prevent ROS and user-site packages from leaking into it:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

## CALVIN

CALVIN is pinned to Python 3.8 because of its legacy dependency stack.

```bash
conda env create -f envs/calvin.yml
conda activate robonexus-calvin

cd calvin
export CALVIN_ROOT="$(pwd)"
export PATH="$CONDA_PREFIX/bin:$PATH"
bash install.sh
cd ..

python -m pip check
python -c "import calvin_env, numpy, OpenGL, pyrender, tacto; print('CALVIN: OK')"
```

The repository is expected at commit
`fa03f01f19c65920e18cf37398a9ce859274af76`, including its submodules.
`calvin/install.sh` contains the required CMake correction. Keep
`numpy==1.23.5`; newer NumPy releases are incompatible with parts of this
stack.

## Qwen3-VL

```bash
conda env create -f envs/qwen3vl.yml
conda activate robonexus-qwen

python -m pip check
python -c "import torch, transformers, qwen_agent; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Download the model separately because weights are not environment
dependencies:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_dir='weights/Qwen3-VL-4B-Instruct')"
```

Minimal weight-loading check:

```bash
python - <<'PY'
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

path = "weights/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(
    path,
    dtype=torch.bfloat16,
    device_map="auto",
)
print(type(model).__name__, type(processor).__name__, model.device)
PY
```

## SAM

The current first-stage segmentation environment uses the original Segment
Anything ViT-B implementation. It can later be replaced by SAM2 without
changing the environment installation contract.

```bash
conda env create -f envs/segmentation.yml
conda activate robonexus-sam

python -m pip install -e RoboNexus-Agent/third_party/segment-anything
python -m pip check
```

The Segment Anything checkout is pinned to
`dca509fe793f601edb92606367a655c15ac00fdf`. Download the checkpoint outside
Conda:

```bash
mkdir -p RoboNexus-Agent/third_party/segment-anything/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -O RoboNexus-Agent/third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth
```

Verify the installation:

```bash
python - <<'PY'
import torch
from segment_anything import SamPredictor, sam_model_registry

path = "RoboNexus-Agent/third_party/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
model = sam_model_registry["vit_b"](checkpoint=path).cuda().eval()
predictor = SamPredictor(model)
print(type(predictor).__name__, next(model.parameters()).device)
PY
```

## Cutie

```bash
conda env create -f envs/tracking.yml
conda activate robonexus-cutie

python -m pip install --no-deps -e RoboNexus-Agent/third_party/Cutie
python -m pip check
```

The Cutie checkout is pinned to
`ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a`. Its local `pyproject.toml`
removes the training-only `thinplate` dependency. `--no-deps` is intentional:
the tested inference dependencies are pinned by `tracking.yml`, so installing
the editable package must not resolve a second, drifting dependency set.

Download and verify the model:

```bash
cd RoboNexus-Agent/third_party/Cutie
python cutie/utils/download_models.py
cd ../../..

python - <<'PY'
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

model = get_default_model()
processor = InferenceCore(model, cfg=model.cfg)
print(type(model).__name__, type(processor).__name__, model.device)
PY
```

## Atomic DiT

DiT does not use Conda. Its dependency source of truth is the upstream
`pyproject.toml` plus the committed `uv.lock`:

```bash
cd multitask_dit_policy
uv sync --frozen
uv run --frozen python -c "import torch, multitask_dit_policy; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
```

Do not duplicate this dependency graph in `envs/dit.yml`.

## Updating an existing environment

Use `conda env update` when a YAML file changes:

```bash
conda env update -n robonexus-qwen -f envs/qwen3vl.yml --prune
```

After any update, run `python -m pip check` and the component-specific smoke
test above. Do not generate these files with a full `conda env export`: that
would include machine-specific build strings, local editable paths and
transitive packages unrelated to RoboNexus.
