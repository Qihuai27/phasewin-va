# Install Into `conda` Env `lima2`

This repository now keeps the install split into:

- `requirements.txt`: main runtime for classification, detection, and caption/VQA tasks
- optional extras: install `tensorflow` and `xplique` only if you need `dhsic`

## Recommended Commands

Activate the environment first:

```bash
source /home/visionx/anaconda3/etc/profile.d/conda.sh
conda activate lima2
```

Ensure the env has its own `pip` and build helpers:

```bash
conda install -y pip setuptools wheel
```

Install the main runtime:

```bash
pip install -r requirements.txt
```

Install local GroundingDINO from the already available source tree on this machine:

```bash
python -m pip install --no-build-isolation -e /home/visionx/workspace/VPS-main/GroundingDINO
```

Install optional D-HSIC dependencies only if needed:

```bash
pip install tensorflow xplique
```

## Notes

- `Qwen2.5-VL-3B-Instruct` recommends a recent `transformers`. If a released pip build raises `KeyError: 'qwen2_5_vl'`, upgrade with:

```bash
pip install git+https://github.com/huggingface/transformers accelerate
```

- `xplique` and `tensorflow` are intentionally separated because they are only needed for `algorithm='dhsic'` and are the least stable part of the stack.

- `GroundingDINO` should be installed with `--no-build-isolation`. Its upstream [setup.py](/home/visionx/workspace/VPS-main/GroundingDINO/setup.py) imports `torch` during build-time, which breaks under isolated editable installs even if `torch` is already installed in `lima2`.

- Detection uses `bert-base-uncased` through `GroundingDINO`. If that model is already present in the local Hugging Face cache, the task entrypoint will switch to offline loading automatically.

- Superpixel segmentation now has two backends:
  - preferred: OpenCV `ximgproc` superpixel operators
  - fallback: `skimage.slic`
  This is why `scikit-image` is included in `requirements.txt`.

## Minimal Verification

```bash
python -c "import clip, cv2, torch, torchvision, numpy, sklearn, transformers; print('core ok')"
python -c "import groundingdino; print('groundingdino ok')"
python -c "from skimage.segmentation import slic; print('skimage slic ok')"
```

Optional verification:

```bash
python -c "import xplique, tensorflow as tf; print('optional ok', tf.__version__)"
```
