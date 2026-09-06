# Third-party components and models

The desktop application bundles third-party software and model weights. Each retains its own license; making this repository public does not remove those terms.

## SuperAnimal models

SuperAnimal-TopViewMouse and SuperAnimal-Quadruped are by Mackenzie Mathis, Shaokai Ye, and contributors. Their upstream model cards specify a modified MIT license for **academic, non-commercial use**, with an animal-welfare condition. The original notices are reproduced in `notices/topview-model-license.md` and `notices/quadruped-model-license.md`.

- https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse
- https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped
- Reference: https://doi.org/10.1038/s41467-024-48792-2

## Software and other models

- DeepLabCut 3.0.1: LGPL-3.0; https://github.com/DeepLabCut/DeepLabCut/tree/v3.0.1
- PyTorch and torchvision: BSD-style notices; https://github.com/pytorch/pytorch and https://github.com/pytorch/vision
- DINOv2-small: Apache-2.0; https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056
- CPython: Python Software Foundation license; https://github.com/python/cpython
- Tauri: MIT or Apache-2.0; https://github.com/tauri-apps/tauri
- FFmpeg is supplied through imageio-ffmpeg: https://github.com/imageio/imageio-ffmpeg and https://ffmpeg.org/legal.html

Python package license files and source metadata are retained within the bundled `runtime/lib/python3.12/site-packages/` distributions. `runtime-lock.txt` records versions. The source repository contains the application and packaging code; the release manifest records model hashes. Laboratory videos and generated experimental results are not distributed.
