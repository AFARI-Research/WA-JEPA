<div align="center">

<img src="docs/wa_jepa_logo.png" width="800" alt="WA-JEPA">

<h1>WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving</h1>

<strong>Xinlin Wang<sup>1*</sup>, Yujiao Xiang<sup>1,2*</sup>, Yuheng Zhou<sup>1,3*</sup>, Jingqi Wang<sup>1*</sup>, Minqing Huang<sup>1*,‡</sup>, Jiajie Huang<sup>1,4</sup>, Dongxu Wei<sup>1†</sup></strong><br>
<strong>Tingguang Zhou<sup>1</sup>, Xiyang Wang<sup>1</sup>, Gong Chen<sup>1,5</sup>, Zhi Xu<sup>1</sup>, Feiyang Tan<sup>1</sup>, Hangning Zhou<sup>1</sup>, Mu Yang<sup>1</sup></strong>

<sup>1</sup> [Afari Intelligent Drive](https://afari.com/) &nbsp;&nbsp;
<sup>2</sup> [UESTC](https://en.uestc.edu.cn/ "University of Electronic Science and Technology of China") &nbsp;&nbsp;
<sup>3</sup> [SEU](https://www.seu.edu.cn/english/ "Southeast University") &nbsp;&nbsp;
<sup>4</sup> [BUPT](https://english.bupt.edu.cn/ "Beijing University of Posts and Telecommunications") &nbsp;&nbsp;
<sup>5</sup> [TJU](https://en.tju.edu.cn/ "Tianjin University")

*Equal contribution, listed in no particular order. † Project lead. ‡ Corresponding author: [mqhuang1211@gmail.com](mailto:mqhuang1211@gmail.com).*

<a href="README.md"><b>English</b></a> | <a href="README.zh-CN.md">中文</a>

[![Project Page](https://img.shields.io/badge/Project-Page-006DFF?style=flat&logo=googlechrome&logoColor=white)](https://wa-jepa.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.20974-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.20974)
[![Hugging Face](https://img.shields.io/badge/Weights-WA--JEPA-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/AFARI-Research/WA-JEPA)
[![NAVSIM v2](https://img.shields.io/badge/NAVSIM%20v2-TOP%201%20%7C%20EPDMS%2091.7-2B6CB0?style=flat)](#results)
[![HUGSIM](https://img.shields.io/badge/HUGSIM-TOP%201%20%7C%20HD--Score%200.4462-805AD5?style=flat)](#results)

</div>

## News

- `[2026/08]` WA-JEPA code, checkpoints, NAVSIM evaluation, and HUGSIM closed-loop evaluation are publicly available.
- `[2026/08]` The paper was released on [arXiv](https://arxiv.org/abs/2608.20974).
- `[2026/08]` The released checkpoint reaches **91.7 EPDMS** on NAVSIM v2.
- `[2026/08]` WA-JEPA obtains **0.4462 HD-Score** on HUGSIM in the zero-shot closed-loop setting.

## Overview

WA-JEPA is a V-JEPA-native world-action model for autonomous driving. It adapts
V-JEPA 2.1 from random-mask completion to future-directed planning through three
designs:

- **Hybrid future-masked causal pre-training** predicts future scene features from past frames.
- **Diffusive world modeling** generates plausible future latent representations instead of deterministic regressions.
- **Joint world-action modeling** denoises future scene tokens and ego trajectories in one spatiotemporal latent space.

The model uses four camera views and four historical frames, and predicts eight
future trajectory points at 2 Hz. The implementation is centered on a
V-JEPA 2.1 ViT-L/16 encoder and a joint flow predictor.

<div align="center">
  <a href="docs/figures/framework_overview.png">
    <img src="docs/figures/framework_overview.png" width="80%" alt="WA-JEPA framework overview">
  </a>
  <p><strong>WA-JEPA framework overview.</strong> Future-masked causal pre-training adapts V-JEPA representations for predictive world modeling, while the joint predictor couples future scene tokens with ego trajectories.</p>
</div>

## Results

| Benchmark | Setting | Score |
| --- | --- | ---: |
| NAVSIM v2.2 | navtest, EPDMS | **91.7** |
| HUGSIM | 436-scenario zero-shot closed loop, HD-Score | **0.4462** |

Detailed sub-metrics, seed statistics, difficulty breakdowns, and evaluation
caveats are available in [`docs/installation.md`](docs/installation.md).

## Visualization

<div align="center">
  <a href="docs/figures/hugsim_vis.png">
    <img src="docs/figures/hugsim_vis.png" width="80%" alt="WA-JEPA HUGSIM closed-loop visualization">
  </a>
  <p><strong>HUGSIM closed-loop rollouts.</strong> The planned trajectory is projected onto the front-view image and bird's-eye-view scene context.</p>
</div>

<div align="center">
  <a href="docs/figures/traj.png">
    <img src="docs/figures/traj.png" width="80%" alt="WA-JEPA trajectory visualization">
  </a>
  <p><strong>Trajectory prediction examples.</strong> Stage-2 predictions are compared with history and ground-truth trajectories across representative driving scenarios.</p>
</div>

## Getting Started

- [Installation, datasets, checkpoints, and environment setup](docs/installation.md)
- [HUGSIM closed-loop guide](docs/hugsim_closed_loop.md)
- [Configuration guide](configs/CONFIG_GUIDE.md)

The public release includes training code, NAVSIM open-loop evaluation, HUGSIM
closed-loop evaluation, and the published WA-JEPA checkpoint. The main entry
points are:

```text
models/multiview_causal_future_jepa.py
scripts/training/train.sh
scripts/evaluation/run_navsim_pdms.sh
scripts/evaluation/run_navsim_epdms.sh
scripts/evaluation/run_hugsim_benchmark.sh
```

## Citation

```bibtex
@article{wang2026wajepa,
  title={WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving},
  author={Wang, Xinlin and Xiang, Yujiao and Zhou, Yuheng and Wang, Jingqi and Huang, Minqing and Huang, Jiajie and Wei, Dongxu and Zhou, Tingguang and Wang, Xiyang and Chen, Gong and Xu, Zhi and Tan, Feiyang and Zhou, Hangning and Yang, Mu},
  journal={arXiv preprint arXiv:2608.20974},
  year={2026},
  eprint={2608.20974},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.20974}
}
```

## Acknowledgements

WA-JEPA builds on [V-JEPA 2](https://github.com/facebookresearch/vjepa2),
[NAVSIM](https://github.com/autonomousvision/navsim), nuPlan, OpenScene, and
[HUGSIM](https://github.com/l27n/hugsim). We thank the authors and maintainers
of these projects for their open-source contributions.

## License

This repository is released under Apache-2.0; see [`LICENSE`](LICENSE).
