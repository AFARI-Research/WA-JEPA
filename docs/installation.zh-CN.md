<p align="center">
  <img src="docs/wa_jepa_logo.png" alt="WA-JEPA" width="760">
</p>

<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

WA-JEPA 是一个面向自动驾驶的多相机 world-action 模型。它使用 V-JEPA 2.1
视觉编码器和一个联合 flow predictor，从历史相机观测中生成未来场景表征与
ego 轨迹。

本分支涵盖训练、NAVSIM 开环评估以及 HUGSIM 闭环评估。

## 模型

- 视觉编码器：V-JEPA 2.1 ViT-L/16
- 相机：`CAM_L0`、`CAM_F0`、`CAM_R0`、`CAM_B0`
- 输入：4 帧历史图像，分辨率 `256 x 512`
- 输出：8 个未来轨迹点，2 Hz
- 联合 flow predictor：12 层，hidden size 512，8 个 attention head
- 参数量：约 480.9M 可训练参数；包含冻结的 EMA 编码器时为 787.7M

主要实现：

```text
models/multiview_causal_future_jepa.py
eval/navsim_agent.py             # NAVSIM open-loop agent
close_loop/hugsim_planner.py     # HUGSIM closed-loop adapter
```

## 安装

推荐使用 Python 3.10 以及支持 CUDA 的 PyTorch 环境。

```bash
conda create -n wa-jepa python=3.10 -y
conda activate wa-jepa

# 安装与当前 CUDA 版本兼容的 PyTorch。
# 后续安装 NAVSIM 时，其固定的 torch 版本可能覆盖现有版本，因此安装完成后需进行核验。
pip install -r requirements.txt
pip install -e .
```

NuPlan 和 NAVSIM 需要单独安装。WA-JEPA 支持两种不同的 NAVSIM 评估环境；请将
它们安装到不同的目录中：

```bash
pip install -e /path/to/nuplan-devkit

# NAVSIM v1.1：仅用于 PDMS
git clone --branch v1.1 https://github.com/autonomousvision/navsim.git navsim-v1.1
pip install -e navsim-v1.1

# NAVSIM v2.2：仅用于 navtest EPDMS
git clone --branch v2.2 https://github.com/autonomousvision/navsim.git navsim-v2.2
pip install -e navsim-v2.2
```

NAVSIM 固定使用 `torch==2.0.1` 和 `numpy==1.23.4`，因此安装时可能会替换现有的
PyTorch。请核验最终环境；如与当前 GPU 不兼容，请重新安装适用的构建版本：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

两个 NAVSIM 版本都声明了相同的发行包名 `navsim`，因此第二次 `pip install -e`
会遮蔽第一次安装：`import navsim` 会解析到最后安装的版本。评估脚本不依赖该
解析结果，因为每个脚本都会将对应的 devkit 前置到 `PYTHONPATH`；但临时执行
`python -c "import navsim"` 并不能告诉你某个脚本将要使用哪个版本。

评估前请验证各代码检出目录：

```bash
git -C navsim-v1.1 describe --tags --always   # v1.1-<n>-g<sha>
git -C navsim-v2.2 describe --tags --always   # v2.2
```

上游同时存在名为 `v1.1` 的分支和标签，`--branch v1.1` 检出的是分支，其提交进度
领先于同名标签。因此，`describe` 输出 `v1.1-<n>-g<sha>` 而非单独的 `v1.1`；
这是正确检出该分支时的预期结果。只有 `v2.2` 会精确输出对应标签。

官方 NAVSIM 仓库有一个 `v1.1` release 以及若干个 v2 release tag（`v2.0`、
`v2.1`、`v2.1.1`、`v2.1.2` 和 `v2.2`）。官方仓库的 README 将其 `main` 分支视为
NAVSIM v2。本项目在公开的 EPDMS 命令中锁定 `v2.2`，以便 devkit 版本明确且可
复现。

这两个 NAVSIM 版本不可互换。NAVSIM v1.1 与 v2 具有不同的 dataclass、评分配置、
场景过滤器以及 metric-cache 格式。

## 数据集

所有数据均为公开数据。NAVSIM 自带下载脚本，因此请使用这些脚本，而不要自行
拼凑 URL：

```bash
cd navsim-v1.1/download

./download_maps.sh          # nuPlan maps v1.1
./download_trainval.sh      # OpenScene v1.1 trainval, used for training
./download_test.sh          # OpenScene v1.1 test, used for navtest scoring

# 仅用于 v2 两阶段评估：
cd ../../navsim-v2.2/download
./download_navhard_two_stage.sh
```

`download_navtrain_hf.sh` 只会获取 `trainval` 中的 `navtrain` 子集，这正是训练
配置所读取的部分；完整的 `trainval` 划分有数 TB，并非必需。

请按下面的布局组织下载结果，使这些路径都能解析——这也是官方 NAVSIM 安装指南
所描述的布局：

```text
$OPENSCENE_DATA_ROOT/
├── navsim_logs/{trainval,test}
└── sensor_blobs/{trainval,test}
$NUPLAN_MAPS_ROOT/          # nuplan-maps-v1.1, needed for metric caching
```

## 权重

训练好的 WA-JEPA 权重发布在
[AFARI-Research/WA-JEPA](https://huggingface.co/AFARI-Research/WA-JEPA)。这份
checkpoint 即[训练](#训练)一节所述的阶段二产物，也是各项评估命令通过
`CHECKPOINT` 加载的权重。

V-JEPA 2.1 ViT-L/16 预训练编码器由 Meta 随 V-JEPA 2 代码
（`facebookresearch/vjepa2`）一起发布；请从那里下载
`vjepa2_1_vitl_dist_vitG_384.pt`。NAVSIM 预设设置了 `require_pretrained: true`，
并在加载 WA-JEPA state dict 之前先加载它，因此训练和 NAVSIM 评估都需要它。
`configs/wa_jepa_hugsim.yaml` 是个例外：它禁用了预训练权重的加载，因为闭环推理
会直接恢复完整的 state dict。

一个 WA-JEPA checkpoint 既可以是一个 `model_state_dict.pt` 文件，也可以是具有
以下布局的目录：

```text
checkpoint/
├── model_state_dict.pt
└── state.pt                 # optional for inference
```

设置公共路径：

```bash
export VJEPA2_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export CHECKPOINT=/path/to/wa-jepa/checkpoint
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
export PYTHON_BIN=/path/to/python
```

评估脚本接受 `OPENSCENE_ROOT` 作为 `OPENSCENE_DATA_ROOT` 的别名；而训练配置
只读取后者。

`CHECKPOINT` 必须与所选评估 YAML 中的架构相匹配。

## 结果

下列数值均来自已发布的 checkpoint，使用本 README 中的命令评测得到。

NAVSIM `navtest`，开环：

| Devkit | 指标 | NC | DAC | TTC | Comf. | EP | 分数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1.1 | PDMS | 99.5 | 98.3 | 97.7 | 100 | 85.0 | **91.8** |

| Devkit | 指标 | NC | DAC | DDC | TL | EP | TTC | LK | HC | EC | 分数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2.2 | EPDMS | 99.4 | 98.2 | 99.7 | 99.9 | 87.8 | 98.9 | 98.3 | 98.3 | 88.1 | **91.7** |

HUGSIM 闭环，全部 436 个场景，零样本（没有任何训练阶段用过 HUGSIM 渲染的观测），
使用 heading 已修正的控制器：

| NC | DAC | TTC | C | PDMS | RC | HD-Score |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6856 | 0.9635 | 0.6120 | 0.6620 | 0.5717 | 0.5689 | **0.4462** |

按难度分级与按来源数据集拆分的 HD-Score：

| 分级 | n | HD-Score | | 数据集 | n | HD-Score |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 80 | 0.7977 | | nuScenes | 88 | 0.4725 |
| medium | 157 | 0.5563 | | KITTI-360 | 113 | 0.2963 |
| hard | 96 | 0.3060 | | Waymo | 108 | 0.5542 |
| extreme | 103 | 0.1362 | | PandaSet | 127 | 0.4702 |

总体 HD-Score 按四个难度分级的场景数加权。改成对四个数据集均匀平均是 0.4483，
对全部 436 个场景均匀平均是 0.4464 —— 这个数不依赖于某一种汇总规则。
`close_loop/aggregate_results.py` 会为一次运行打印按分级和按数据集的拆分。

不同环境下的复现结果可能存在差异。上述 NAVSIM 分数为固定十个推理种子的均值：EPDMS 在
这十个种子上的标准差为 0.05，取值范围为 `[91.63, 91.81]`。因此，单个种子的运行结果通常
接近 91.7，但不一定与均值完全一致。除随机种子外，GPU 型号、驱动版本、devkit 版本、
OpenScene 数据副本和指标缓存均可能影响分数；HUGSIM 闭环评估在多次运行之间也并非完全
确定。上述数值用于提供参考，并非要求逐位一致的复现目标。完整评测协议、各随机种子的统计
结果及对比基线详见论文。

## 训练

训练分两个阶段。阶段一在 NuPlan 视频上学习未来场景预测，此时关闭轨迹头；阶段二
打开轨迹头，并以阶段一的权重为起点在 NAVSIM `navtrain` 上做微调。

| | 阶段一 | 阶段二 |
| --- | --- | --- |
| 配置 | `configs/causal_future_jepa_mv_video_stage1.yaml` | `configs/causal_future_jepa_mv_navsim.yaml` |
| 数据 | NuPlan `train` | NAVSIM `navtrain` |
| 轨迹头 | 关闭 | 打开 |
| 损失 | `loss_jepa_pred` 1.0 | `loss_jepa_pred` 0.5、`loss_traj` 1.0 |
| `full_mask_prob` | 0.5 | 1.0 |
| 编码器 | 前 25k 步冻结 | 从第 0 步就可训练 |
| 初始化来源 | V-JEPA 2.1 编码器 | V-JEPA 2.1 编码器，再叠加阶段一 |

两个阶段都从环境变量读取路径：

```bash
export WM_SAVE_ROOT=/path/to/experiments
export VJEPA2_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
```

`WM_SAVE_ROOT` 用于指定基础目录。每次运行都会在该目录下创建独立的
`<user>/<name>/<experiment_name>/<timestamp>` 目录，解析后的 `config.yaml`、
checkpoint 以及加载报告都写在那里。

阶段二还需要在 `PYTHONPATH` 中指定 NAVSIM 代码检出目录，用于提供日志 dataclass 和
`navtrain` 的场景过滤器；`train.sh` 会为此把 `NAVSIM_DEVKIT_ROOT` 前置。整个流程请
使用同一个代码检出目录：两个 NAVSIM 版本的 dataclass 不同，中途切换会改变数据加载器
读取的数据结构。

两个阶段默认都使用 DeepSpeed ZeRO-2 + bf16
（`configs/deepspeed/ds_zero2_bf16.json`）。`train.sh` 会根据
`CUDA_VISIBLE_DEVICES` 推导进程数，未设置时则退回 `nvidia-smi`；设置
`NPROC_PER_NODE` 可以覆盖它。

### 阶段一：NuPlan 视频

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan

bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_video_stage1.yaml
```

这一阶段关闭了轨迹头，因此唯一的损失是 `loss_jepa_pred`，NuPlan 除了自带的 ego
位姿之外不需要任何轨迹标注。checkpoint 落在 `<save_root>/checkpoints/`：每次验证
之后的 `last`，以及按 `val_active_scene_xpred_mse` 选出的最好的五个。阶段一的
checkpoint 仅用于阶段二的初始化。

### 阶段二：NAVSIM 微调

```bash
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NAVSIM_DEVKIT_ROOT=$PWD/navsim-v1.1
export STAGE1_CKPT=/path/to/stage1/checkpoints/step_24000

bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_navsim.yaml
```

`STAGE1_CKPT` 用于设置 `model.init_from_checkpoint`。该字段在配置中为必填项，因此路径
缺失或无效时，模型构建会直接失败，不会仅加载预训练编码器后继续训练。也可以通过配置覆盖
传入该路径：

```bash
bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_navsim.yaml \
  model.init_from_checkpoint=[/path/to/stage1/checkpoints/step_24000]
```

应通过 `model.init_from_checkpoint` 加载，而非顶层的 `checkpoint`。
`init_from_checkpoint` 仅复制模型权重，并允许部分匹配；跨阶段初始化正需要这一点，因为
阶段二新增的轨迹头在阶段一的 checkpoint 中没有对应参数。顶层的 `checkpoint` 是训练器的
resume，会同时恢复 optimizer、scheduler 与 step 计数，因而继续的是阶段一的 schedule，
而非开始阶段二的 schedule。

每次加载都会在本次运行的 save root 下写出 `checkpoint_load_report.jsonl` 与 `.txt`，
以及一个 `checkpoint_load_keys/` 目录，其中记录加载比例，以及 missing、unexpected 与
形状不匹配的 key。阶段二启动后应查阅该报告。由于 `init_checkpoint_allow_partial` 已启用，
即使 checkpoint 仅匹配少量参数，训练仍可能继续执行。

训练期间只做验证，不做测试集评估。阶段二的验证集为 `navtrain` 中按日志名哈希留出的 5%，
选择指标为模型自身返回的验证损失。`navtrain` 与 `navtest` 不共享任何日志，因此训练与
checkpoint 选择全程都不接触测试数据。`navtest` 的评分在训练结束之后单独进行，使用下文的
评估命令。

已发布的权重就是这一阶段的产物；参见[权重](#权重)。

CLI 点式列表覆盖适用于所有配置项，因此执行短程测试时无需新建配置文件：

```bash
bash scripts/training/train.sh \
  --config configs/causal_future_jepa_mv_navsim.yaml \
  trainer.batch_size_per_gpu=1 trainer.max_epochs=0 trainer.max_steps=20
```

在该训练器中，`max_epochs` 的优先级高于 `max_steps`。因此，按步数限制运行时还必须
设置 `max_epochs=0`。

`configs/CONFIG_GUIDE.md` 记录了全部配置项。

## 生成 NAVSIM 指标缓存

NAVSIM v1.1 和 v2.2 的指标缓存必须分别生成，并在对应的评估命令中使用相同的缓存目录。
两个版本都需要 OpenScene 标注和 nuPlan 地图文件：

```bash
export PYTHON_BIN=/path/to/python
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
mkdir -p metric-cache
```

生成 v1.1 PDMS 缓存：

```bash
export NAVSIM_V1_DEVKIT_ROOT=$PWD/navsim-v1.1

$PYTHON_BIN "$NAVSIM_V1_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  cache.cache_path="$PWD/metric-cache/navsim-v1.1"
```

生成 v2.2 EPDMS 缓存：

```bash
export NAVSIM_V2_DEVKIT_ROOT=$PWD/navsim-v2.2

$PYTHON_BIN "$NAVSIM_V2_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  navsim_log_path="$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  metric_cache_path="$PWD/metric-cache/navsim-v2.2"
```

如需强制重新计算已有缓存，请追加相应的配置覆盖项：

```bash
# v1.1
cache.force_feature_computation=true

# v2.2
force_feature_computation=true
```

## NAVSIM v1.1 PDMS

这是原始的 NAVSIM `navtest` 评估，使用 v1.1 devkit、v1.1 场景过滤器以及 v1.1
指标缓存。该评估**不是** v2 EPDMS 评估。

配置：

```text
configs/wa_jepa_navsim_pdms.yaml
```

运行：

```bash
export NAVSIM_V1_DEVKIT_ROOT=$PWD/navsim-v1.1
export METRIC_CACHE_PATH=$PWD/metric-cache/navsim-v1.1

bash scripts/evaluation/run_navsim_pdms.sh
```

## NAVSIM v2 EPDMS

该评估使用 NAVSIM v2 的 `navtest` 单阶段评分器和 v2 指标缓存格式，**不同于**
v1.1 PDMS 评估，也不同于 v2 的 `navhard_two_stage` 高难度伪闭环基准。

配置：

```text
configs/wa_jepa_navsim_epdms.yaml
```

运行：

```bash
export NAVSIM_V2_DEVKIT_ROOT=$PWD/navsim-v2.2
export METRIC_CACHE_PATH=$PWD/metric-cache/navsim-v2.2

bash scripts/evaluation/run_navsim_epdms.sh
```

EPDMS 预设使用 `flow_inference_seed: 1`，即[结果](#结果)中 91.7 EPDMS 对应的十个推理
种子之一。该预设用于 NAVSIM v2 `navtest` 单阶段评估，因此单次运行对应其中一个种子的
结果，而非十个种子的均值。

PDMS 与 EPDMS 使用不同的 YAML 文件、devkit 和指标缓存。请勿混用 v1.1 和 v2
的评估资源。具体而言：

| 项目 | PDMS | EPDMS |
| --- | --- | --- |
| Devkit | NAVSIM `v1.1` | NAVSIM `v2.2` |
| 脚本 | 来自 v1.1 的 `run_pdm_score.py` | 来自 v2 的 `run_pdm_score_one_stage.py` |
| 划分 | `navtest` | `navtest` |
| 指标 | PDM/PDMS | EPDMS 风格的 navtest 分数 |
| 缓存 | v1.1 指标缓存 | v2 指标缓存 |
| YAML | `wa_jepa_navsim_pdms.yaml` | `wa_jepa_navsim_epdms.yaml` |

## 多 GPU EPDMS

多 GPU 启动器会把 `navtest` 日志拆分到可见的各个 GPU 上，并合并结果 CSV 文件。
它读取与上面单 GPU EPDMS 命令相同的 `NAVSIM_V2_DEVKIT_ROOT` 和
`METRIC_CACHE_PATH`：

```bash
export NAVSIM_V2_DEVKIT_ROOT=$PWD/navsim-v2.2
export METRIC_CACHE_PATH=$PWD/metric-cache/navsim-v2.2

export NAVSIM_NUM_GPUS=8
bash scripts/evaluation/run_navsim_navtest_epdms.sh
```

当未设置 `NAVSIM_NUM_GPUS` 时，启动器会统计 `CUDA_VISIBLE_DEVICES`，或退回到
`nvidia-smi -L`。

评估结果默认写入 `outputs/` 目录下。设置 `OUTPUT_DIR` 可使用其他位置。

## 预运行检查

使用 `--dry-run` 可以查看生成的 NAVSIM 命令，而不会启动模型推理：

```bash
bash scripts/evaluation/run_navsim_pdms.sh --dry-run
bash scripts/evaluation/run_navsim_epdms.sh --dry-run
```

## HUGSIM 闭环评测

HUGSIM 是基于 3D Gaussian Splatting 的闭环基准。模型生成的规划会反馈至仿真器，自车随之
移动，相机基于新的位姿重新渲染，误差也会在闭环过程中累积。相比之下，NAVSIM 根据录制日志
对预测轨迹进行评分。

官方基准包含 436 个场景，分布于四个数据划分（nuScenes 88、KITTI-360 113、Waymo 108、
PandaSet 127），难度等级包括 easy、medium、hard 和 extreme。

配置：

```text
configs/wa_jepa_hugsim.yaml
```

该配置与 EPDMS 预设采用相同架构，但不加载预训练编码器，因为推理时会直接恢复完整的已训练
状态字典。配置新增 `hugsim` 字段，用于定义相机映射、仿真步长和命令置换。

请按照 HUGSIM 官方 README 完成安装并获取数据包，然后对官方场景文件执行一次预处理：

```bash
export HUGSIM_ROOT=/path/to/HUGSIM
export HUGSIM_DATA_ROOT=/path/to/hugsim_data
export SIM_PYTHON=/path/to/hugsim-env/bin/python

python close_loop/prep_scenarios.py --data-root "$HUGSIM_DATA_ROOT"
```

运行单个数据划分：

```bash
export PYTHON_BIN=/path/to/wa-jepa-env/bin/python
export CHECKPOINT=/path/to/wa-jepa/checkpoint

# 建议先运行单个场景，以验证完整流程。
# 场景参数使用仿真器中的连字符命名；输出目录则使用下划线。
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes scene-0013-easy-00

# 随后运行完整数据划分。
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes
```

默认是 8 个 GPU、每个 4 个场景，即 32 个并发场景；单个场景大约需要 8.5 GB。设置
`HUGSIM_NUM_GPUS`、`HUGSIM_PER_GPU` 或 `JOBS` 可以更改这一设置。在该并发度下，全部 436 个
场景大约需要 3-5 小时。

汇总评估报告：

```bash
python close_loop/aggregate_results.py --dataset nuscenes \
  --results-root outputs/hugsim/nuscenes_ltf \
  --log-dir      outputs/hugsim/nuscenes_logs

# 完成四个数据划分后
python close_loop/aggregate_results.py --combined --out-dir outputs/hugsim
```

报告包含官方指标（NC、DAC、TTC、C、PDMS、RC、HD-Score）的总体结果和分难度结果、
HUGSIM 论文中已发布的基线结果（仅供背景参考），以及审计小节。审计结果不可忽略：当推理
发生异常时，客户端会回退至制动规划，而全程制动的场景仍可能取得较高分数。因此，这些场景
会被单独列出，并在另一组均值中排除。

报告在单个数据划分内对场景取等权平均，汇总报告再对四个数据划分取等权平均。相关论文也会
按照场景数（80 / 157 / 96 / 103）对四个难度等级进行加权。两种聚合方法会使 HD-Score 的
第三位小数产生差异，因此报告结果时应明确所采用的聚合方式。

在开始长时间运行之前，请先离线检查各项转换。下面两个都可以在没有 checkpoint 的情况下在
CPU 上运行：

```bash
python close_loop/selfcheck.py
python -m unittest tests.test_hugsim_planner
```

### 对 HUGSIM 的两处修复

本仓库不修改 HUGSIM 的上游文件。两处修复均在评估进程内、导入 `closed_loop` 之前应用
monkey-patch，因此仿真器代码检出目录与上游逐字节相同，两处补丁都以入口脚本的形式
放在本仓库的 `close_loop/` 下。

- **控制器航向角。** `traj2control` 将目标航向角计算为 `arctan2(d_forward, d_right)`，
  而其输入的 iLQR 状态要求使用 `arctan2(d_right, d_forward)`。因此，直线规划会得到
  90 度的目标航向角，近似直线的规划则可能使转向速率饱和。由此产生的分数损失随规划轨迹的
  横向抖动程度而变化，无法在不同方法之间相互抵消。因此，使用未修正控制器得到的排名同时
  受到规划质量和轨迹抖动的影响。每次运行都会应用该修正：`run_hugsim_benchmark.sh` 始终通过
  `close_loop/run_fixed_controller.py`。上游 PR #57 修的是同一个缺陷，目前仍处于 open 状态。
- **两处异常会导致 436 个场景中的 16 个无法评分。** `ConstantPlanner.__init__` 不接受 15 个 PandaSet
  `medium_02` 场景传给它的 `max_t`；而当一辆背景车辆只剩一个 waypoint 时，`IDM.update` 会
  索引一个空的 lookahead。两者都会在场景能够被评分之前就中止它。
  `close_loop/run_upstream_compat.py` 修补两者（`HUGSIM_COMPAT=1`，默认值），并且不会改变
  原本就能跑通的场景的分数。

因此，基线需要在同一协议下重新评分，不能直接引用原始结果。HUGSIM 论文中的 UniAD、VAD 和
LTF 数值基于未修正的控制器及较早版本的场景数据，`aggregate_results.py` 输出的相关列仅用于
提供背景信息，不构成直接对比。论文中的结果在相同场景、修正后的航向控制器、相同真值驾驶
命令和相同聚合方式下重新评估全部基线，并使用随 DrivoR 发布的修正版评估代码，以确保各方法
采用一致的指标实现。

`max_t` 补丁的做法是接受该参数并忽略它，因此这 15 个场景里的背景交通在整个 rollout 中都保持
匀速。模拟器里没有任何地方说明 `max_t` 本该做什么，但它的每一种解读——到时间后停下、消失
（despawn）、只规划到那么远——在 `t = max_t` 之前都与匀速一致。`dt` 为 0.25 s，因此
`max_t: 25` 就是第 100 步：在第 100 步之前结束的 rollout 完全不受该补丁影响，而跑过第 100 步
的 rollout 则是在一个假设之下被打分的。

轨迹展开长度取决于规划器输出，而非代码本身，因此每次完成 PandaSet 评估后都应执行检查：

```bash
python close_loop/check_maxt_reached.py \
  --results-root outputs/hugsim/pandaset_ltf --dataset pandaset
```

只要这 15 个中有任何一个达到了第 100 步，该脚本就以非零码退出。在所报告的 checkpoint 上有
7 个达到了，最长的跑了 152 步；这些分数会与均值分开报告。

`HUGSIM_COMPAT=0` 会跳过这两个崩溃补丁，用来复现补丁出现之前的运行：那种运行里，PandaSet
只有 127 个场景中的 112 个能被打分。一份报告用的是哪个分母，聚合器的 PandaSet 注意事项里
会写明。

完整指南包含仿真器到模型的各项转换及各数据划分的特定注意事项：

```text
docs/hugsim_closed_loop.zh-CN.md
```

### 闭环过程录像

闭环分数反映车辆的行驶进度，但不能完整描述其行为。在运行时设置 `SAVE_VIZ=1` 后，客户端会
逐步归档 `CAM_FRONT` 渲染图、模型读取的四视图拼接图以及模型输出的规划：

```bash
SAVE_VIZ=1 bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes
```

每个场景的归档约占 40 MB，且不会影响规划结果。该选项必须在运行前启用，因为归档无法在
运行结束后重建。

可以渲染单个场景、一组精选场景，或并行渲染各数据划分：

```bash
# 单个场景
python close_loop/make_video.py \
  --root    outputs/hugsim/nuscenes_ltf \
  --out-dir outputs/hugsim/videos/nuscenes \
  --scenes  scene-0013_easy_00 \
  --dataset nuscenes --hugsim-root "$HUGSIM_ROOT"

# 各难度等级中得分最高和最低的场景
python close_loop/make_video.py \
  --root    outputs/hugsim/nuscenes_ltf \
  --out-dir outputs/hugsim/videos/nuscenes \
  --pick    outputs/hugsim/results_nuscenes.json \
  --dataset nuscenes --hugsim-root "$HUGSIM_ROOT"

# 四个数据划分，每次并行处理八个场景
python close_loop/render_videos.py --workers 8 \
  --hugsim-root "$HUGSIM_ROOT" --skip-existing
```

每一帧都包含投影到路面上的规划轨迹、一个包含 plan、其他 agent 以及至今已行驶路径的鸟瞰图
面板，以及一个显示步数、时间、速度、导航命令和该场景分数的状态栏。推理抛出异常的那些步会被
标记为 `FALLBACK: inference failed`，这样视频就无法把客户端的刹车 plan 冒充成模型的 plan。
`render_videos.py` 还会写出一个 `index.md`，按数据划分和难度等级分组，并按 HD-Score 排序。

渲染需要相机几何参数，而四个数据划分并不共享这些参数：视场角不同（水平方向 65.12 对 58.12 度），
KITTI-360 会以 5 度俯仰角进行校正，而 PandaSet 把相机安装得低了 0.4 m。因此两个脚本都会读取
模拟器的 `<dataset>_camera.yaml`——通过 `--hugsim-root` 和 `--dataset`，或直接通过
`--camera-config`——并将其与客户端在运行期间记录的参数进行交叉核对。若不匹配，则会停止渲染，
而不是用错误的几何参数生成看似合理的叠加图。渲染仅使用 CPU，且只访问归档数据。

## 测试

```bash
python -m unittest \
  tests.test_inference_config \
  tests.test_navsim_agent_imports \
  tests.test_hugsim_planner \
  tests.test_hugsim_video
```

## 说明

- 当前的公开预设需要四个相机视图和四帧历史。
- 所发布的完整模型 checkpoint 是以严格模式（strict）加载的。
- 消融 checkpoint 需要各自匹配的 YAML 配置。
- NAVSIM v2 单阶段 EPDMS 不应直接与高难度伪闭环或两阶段评估结果进行比较。

## 许可

本仓库以 Apache-2.0 许可发布；参见 `LICENSE`。

`third_party/vjepa2/` 是从 Meta 的 V-JEPA 2 发布版 vendored 而来，并保留其自身的
许可（`third_party/vjepa2/LICENSE`、`third_party/vjepa2/APACHE-LICENSE`）。
NAVSIM、nuPlan、OpenScene 和 HUGSIM 未在此处重新分发——请按照它们各自的条款从
它们自己的仓库安装，并从原始来源获取它们的数据集。
