# HUGSIM 闭环推理与评测

[English](hugsim_closed_loop.md) | **中文**

HUGSIM 是基于 3D Gaussian Splatting 的照片级真实感闭环基准。NAVSIM 根据录制日志对
预测轨迹进行评分；HUGSIM 则将模型生成的规划反馈至仿真器，使自车随之移动，相机基于新的
位姿重新渲染，误差也会在闭环过程中持续累积。该过程能够更严格地评估连续驾驶行为。

官方基准包含 436 个场景，分布于四个数据划分（nuScenes 88、KITTI-360 113、Waymo 108、
PandaSet 127），难度等级包括 easy、medium、hard 和 extreme。

本仓库不修改 HUGSIM 的上游文件。下文所述修正均在评估进程中、导入 `closed_loop` 之前
应用，因此仿真器代码检出目录与上游保持逐字节一致，相关实现均位于本仓库的 `close_loop/`
目录。修正包括每次运行都会应用的[控制器航向角修复](#控制器航向角修复)，以及确保 436 个
场景中 16 个场景能够完成评分的[上游兼容补丁](#上游兼容补丁)。已发布的 UniAD、VAD 和 LTF
基线均未包含这些修正，因此需要在相同协议下重新评分，不能直接引用原始结果；参见
[与已发表基线的比较](#与已发表基线的比较)。

## 集成方式

上游已支持外部驾驶系统。`sim/utils/launch_ad.py` 以子进程方式启动 shell 脚本，并通过
两个 FIFO 交换数据：

```text
closed_loop.py  --(pickle (obs, info))-->  <output>/obs_pipe   -->  hugsim_client.py
closed_loop.py  <--(pickle plan [8,2])---  <output>/plan_pipe   <--  hugsim_client.py
```

因此，本仓库仅实现客户端侧集成，主要由以下三个文件组成：

| 文件 | 作用 |
| --- | --- |
| `close_loop/hugsim_planner.py` | 加载 checkpoint，维护历史缓冲区，将观测转换为模型输入，并将路径点转换回仿真器坐标系 |
| `close_loop/hugsim_client.py` | 子进程：打开 FIFO，循环直到仿真器发送 `'Done'`，写出每个场景的统计数据 |
| `scripts/evaluation/run_hugsim_ad.sh` | 仿真器实际调用的启动脚本；在执行客户端前配置环境并完成预检 |

仿真器仅接受 `--ad uniad|vad|ltf`，因此 WA-JEPA 使用 `ltf` 槽位。该选择只影响仿真器读取
哪个 `base.*_path` 配置项；生成的基础配置会将三个路径均指向同一个启动器。

此外，本仓库还提供不依赖 GPU 或仿真器的离线工具：`prep_scenarios.py` 处理官方场景文件，
`selfcheck.py` 校验各项转换，`aggregate_results.py` 汇总各场景的 `eval.json`，
`make_video.py` 和 `render_videos.py` 则用于渲染启用 `SAVE_VIZ=1` 后生成的归档。

### 六处坐标与语义转换

这些转换发生错误时不会抛出异常。相机槽位错误或命令映射反向时，系统仍可能生成全部 436 个
场景的分数，但性能下降与模型能力不足难以区分。以下转换均通过仿真器实测验证，而非仅依据
文档推断。

1. **Camera slots.** 模型按照在 `model.camera_names` 中的位置来索引一个学习得到的相机
   embedding，因此这个映射是有序的：

   | model | HUGSIM |
   | --- | --- |
   | `cam_l0` | `CAM_FRONT_LEFT` |
   | `cam_f0` | `CAM_FRONT` |
   | `cam_r0` | `CAM_FRONT_RIGHT` |
   | `cam_b0` | `CAM_BACK` |

2. **采样率。** HUGSIM 的步长为 0.25 s（4 Hz），模型的历史帧和路径点间隔为 0.5 s
   （2 Hz），因此历史序列需要每隔 2 个仿真帧采样一次。若使用每个仿真帧，模型观察到的
   运动速度将减半。

3. **World to ego.** `(x, y, yaw) = (ego_box[0], ego_box[1], ego_box[6])` 是右手系，
   x 朝前、y 朝左，与训练时的约定一致。

4. **Command encoding.** HUGSIM 使用 0 = right、1 = left、2 = straight；NAVSIM 使用
   0 = left、1 = straight、2 = right。映射为 `[2, 0, 1]`，以 HUGSIM 的值作为索引。若使用
   恒等映射，则会把每一次右转都变成左转。

5. **Output frame.** `closed_loop.py` 把规划当作 lidar 坐标系的 `(x right, y forward)`
   来读取，并施加 `plan[:, [1, 0]]`，因此客户端会从 ego 坐标系的预测中发出 `(-y, x)`。

6. **精度。** 在 `torch.autocast(bfloat16)` 下使用 fp32 权重，与生成已报告 NAVSIM 结果的
   执行路径一致。直接将权重转换为 bfloat16 属于不同的执行方式，无法复现相同结果。

`close_loop/selfcheck.py` 会在离线状态下断言所有这些转换，而 `tests/test_hugsim_planner.py`
则把它们固化在测试套件里。

## 前置条件

### 1. 两个环境

仿真器需要 `gsplat`、`nerfstudio`、`open3d` 和 CUDA 扩展；WA-JEPA 需要 `torch` 及本仓库。
二者可以共用同一环境，但独立环境更便于管理依赖。客户端以独立进程启动，因此只需将
`PYTHON_BIN` 指向能够导入 `torch` 和本仓库的 Python 解释器。客户端会导入 `models/`、
`training/checkpoint.py` 和 `utils/`，这些模块均不依赖 nuPlan 或 NAVSIM。

请按照 HUGSIM 官方 README 完成安装，并验证相关扩展能够实际运行，而非仅能成功导入：

```bash
python -c "
import torch, gsplat
from simple_knn._C import distCUDA2
print(gsplat.__version__, distCUDA2(torch.rand(100, 3, device='cuda')).shape)
"
```

### 2. 数据包

场景和智能体资产来自 HUGSIM 的发布包：

```text
$HUGSIM_DATA_ROOT/
├── scenes/{nuscenes,kitti360,waymo,pandaset}/...   # 3DGS scenes, ground.ply per scene
├── 3DRealCar/<id>/{gs.pth,wlh.json}                # inserted vehicle assets
└── {nuscenes,kitti360,waymo,pandaset}/*.yaml       # 436 official scenarios
```

### 3. 准备场景

发布包中的场景文件与仿真器代码对智能体资产路径的约定不一致：每个 `plan_list` 条目的
第 5 个字段以 `/postprocess/shadow.pth` 结尾，而 `sim/utils/plan.py` 会在其后追加
`gs.pth`，去查找 `.../shadow.pth/gs.pth`。上游自己的导出脚本会剥掉同样的后缀，所以这是
发布包里的一处不一致，而非使用错误。

```bash
python close_loop/prep_scenarios.py --data-root "$HUGSIM_DATA_ROOT"
```

该命令会生成 `close_loop/configs/benchmark/<dataset>/*.yaml` 和 `runnable.txt`，后者列出
资产完整的场景。脚本只修改智能体资产路径，`start_*`、`mode`、`plan_list` 结构和
`load_HD_map` 均保持不变；同时还会验证重写后的 ID 仍被解析为字符串。该检查不可省略：
3DRealCar ID 形如 `2024_07_05_16_16_38`，YAML 1.1 会将下划线视为数字分隔符，因此未加
引号的 ID 会被解析为整数 `20240705161638`，导致后续资产查找失败，但 YAML 解析阶段不会报错。

## 运行基准

```bash
export HUGSIM_ROOT=/path/to/HUGSIM
export HUGSIM_DATA_ROOT=/path/to/hugsim_data
export SIM_PYTHON=/path/to/hugsim-env/bin/python
export PYTHON_BIN=/path/to/wa-jepa-env/bin/python
export CHECKPOINT=/path/to/wa-jepa/checkpoint
export OUTPUT_DIR=outputs/hugsim

# 建议先运行单个场景。单场景耗时约 2-20 分钟，可验证完整流程。
# 场景参数使用仿真器中的连字符命名。
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes scene-0013-easy-00

# 随后运行完整数据划分。
bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes
```

添加 `--dry-run` 可打印解析后的执行计划，而不启动基准评估。

默认配置使用 8 块 GPU，每块并行运行 4 个场景，共 32 个并发场景。单个场景约需 8.5 GB
显存（3DGS 渲染约 3.5 GB，规划器约 4.9 GB），每块 GPU 运行四个场景约占用 25 GB。此时
主要瓶颈为计算资源而非显存。可通过 `HUGSIM_NUM_GPUS`、`HUGSIM_PER_GPU` 或 `JOBS` 调整。

场景通过 `xargs -P` 以工作池方式调度，而非预先静态分片。不同场景的运行时间相差超过十倍：
发生早期碰撞的场景约 80 秒结束，完整轨迹展开约需 19 分钟。静态分片会使部分 GPU 在等待
最长分片完成期间处于空闲状态。

**每个场景必须使用独立进程。** `sim/utils/plan.py` 用 `enumerate(plan_list)` 构建
`self.stats[f'agent_{i}']`，并且从不清除上一个场景的 key，因此在同一进程里第二次
`gymnasium.make` 会抛出 `KeyError: 'agent_N'`，其中 N 是上一个场景的 agent 数量。运行器
已经这样派发；不要在 Python 循环里批量跑场景。

在默认并发下，跑完全部 436 个场景大约需要 3-5 小时。

### 其余划分

```bash
for ds in nuscenes kitti360 waymo pandaset; do
  bash scripts/evaluation/run_hugsim_benchmark.sh "$ds"
done
```

带有 `load_HD_map: true` 的场景还额外需要 `HD_MAP_PATH` 和 `HD_MAP_VERSION` 指向原始
数据集；没有它们，这些场景就无法启动。当某个数据划分包含这类场景、而这些变量又未设置时，
运行器会发出警告。

### 断点续跑

`SKIP_DONE=1`（默认值）仅在 `eval.json` 和 `planner_stats.json` **同时**存在时跳过场景。
若只有 `eval.json`，说明仿真器已完成评分，但驾驶系统未正常关闭，该结果需要重新运行，不应
直接保留。

当用另一个 checkpoint 续跑时，把 `WM_RUN_TAG` 设为一个必须出现在输出路径中的子串。如果它
没有出现，预检会拒绝运行，从而防止两个 checkpoint 的结果被误合并至同一结果表。
对于训练会就地覆写的 checkpoint 目录（`last`），还要设置 `WM_EXPECT_STEP`：文件大小在
各个 epoch 之间保持一致，因此只有 `state.pt` 才能标识当前存在的是哪一份权重。

## 汇总结果

```bash
python close_loop/aggregate_results.py --dataset nuscenes \
  --results-root  outputs/hugsim/nuscenes_ltf \
  --log-dir       outputs/hugsim/nuscenes_logs \
  --checkpoint    "$CHECKPOINT"
```

该命令会生成 `RESULTS_nuscenes.md` 和 `results_nuscenes.json`。完成四个数据划分后：

```bash
python close_loop/aggregate_results.py --combined --out-dir outputs/hugsim
```

`--log-dir` 为可选参数，但建议提供。终止原因仅记录在日志中；未提供时，所有场景的终止原因
均会显示为 `unknown`。

### 指标

评分使用未经修改的 `sim/utils/score_calculator.py`。逐帧公式如下：

```text
pdms = nc * dac * (5 * ttc + 2 * c) / 7
```

而每个场景 `hdscore = mean(pdms) * rc`，其中 `rc` 会被裁剪到 1.0。HD-Score 是核心指标。

| 指标 | 含义 |
| --- | --- |
| NC | 没有发生本方责任碰撞（at-fault collision）的帧所占的比例 |
| DAC | 处于可行驶区域内的帧所占的比例，依据是场景的 `ground.ply` 点云，而非 HD map |
| TTC | 满足碰撞时间（time-to-collision）阈值的帧所占的比例 |
| C | 处于加速度 / jerk / 横向加速度限制之内的帧所占的比例 |
| PDMS | 上面逐帧公式的均值 |
| RC | 路线完成度（route completion），`(furthest waypoint reached + 1) / (waypoint count * 0.9)`，裁剪到 1.0 |
| HD-Score | **PDMS × RC** |

### 报告为何带有审计小节

当推理抛出异常时，`hugsim_client.py` 会回退至制动轨迹并记录次数。即使每一步均发生回退，
场景仍可能获得非零且相对较高的 HD-Score，因为静止车辆不会发生碰撞，并会保持在可行驶区域
内。若仅报告总体均值，可能会高估未实际参与轨迹展开的模型性能。

因此聚合器会列出每一个 `n_failures > 0` 的场景，并报告一个把它们剔除之后的第二个均值。
对于那些有 `eval.json` 却没有 `planner_stats.json` 的场景也是如此。完全没有产生任何结果
的场景会逐一列出，并附上日志中的实际异常，而非仅报告数量。

终止原因来自日志，但有一个例外：`closed_loop.py` 会在 `cnt > 400` 时结束一个 episode，
且不输出终止信息，因此需要根据步数推断，并标记为 `StepLimit`。

### 各划分已知的注意事项

聚合器会在每份报告中记录以下事项，解读绝对数值时需要一并考虑。

- **KITTI-360 和 Waymo 只渲染三个前向相机。** 两个 `*_camera.yaml` 文件仍列出六个相机，
  因此不会触发异常，但 `CAM_BACK` 返回全黑图像。四相机模型在这两个数据划分中会损失四分之一
  的视觉输入。这是基准本身的特性，并非实现缺陷，但会对依赖后视图的方法产生影响，且影响
  程度不一定与基线相同。
- **KITTI-360 对所有已评估方法均具有较高难度。** 三个已发布基线在该数据划分上的 HD-Score
  均低于 0.09，因此应结合该尺度解读绝对数值，而不宜直接参照 nuScenes。
- **有十五个 PandaSet `medium_02` 场景需要兼容性补丁才能运行。** 它们的 `plan_list` 会把
  `controller_args: {max_t: 25}` 传给 `ConstantPlanner`，而后者的 `__init__(self)` 不接受
  任何参数，因此 `gymnasium.make()` 会在第一帧渲染之前就抛出 `TypeError`。这个数据包比
  代码更新。`HUGSIM_COMPAT=1`（默认值）会接受并忽略该参数；这对每一次在 `max_t` 之前
  结束的轨迹展开均无影响，`close_loop/check_maxt_reached.py` 会在评估后进行验证。当
  `HUGSIM_COMPAT=0` 时，这十五个场景会被标记为缺失，PandaSet 仅对 127 个场景中的 112 个
  进行评分。参见[上游兼容补丁](#上游兼容补丁)。

### 与已发表基线的比较

报告包含 HUGSIM 论文（arXiv:2412.01718v1，Table XIII）按难度等级给出的 UniAD、VAD 和 LTF
HD-Score。这些结果基于未修正的控制器和较早版本的场景数据，且三个基线均**在对应的评估数据
划分上进行训练**，与零样本跨域模型的评估设置不同。因此，相关列仅用于提供背景信息，不应
作为排行榜使用。

具有可比性的结果需要在本文协议下重新评估各基线：使用相同场景、修正航向角后的控制器、
相同的真值驾驶命令、相同的聚合方式及统一的指标实现。随 DrivoR 发布的修正版评估代码提供
了该条件，论文报告的是基于该代码重新评分的结果。

聚合器在单个数据划分内对场景取等权平均，`--combined` 则对四个数据划分取等权平均。相关
论文也会按场景数（80 / 157 / 96 / 103）对四个难度等级加权。两种方法会使 HD-Score 的
第三位小数产生差异，因此报告结果时必须注明所采用的聚合方式。

## 验证与调试

### 离线检查

```bash
python close_loop/selfcheck.py                    # config plus the pure conversions
python -m unittest tests.test_hugsim_planner      # the same invariants, in the suite
```

两项检查均在 CPU 上运行，且不需要 checkpoint。检查范围包括上述转换及三个 shell 层之间的
连接关系。例如，基础配置中的 `output_dir` 必须与运行器扫描的目录一致，否则运行结束时可能
仅得到空结果目录和退出码 0。

### 检查真实帧

```bash
# 运行参数为使用连字符的场景名；转储文件位于使用下划线命名的输出目录。
HUGSIM_DUMP_OBS=5 bash scripts/evaluation/run_hugsim_benchmark.sh nuscenes scene-0013-easy-00
python close_loop/selfcheck.py --dump outputs/hugsim/nuscenes_ltf/scene-0013_easy_00/dumped_obs.pkl
```

客户端会写出前五个真实的 `(obs, info)` 帧，可用于发现全黑相机、始终不变的命令通道以及
尺寸小于模型输入的渲染结果。这些问题无法通过合成输入识别。

### 逐步归档

`SAVE_VIZ=1` 会逐步保存模型的实际输入和规划，用于分析低分场景源于观测异常还是规划输出。
每个场景会在 `viz/` 下保存 `CAM_FRONT` 渲染结果和模型逐步读取的四视图拼接图，并在结束时
生成 `plan_log.pkl`。该文件包含规划、自车与物体边界框、导航命令、速度和每一步的时间戳。
每个场景约占 40 MB，不影响规划结果。

该选项必须在运行前启用。归档无法从 `eval.json` 重建，因此事后分析需要重新运行仿真；由于
闭环执行在多次重启之间并非完全确定，重新运行可能无法复现原始轨迹展开。

`viz/camera.json` 每个场景写一次，保存仿真器所报告的每个相机的内参和 `v2c`，外加当前
所用相机配置的路径。它之所以存在，是因为 `make_video.py` 必须用这次运行实际渲染所用的
相机来投影规划，而四个数据划分并不共用同一个相机——见下文。

### 录像

```bash
python close_loop/make_video.py \
  --root    outputs/hugsim/nuscenes_ltf \
  --out-dir outputs/hugsim/videos/nuscenes \
  --pick    outputs/hugsim/results_nuscenes.json \
  --dataset nuscenes --hugsim-root "$HUGSIM_ROOT"

python close_loop/render_videos.py --workers 8 \
  --hugsim-root "$HUGSIM_ROOT" --skip-existing
```

`make_video.py` 渲染单个结果目录中的场景；`render_videos.py` 在各数据划分上并行调用该工具，
并生成按数据划分和难度等级分组的 `index.md`，其中包含各场景分数。两者均仅使用 CPU 和归档
数据，因此可在任何能够访问输出目录的环境中运行，不需要 GPU、checkpoint 或仿真器。

每一帧都是 `CAM_FRONT` 的渲染结果，上面把规划轨迹投影到路面上，还有一个鸟瞰面板显示
规划、其他 agent 以及到目前为止已行驶的路径，以及一个状态栏，显示步数、时间、速度、导航
命令和该场景的分数。还有两个红色标记：在 `planner_stats.json` 所列出的步上
显示 `FALLBACK: inference failed`，在 history buffer 填满之前的热身步上显示
`history padded`。这些标记用于区分模型生成的驾驶行为与可能获得非零分数的回退制动行为。

投影链，源自 HUGSIM 的 `sim/utils/sim_utils.py` 和 `hugsim_env`：

```text
p_v = (-y_ego, 1.5, x_ego)                    vehicle frame: x right, y down, z forward
p_c = inv(cam_rect) @ v2c @ inv(v2front) @ p_v
u   = fx * x_c / z_c + cx,  fy from fovy and H, not from fx
```

`whl = [1.6, 1.5, 3.0]` 把地平面置于 `v_y = 1.5`，正是这一点让叠加层贴在路面上、而不是
悬浮。对 `CAM_FRONT` 而言，`v2c` 各项会相互抵消，只剩下 `cam_rect`；保留这个一般形式
意味着同一份代码可以投影其他相机。

四个数据划分在以下参数上存在差异，错误设置可能生成表面合理但实际不正确的叠加结果：

| Split | fovx | fx, fy | `cam_rect` |
| --- | --- | --- | --- |
| nuScenes | 65.12° | 626.41, 626.41 | 沿 y 平移 0.3 |
| KITTI-360 | 65.12° | 626.41, 626.41 | 绕 x 旋转 5°，无平移 |
| Waymo | 58.12° | 719.82, 771.55 | 沿 y 平移 0.3 |
| PandaSet | 58.12° | 719.82, 771.55 | 沿 y 平移 0.7 |

因此 `fy` 必须来自 `fovy`，而不是来自 `fx`——在 Waymo 和 PandaSet 上二者相差 7%——
KITTI-360 的 5 度俯仰会把地面移动几十个像素，而 PandaSet 的相机比其他相机低 0.4 m。
渲染器从配置中读取这些参数，并与 `viz/camera.json` 交叉核对。该检查存在一个已知限制：
`CAM_FRONT` 的 `v2c` 在四个数据划分中逐字节相同，因此仅凭内参只能区分 65.12° 和 58.12°
两组配置。客户端记录的 `_dataset` 字段用于进一步区分；若其与 `--dataset` 不一致，渲染会
终止。

相机 YAML 以度（degree）存储视场角，而 `load_camera_cfg` 会就地把它们转成弧度——因此
`info['cam_params']`（以及由此而来的 `camera.json`）里存的是弧度（radian）。两者都会被
读取，靠数值是否超过 pi 来区分。

`tests/test_hugsim_video.py` 固化了这些不会主动报错的关键约束：Euler 约定
（`from_euler('XYZ')` 是内旋，即 `Rx @ Ry @ Rz`）、四个数据划分的焦距、车辆的左侧在图像里
也是左侧、相机后方的点会被丢弃，以及指向另一个数据划分的 `camera.json` 会中止渲染。

### 场景失败时该看哪里

| 症状 | 该看哪里 |
| --- | --- |
| 没有 `eval.json` | `<LOG_DIR>/<scenario>.log`，即仿真器自己的 stdout |
| 有 `eval.json` 但没有 `planner_stats.json` | 客户端异常退出；回溯信息位于同一日志中 |
| `planner_stats.json` 里 `n_failures > 0` | `failures` 列表记录了步数和异常 |
| 无输出文件，退出码为 0 | `base.output_dir` 与扫描目录不一致；对比生成的 `*_base.yaml` 与 `OUT_ROOT` |

不要将运行命令通过管道传递给 `tail`：`cmd | tail -n 50` 返回的是 `tail` 的退出码，可能
使失败场景显示为成功。应根据 `eval.json` 是否存在判断运行是否成功，而不是依赖进程退出码；
`closed_loop.py` 会捕获 `create_gym_env` 周围的异常，并仍以退出码 0 结束。

## 控制器航向角修复

上游的 `traj2control` 将目标航向角计算为 `arctan2(d_forward, d_right)`，而 iLQR 状态要求
使用 `arctan2(d_right, d_forward)`。因此，直线轨迹会得到 90 度的目标航向角；对于近似
直线的轨迹，零值附近的横向抖动会使目标航向角在较大范围内变化，从而导致转向速率饱和。

该惩罚并非可在不同方法之间抵消的常量偏移，而是随规划轨迹的横向抖动程度变化。因此，使用
未修正控制器得到的排名同时受到驾驶性能和轨迹抖动的影响。该问题会在每次运行中修正，而不
提供恢复缺陷行为的开关：
`run_hugsim_benchmark.sh` 的入口始终是 `close_loop/run_fixed_controller.py`，也没有任何
设置能把这个缺陷放回去。

已发布的 UniAD、VAD 和 LTF 结果基于存在该缺陷的控制器，直接比较无法形成同等条件下的规划
质量评估。具有可比性的结果应在修正后的控制器下重新评估各基线；参见
[与已发表基线的比较](#与已发表基线的比较)。

`close_loop/run_fixed_controller.py` 会在 `closed_loop` 被 import 之前对该函数打
monkey-patch，因此 HUGSIM 的文件保持逐字节相同：修复以
`close_loop/run_fixed_controller.py` 的形式保留在本仓库，不修改仿真器代码检出目录。上游
PR #57 处理同一缺陷，目前仍处于开放状态；该 PR 调整解包顺序而非参数顺序，两种实现的结果
逐元素一致。

## 上游兼容补丁

HUGSIM 中有两个缺陷会在评分前终止场景，且与所使用的驾驶系统无关。两者均源于数据包与代码
之间的版本不一致，共导致 436 个场景中的 16 个无法评分。两项缺陷均由
`close_loop/run_upstream_compat.py` 打补丁，除非 `HUGSIM_COMPAT=0`，否则
`run_hugsim_benchmark.sh` 会选用它。

入口脚本先应用补丁，再转交给 `HUGSIM_COMPAT_INNER` 指定的原始入口，通常为
`run_fixed_controller.py`。这种方式可以组合两类修正，且无需在不同文件中重复实现。所有
补丁均在导入 `closed_loop` 前安装，并在同一进程内生效；HUGSIM 文件保持逐字节一致。

### 缺陷 1 —— `ConstantPlanner` 收到了它从未实现的 `max_t`

```text
TypeError: ConstantPlanner.__init__() got an unexpected keyword argument 'max_t'
```

`sim/utils/plan.py:55` 通过
`getattr(agent_controller, model.controller)(**model.controller_args)` 实例化控制器，
而上游的 `ConstantPlanner.__init__(self)` 不接受任何参数。补丁接受并忽略该参数，使这
十五个场景与其余 421 个场景一样使用匀速背景交通。

`max_t` 的预期语义不会影响达到限制前的行为。无论将其解释为到时停止、移除或结束规划，
在 `t < max_t` 时均与匀速运动一致，仅可能在达到限制后产生差异。`dt` 为 0.25 s
（`configs/sim/kinematic.yaml`），因此 `max_t: 25` 对应第 100 步；在第 100 步前结束的
轨迹展开在各解释下均保持一致。

轨迹展开是否达到第 100 步取决于具体运行结果，因此需要在评估后检查：

```bash
python close_loop/check_maxt_reached.py \
  --results-root outputs/hugsim/pandaset_ltf --dataset pandaset \
  --json-out outputs/hugsim/maxt_check_pandaset.json
```

该脚本从各场景 YAML 读取 `max_t`，从运动学配置读取 `dt`，并从 `planner_stats.json`
读取 `n_steps`。若任一轨迹展开达到 `max_t / dt` 步，脚本会以非零状态退出。此类分数依赖
未明确规定的语义，因此必须单独报告，不应并入均值。

每次完成 PandaSet 评估后都应执行该检查，因为轨迹展开长度取决于规划器，并会随 checkpoint
变化。在所报告的 checkpoint 上，十五个场景中有七个达到上限，最长为 152 步。驾驶性能提高
后，车辆可能在场景中持续更长时间，因此该限制的影响也会增加。此前两个其他驾驶系统的运行
均未超过上限，但其中一个在第 97 步停止，距离 100 步上限较近。

补丁替换现有类的 `__init__`，而非使用子类，因为 `sim/utils/plan.py:111` 根据
`type(controller) is ConstantPlanner` 分派。使用子类会进入 `raise NotImplementedError`
分支，将初始化阶段的异常转变为轨迹展开过程中的异常。

### 缺陷 2 —— 只剩一个路径点时 IDM 的前车距离计算

```text
  File "sim/utils/plan.py", line 104, in plan_traj
  File "sim/utils/agent_controller.py", line 170, in update
IndexError: index is out of bounds for dimension with size 0
```

在 `IDM.update` 内部，前方的弧长是由相邻两点之差构建出来的，然后再补回到完整长度：

```python
lookahead_path = path[nearest_index + 1:][:self.lookahead_path_length]      # n rows
lookahead_lengths = torch.cumsum(torch.norm(
    lookahead_path[1:, :2] - lookahead_path[:-1, :2], dim=1), dim=0)        # n - 1
lookahead_lengths = torch.cat([lookahead_lengths, lookahead_lengths[-1:]])  # back to n
lead_distance = lookahead_lengths[indices_lead]
```

当 `n == 1` 时，差值为空，因此 `cumsum` 和用来补齐它的那个切片都是空的，`lookahead_lengths`
也就一直为空，而 `indices_lead` 却仍然持有索引 0。它需要同时具备：一辆只剩一个 waypoint
的背景车，以及在同一步中位于 `lead_distance_threshold` 内的相邻车辆时触发，因此不会周期性
出现，并会随规划器变化。在所报告的 checkpoint 上，该问题导致一个 nuScenes 场景无法评分；
整个轨迹展开会随之终止。

补丁会在恰好处于那种状态时，把最后一个 waypoint 的一个副本追加到 `path`。前方弧长会变成
`[0.0, 0.0]`，前车距离变为 0，表示前车位于唯一剩余的路径点上。这与 `cat` 对更长路径执行
的填充方式一致。该修改仅影响这一特定状态，原因如下：

- 它只在 lookahead 切片恰好只有一行时才触发，因此其他每一次调用都是上游的调用；
- `nearest_index = argmin(dists)` 不会移动——追加的那一行是最后一行的复制，而 `argmin`
  返回的是第一个最小值；至于最后一行 *就是* 最小值的那种状态，会得到一个长度为 0 的切片，
  而补丁对它不做处理；
- `target` / `is_end` 的扫描会多出一个索引，其坐标与之前的最后一行相等，因此它要么在同一个
  较早的索引处 break，要么落到同一个 `path[-1]`；
- 被复制的 waypoint 可能会让 `is_lead` 的某一列重复，而 `acceleration.min()` 会吸收掉一个
  重复的相同值。

当 `lookahead_path_length == 1` 时，无论 `path` 变得多长，切片都会一直保持只有一行，因此
补丁会报告该情况，而不会将其视为已修复。上游默认值为 100，且没有场景覆盖该值。

### 复现未打补丁的运行

`HUGSIM_COMPAT=0` 会恢复未应用补丁的入口，此时 PandaSet 仅有 127 个场景中的 112 个能够
评分。聚合器会在 PandaSet 注意事项中注明所使用的分母，避免将两类结果直接比较。

## 输出目录结构

```text
outputs/hugsim/
├── nuscenes_base.yaml                    # generated per run, not checked in
├── nuscenes_logs/                        # one log per scenario
├── nuscenes_ltf/                         # results; "ltf" is appended by closed_loop.py
│   └── <scene>_<tier>_<nn>/
│       ├── eval.json                     # the official metrics
│       ├── planner_stats.json            # steps, failures, warm-up padding, latency
│       ├── plan_log.pkl                  # per-step plans
│       ├── dumped_obs.pkl                # only with HUGSIM_DUMP_OBS
│       └── viz/                          # only with SAVE_VIZ=1
│           ├── camera.json               # the camera this run rendered with
│           ├── front/0000.jpg            # CAM_FRONT per step
│           └── input/0000.jpg            # the four views the model reads
├── videos/                               # close_loop/render_videos.py
│   ├── index.md                          # 按数据划分和难度等级组织，并附带分数
│   └── nuscenes/easy/<scene>_easy_00.mp4
├── RESULTS_nuscenes.md
├── results_nuscenes.json
└── RESULTS.md                            # --combined
```

## 环境变量

`scripts/evaluation/run_hugsim_benchmark.sh`：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `HUGSIM_ROOT` | 必填 | HUGSIM 的代码检出目录 |
| `HUGSIM_DATA_ROOT` | 必填 | 数据包 |
| `SIM_PYTHON` | 必填 | 仿真器环境的 python |
| `PYTHON_BIN` | 必填 | WA-JEPA 环境的 python |
| `CHECKPOINT` | 必填 | checkpoint 目录或 `.pt` |
| `CONFIG` | `configs/wa_jepa_hugsim.yaml` | 推理配置 |
| `OUTPUT_DIR` | `outputs/hugsim` | 结果根目录 |
| `SCENARIO_DIR` | `close_loop/configs/benchmark/<dataset>` | 准备好的场景 |
| `HUGSIM_NUM_GPUS` | 自动检测，否则 8 | GPU 数量 |
| `HUGSIM_PER_GPU` | 4 | 每块 GPU 的场景数 |
| `JOBS` | GPU 数 × 每卡数 | 总并发数 |
| `HUGSIM_COMPAT` | 1 | 应用上游兼容性补丁；0 会复现一次未打补丁的运行 |
| `SKIP_DONE` | 1 | 跳过已完成的场景 |
| `SAVE_VIZ` | 0 | 归档逐步的输入和规划 |
| `CAMERA_YAML` | `$HUGSIM_ROOT/configs/sim/<dataset>_camera.yaml` | 相机配置；会记录在 `viz/camera.json` 中 |
| `HD_MAP_PATH`, `HD_MAP_VERSION` | 未设置 | 仅用于 `load_HD_map: true` 的场景 |
| `HUGSIM_ENV_SETUP`, `WM_ENV_SETUP` | 未设置 | 每个环境要 source 的脚本 |
| `WM_RUN_TAG` | 未设置 | 断言必须出现在输出路径中的子串 |
| `WM_EXPECT_STEP` | 未设置 | 对照 `state.pt` 进行断言 |
| `DRY_RUN` | 0 | 打印计划并退出 |

此处不为 `CHECKPOINT` 设置默认值。该变量需要经过三层进程传递；如果任一层未正确传递，
`${CHECKPOINT:-<path>}` 形式的回退可能在无报错的情况下使用非预期 checkpoint 完成全部
436 个场景的评分。
