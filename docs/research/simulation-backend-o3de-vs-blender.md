# 自研仿真底座调研：O3DE 与 Blender 评估

调研日期：2026-10-29
代码基线：本仓库 `main` 分支，commit `e1d7e9d4b`，VERSION 3.0.0

## 结论先行

两个候选都不适合作为 Isaac Sim 的整体替代。Blender 缺少可步进的 GPU 物理与关节系统，应排除；
O3DE 可作为渲染器、编辑器与资产管线的底座，但 GPU 批量物理、Python 张量桥、分块多相机
四项核心能力需从零自建，且无任何先例。更重要的发现是：Isaac Lab 3.0 已经基本脱离了
Omniverse Kit，真正要替换的不是"一个引擎"，而是四项具体能力。建议沿 Isaac Lab 既有的后端
接口自建后端，而非引入游戏引擎重做一遍。

## 一、调研范围与可信度说明

原始问题是"把 Isaac Sim 换成自己的实现，基于 O3DE 还是 Blender"。本文先核实这个问题的前提，
再评估两个候选，最后给出建议。

三点说明：

1. **两个候选都没有机器人强化学习的先例。** O3DE 方向检索 GitHub 组织内"reinforcement
   learning"返回 0 条，仓库检索亦为 0；Blender 方向有合成数据项目但没有任何 RL 在环渲染系统。
2. **部分材料抓取失败**，包括 ROSCon 2023 与 FOSDEM 2025 的演示文稿、gnu.org 的许可证
   FAQ、Blender 基金会 2024 年报 PDF。相关结论来自可靠二手来源，文中标注。
3. **缺少吞吐量数据。** O3DE 的 Atom 离屏渲染、ROS 2 传感器吞吐，以及 Isaac Lab 自身
   Newton 与 PhysX 的对比，都没有公开数字。本文不编造性能结论。

## 二、先核实前提：Isaac Sim 现在还提供什么

这一节的结论改变了问题的形状，建议先读。

### 2.1 Kit 耦合已经很窄

按包统计非测试文件中导入 Kit 专有 API（`omni`、`carb`、`isaacsim`、`usdrt`）的数量，
对比仅导入 `pxr`（OpenUSD）的数量：

| 包 | Kit 专有 API | 仅 pxr | 非测试文件总数 |
| --- | --- | --- | --- |
| isaaclab | 23 | 48 | 510 |
| isaaclab_physx | 22 | 16 | 80 |
| isaaclab_teleop | 8 | 3 | 49 |
| isaaclab_tasks | 4 | 20 | 657 |
| isaaclab_visualizers | 3 | 4 | 20 |
| isaaclab_mimic | 1 | 3 | 49 |
| isaaclab_newton | 0 | 16 | 128 |
| isaaclab_ov | 0 | 15 | 76 |
| isaaclab_rl / _assets / _experimental / _tasks_experimental | 0 | 0 | — |

**真正的硬依赖是 OpenUSD，不是 Kit。** 而 OpenUSD 在无 Kit 模式下由 `usd-exchange==3.0.1`
提供。`source/isaaclab/isaaclab/sim/utils/stage.py` 已经用纯 USD 重新实现了
`omni.usd.check_ancestral` 等便利函数。

什么时候必须启动 Kit，由 `source/isaaclab/isaaclab/app/sim_launcher.py` 的 `validate_launch`
明确枚举：完全没有物理配置、`PhysxAutoCfg` 且未声明 ovphysx 替代、相机使用 default 或
isaac_rtx 渲染器、配置的 `launcher_type` 指向 Kit 启动器、显式指定 `.kit` 体验文件、
启用 livestream、调用方要求 `require_kit`。

### 2.2 无 Kit 路径已经覆盖的能力

| 能力 | Kit 实现 | 无 Kit 实现 | 差距 |
| --- | --- | --- | --- |
| 场景描述 | `omni.usd` 舞台 | 纯 `pxr` Usd.Stage，`isaaclab_ov/stage.py` 内存序列化 | 基本无 |
| 刚体与关节物理 | PhysX via `omni.physics.tensors` | Newton 六种求解器（mjwarp/kamino/featherstone/xpbd/vbd/mpm）；OvPhysX | 求解器行为差异，换求解器需重新调参 |
| 非相机传感器六类 | `isaaclab_physx/sensors` | `isaaclab_newton/sensors` 与 `isaaclab_ov/sensors` 均完整实现 | OvPhysX 有位姿跟踪与静态网格限制 |
| 分块相机 | IsaacRtxRenderer（Replicator） | NewtonWarpRenderer（RGB/反照率/深度/法线/语义/实例分割）；OVRTX（路径追踪，含运动矢量与标签） | Newton 渲染器无运动矢量与 RTX 材质传输 |
| 可视化 | Kit 视口与 `omni.ui` | newton_gl、newton_rtx、rerun、viser | Kit UI 控件（实时曲线等）无对应物 |
| URDF/MJCF 导入 | `isaacsim.asset.importer.*` | 回退到独立 `isaacsim-asset-isolated` wheel | 该 wheel 仍是 NVIDIA 二进制，非开源 |
| 资产下载 | `omni.client` | 同一个 `omniverseclient` pip wheel | 仍依赖 NVIDIA 托管的存储桶 |

仍然是 Kit 独有的：Kit 视口与 UI 控件、基于 Replicator 的随机化、网格与 instanceable 转换器、
WebRTC 推流、Kit XR 头显渲染、Fabric/usdrt 场景数据、surface gripper、版本探测。
**这些都不在 RL 训练热路径上**，唯一的例外是基于 Replicator 的域随机化事件（17 个文件引用
`omni.replicator`）。

任务覆盖方面，生成的环境浏览器中 44 个训练环境全部至少有一个无 Kit 物理预设
（newton_mjwarp 37 个、ovphysx 37 个、newton_kamino 18 个），9 个相机任务提供三种渲染器预设。

### 2.3 但无 Kit 不等于脱离 NVIDIA

这是本节最关键的限定。即便完全不用 Kit，依赖链里仍有：`ovphysx`、`ovrtx`、`ovstage`、
独立导入器 wheel、`omniverseclient` 五个 NVIDIA 二进制包；硬件下限是 CUDA 计算能力 7.5
以上；渲染需要 Vulkan ICD。更根本的是 Warp——详见同目录的
[国产芯片生态适配调研](domestic-accelerator-adaptation.md)，其中统计了 523 个非测试 Warp
内核与 2,968 处 `ProxyArray` 引用。

所以"替换 Isaac Sim"和"摆脱 NVIDIA"是两个不同目标，前者已经完成了大半，后者一步未动。

### 2.4 问题的真正形状

综合以上，一个替代底座需要满足 Isaac Lab 核心已定义的五组接口：

1. 消费 USD 舞台的 `PhysicsManager` 与 `SceneDataBackend`
2. 产出九种输出缓冲的分块相机 `BaseRenderer`
3. 一个 `VisualizerBackendCfg`
4. 六类传感器后端
5. 一个资产获取器

Newton 与 OvPhysX 两套独立后端已经分别实现了这组接口，这降低了第三套的风险。
**问题不是"要不要换引擎"，而是"自建后端"与"引入游戏引擎再自建后端"之间的选择。**

## 三、候选一：O3DE

### 3.1 项目健康度

截至 2026-10-29 的实测数据（GitHub API 当日查询）：

| 指标 | 数值 |
| --- | --- |
| 最新版本 | 26.05.0（2026-05-27），节奏为每年两个大版本 |
| Star / Fork | 9,801 / 2,556 |
| 未关闭 issue 与 PR | 3,519 |
| 2025-01 至今 development 分支提交 | 981，其中 2026 年 445 |
| 同期独立提交者 | 88 人 |
| 历史贡献者总数 | 269 |
| 2025 年下载量 | 112,354 |
| 完整 CI 一次运行 | 约 207 分钟 |

治理在 Linux Foundation 下的 Open 3D Foundation，执行总监为 Joe Bryant。Premier 成员名单
（最后一份官方名单为 2022 年）包括 Adobe、AWS、Epic、华为、Intel、微软、Niantic、
腾讯 LightSpeed、OPPO。

两点需要注意。其一，**贡献者基数偏薄**：22 个月 88 人，单个个人贡献者占约 15% 的提交，
存在真实的 bus factor 风险。其二，**华为是 2025-2026 提交量最大的企业贡献者**
（三个账号合计 132 次提交，超过 AWS 系账号），工作集中在 Atom 渲染器、多 GPU、粒子系统、
光追材质。这对国产化是正面信号，但其投入方向是图形而非机器人。基金会官网 o3df.org 的新闻
停留在 2024 年 10 月，市场运营明显弱于工程。

### 3.2 机器人栈

由 Robotec.ai 开发，原生 ROS 2（仿真进程本身就是一个 rclcpp 节点，无需桥接），仅支持 Linux，
测试覆盖 Humble/22.04 与 Jazzy/24.04。

传感器：RGB 与深度相机、2D/3D 激光雷达、IMU、里程计、GNSS、接触。
导入器支持 URDF、XACRO、SDFormat，产出 prefab，可选用 PhysX 缩减坐标 articulation 或
刚体加关节。实现了 ROS 2 `simulation_interfaces` 标准。

两个限制值得记录：**没有 MJCF 导入器**；GPU 激光雷达 RGL 依赖 CUDA 与 OptiX，要求
NVIDIA Turing 以上，且其 UDP 扩展是闭源的。

对我们而言，这整套 ROS 2 控制面在 Isaac Lab 场景下都会被绕过，能复用的只有导入器，
可能还有部分传感器管线。

### 3.3 物理：这是最大的问题

O3DE 当前内置 PhysX SDK **5.1.1**，从源码可以确认：场景创建只设置了 CPU dispatcher
（`sceneDesc.cpuDispatcher = physXSystem->GetPxCpuDispathcher()`），`PhysXSystem.cpp` 中
**没有任何 CUDA 上下文管理器或 GPU 动力学代码**。

因此：

- **没有 GPU 刚体管线**
- **没有批量张量 API**，也就没有 Isaac Lab `ArticulationView` 那类批量 GPU 状态读写
- **没有可变形体**，只有 NvCloth
- 官方文档明确承认"O3DE 中的 PhysX 仿真不保证确定性"

多物理场景在一个进程内是支持的，但每个都是独立的 CPU PhysX 场景，不是向量化的环境批次。
把上千个并行环境建成 O3DE 实体，会先撞上实体组件系统的开销与 CPU PhysX 的瓶颈，
远早于触及 GPU 上限。

要达到 Isaac Lab 的物理能力，需要：升级第三方 PhysX 包到 5.3 或更高、启用 CUDA 动力学、
自行编写批量视图层——**实质是把 Isaac Sim 的 physics tensor 层重新实现一遍**。
或者在 AzPhysics 旁边并行接入 Newton/MuJoCo-Warp，但 O3DE 的物理抽象是按实体按组件组织的，
批量 GPU 引擎只能"并排接入"而非"替换到抽象之下"。

### 3.4 渲染

Atom 支持 DX12、Vulkan、Metal、Null 四种 RHI，有硬件光追管线（24.09 加入相交着色器，
华为在 2025 下半年重构光追材质），25.05 起多 GPU 支持成熟，子通道带来约 40% 带宽优化。

现有的 ROS 2 相机传感器展示了所需模式：每个相机一条 `RPI::RenderPipeline`（Color 或 Depth
模板），再通过 `AttachmentReadback` 回读到 CPU。**但这恰恰是 Isaac Lab 中已知扩展性差的
`Camera` 模式，而不是 `TiledCamera` 模式。**

缺口明确：没有分块多相机渲染、没有 GPU 常驻的传感器输出、没有 CUDA 与 Vulkan 互操作。
做零拷贝到 PyTorch 需要在回读或渲染目标上加 `VK_KHR_external_memory_fd` 导出，
Python 侧用 `cudaImportExternalMemory` 接收。在 Linux 加 NVIDIA 上可行，但完全是自研工作。
渲染确定性没有任何文档提及，有 TAA 等时域效果和异步资产流送的情况下需要专门工程化。

无头运行是支持的：`-NullRenderer -rhi=null` 完全不需要 GPU，但相机传感器需要真实 RHI，
此时用 `-console-mode`。

### 3.5 OpenUSD

O3DE 的原生场景格式是自有的 JSON prefab。USD 支持的现状是：2024-10-04 合入了一个导入器，
仅支持"简单的、无引用的 USD 文件"；2025-03-20 合入了"USD in the Scene Pipeline"的 RFC；
"改进 OpenUSD 支持"列在 2025 下半年路线图上，但 26.05 的发布说明里没有相关条目。
Omniverse 兼容性 issue #15848 自 2023 年开启至今未关闭。

**没有 USD 运行时舞台 API，没有 USD 导出，没有 Omniverse 连接器。**
由于已合入的导入器只处理扁平无引用文件，它无法读取 Isaac Sim 风格的组合舞台。

这对我们是重大缺口：Isaac Lab 的资产全部是 USD（Python 源码中 `.usd` 引用 327 处、
`.usda` 176 处，`ISAAC_NUCLEUS_DIR` 使用 208 处），且依赖 UsdPhysics 与 PhysX/Newton/MJC
schema 属性。要么写一个 USD 到 O3DE prefab 的翻译器并映射 PhysX schema，要么放弃 USD
作为舞台格式、改用 prefab 加 URDF/MJCF，后者意味着重新编排约 400 处资产引用。

### 3.6 脚本与嵌入

Python 在 O3DE 中是**编辑器侧工具**：内嵌 CPython 3.10.13（EditorPythonBindings gem），
配 PySide6 做 Qt 工具，另有 PythonAssetBuilder 用于资产管线。
运行时脚本是 Lua 5.4 与 Script Canvas。

**没有受支持的运行时 Python API，没有"把 O3DE 当模块 import"的嵌入模式，没有 CUDA 或
PyTorch 互操作。** 文档中的进程外接口只有 ROS 2 的话题与服务。

现实的集成形态是反过来的：O3DE 作为 C++ 进程运行，我们自己用 pybind11 或共享内存暴露接口。
Isaac Lab 3.0 已经为 Newton 解耦了 Kit，架构上接受一个暴露张量状态的 C++ 服务是可行的，
但 O3DE 今天一样都不提供。

### 3.7 许可与构建

引擎与一方 gem 为 Apache-2.0 或 MIT 双许可，PhysX 5 SDK 为 BSD-3，机器人 gem 与
`simulation_interfaces` 为 Apache-2.0。**对闭源商业产品没有障碍**，这明显优于
Isaac Sim 的 NVIDIA Omniverse 许可。需要注意的是编辑器依赖 Qt 6.11.2（LGPL，单体运行时
已排除 Qt）、Wwise 为可选专有组件、RGL 的 UDP 扩展闭源且 GitHub 上许可标注为 Other。

构建方面：Ubuntu 22.04 为主（24.04"应该可用"），x86_64 与 aarch64 均有包，
Clang 14 起步（下个版本要求 18），需要 40 至 100 GB 以上磁盘、每编译线程约 2 GB 内存，
使用自有的第三方二进制包系统（不是 vcpkg 或 conan）。

## 四、候选二：Blender

### 4.1 基本情况

每年三个版本加一个 LTS（支持两年）。4.5 LTS 于 2025-07-15 发布，5.0 于 2025-11-18，
5.1 于 2026-03-17（Python 3.13），5.2 LTS 于 2026-07-14（支持至 2028-07），
当前为 5.2.3（2026-10-22）。基金会 2024 年收入 310 万欧元，但当年经营亏损、储备低于三个月
目标。赞助方包括 OTOY、Chaos、Netflix、Dell 等，NVIDIA、AMD、Intel、高通、Epic 已是顶级赞助。
**路线图上没有任何机器人或仿真相关条目。**

### 4.2 物理：直接否决项

Blender 的物理只有面向动画的 Bullet 刚体（8 种约束）、CPU 布料与软体、Mantaflow 流体，
以及 5.2 引入的实验性 XPBD 几何节点求解器（仅布料与毛发）。

关键缺陷：

- 物理按场景帧求值，通过点缓存驱动；步进的唯一方式是 `frame_set` 循环，
  **没有不推进帧而单独步进的 API**
- 官方 bug #83594 描述其为"不确定且基本不可用"
- **没有 articulation，没有关节驱动，没有 GPU 物理**
- 游戏引擎 BGE 已在 2.80（2018）移除；UPBGE 分支（0.50，2026-01，基于 Blender 5.0）
  仍是 Bullet，且非固定时间步

所有真实的机器人相关项目都把物理放在 Blender 之外：Kubric 用 PyBullet，MuBlE 用 MuJoCo，
BlenderProc 不做动力学。

### 4.3 渲染

Cycles 支持 CUDA、OptiX、HIP、oneAPI、Metal，可无头运行且不需要 OpenGL，
是 Apache-2.0 许可（这一点有复用价值），输出通道齐全含 Cryptomatte。
但每次渲染要静态重建 BVH，小分辨率下 BVH 重建与场景同步开销占主导
（有报告为 BVH 20 秒对渲染 7 秒）；多 GPU 不能合并显存（NVLink 除外）。

EEVEE 在 4.2 重写，Vulkan 后端在 4.5 达到"与 OpenGL 相当"但仍非默认。
检索到的官方开发者表态是"即便后台渲染也必须有一个显示设备"，无显示器运行仅有间接证据
（第三方 Docker 镜像，基于 4.3）与 5.2 新增的 `gpu.init()`。

**没有任何分块多相机接口**，GPU 数据回读经由 CPU（`GPUTexture.read`），无法零拷贝给 PyTorch。

### 4.4 OpenUSD

视觉层面的 USD 读写质量不错（UsdSkel、实验性点实例化、MaterialX 与 OpenPBR 导出）。
**但完全不支持 UsdPhysics 读写**，NVIDIA 方面确认 Blender 导出的骨架 USD 缺少
articulation root、关节与刚体 API。

NVIDIA 已停止提供定制 Blender 构建，改为针对官方 LTS 发布 Omniverse 插件；
其 SimReady 插件 v2026.4.0 可以编写 UsdPhysics 关节并导入 MJCF，但仅支持 Windows。

### 4.5 bpy 作为库

PyPI 上的 bpy 5.2.3 **只支持 Python 3.13**，Linux 需 glibc 2.28 以上。
文档明确：等价于进程内的 `--background` 模式；`importlib.reload` 会抛异常；
一个进程只能有一个 .blend；其他使用 GPU 的模块可能导致 Cycles 无法使用 GPU；
**Python 线程"会导致 Blender 崩溃"**，建议用多进程。
bpy 4.0 至 4.2 的多帧 Cycles 循环有内存泄漏报告（#129384）。

结论是一个进程一个实例，N 个环境就是 N 个进程。

### 4.6 许可：必须注意的边界

Blender 源码为 GPL-2.0-or-later，二进制为 GPLv3+。基金会立场是使用其 API 的已发布脚本
必须 GPL 兼容。Cycles 单独为 Apache-2.0。

按 FSF 的口径：进程内链接构成合并作品；fork/exec 加非紧密 IPC 可视为独立程序，
但"用共享内存传递复杂数据结构"被认为"基本等同于动态链接"。
自用与 SaaS 不触发分发义务，对外交付则触发。

Isaac Lab 自身是 BSD-3-Clause，两者不是一个量级的约束。如果产品需要对外交付，
把 bpy 嵌入进程是有实质法律风险的。

### 4.7 可用的位置

Blender 作为底座不成立，但以下三处有价值：

- **资产与场景编辑前端**，配合 USD 导出（物理属性需另行补齐）
- **离线合成数据渲染器**，以独立进程运行，参考 BlenderProc2、Kubric、Infinigen
- **Cycles 内核或 Hydra delegate 的来源**，它是 Apache-2.0，可单独复用

需要注意性能量级：BlenderProc2 每个场景 5 至 20 张图，Infinigen 每对 1080p 立体图约 3.5 小时
（二手来源）。这是离线数据生成的量级，不是 RL 在环渲染的量级。

## 五、两个候选的对比

| 维度 | O3DE | Blender |
| --- | --- | --- |
| GPU 批量物理 | 无（PhysX 5.1.1 仅 CPU），可改造 | 无，且架构上不可能 |
| 关节与驱动 | 有（PhysX 缩减坐标 articulation，含驱动与力矩传感） | 无 |
| 可独立步进 | 是 | 否，必须推进帧 |
| 确定性 | 官方声明不保证 | 官方 bug 描述为不确定 |
| 分块多相机 | 无，但 RPI 多管线多视图是可行基础 | 无 |
| GPU 常驻输出 | 无，现为 CPU 回读 | 无，现为 CPU 回读 |
| 光追 | 有硬件光追管线 | Cycles 成熟，但每次重建 BVH |
| OpenUSD | 仅扁平无引用文件导入，无运行时 API，无导出 | 视觉层较完整，**完全无 UsdPhysics** |
| Python 嵌入 | 仅编辑器侧，运行时无 Python | bpy 可作库，但单进程单实例且线程不安全 |
| URDF 导入 | 有（含 XACRO、SDFormat），无 MJCF | 无（Phobos 做编辑，非运行时） |
| 许可 | Apache-2.0 或 MIT，对闭源无障碍 | GPL，嵌入有实质风险 |
| 机器人社区 | 有，但集中在 Robotec.ai 一家，面向 ROS 2 系统级仿真 | 无，只有合成数据项目 |
| RL 先例 | 0 | 0 |
| 国产化相关 | 华为为最活跃企业贡献者（图形方向） | 无 |

## 六、必须列出的第三类选项

原题只给了两个候选，但调研发现的事实使得不列出以下选项就无法做出合理决策。

当前真正做到"无 Kit 且 GPU 批量"的开源栈只有两个家族：

- **Newton 加 MuJoCo-Warp**（Apache-2.0，读 USD、URDF、MJCF），即 Isaac Lab 现在的
  Newton 后端。
- **SAPIEN 与 ManiSkill3**（Apache-2.0，PhysX GPU 但不用 Kit，读 URDF 与 MJCF，
  **不支持 USD**）。公开对比称在真实分辨率相机下约为 Isaac Lab 的 2 倍速度、显存少 2 至 4 倍，
  但在 128 至 256 像素多环境场景下 Isaac Lab 反而快约 1.25 倍；论文作者注明该对比中
  Isaac Lab 使用了光线追踪，并非同等条件。

**Genesis** 是唯一活跃的多厂商 GPU 引擎（经自有编译器 Quadrants 支持 CUDA、ROCm、Metal、
Vulkan），v1.4.5 发布于 2026-10-22，约两周一个版本，可读 URDF、MJCF、USD。
但其"4300 万 FPS"的标题数字是单平面 Franka 微基准，存在争议，且没有 RL 任务级吞吐数据，
也没有 Isaac Lab 集成。

其他栈（Unity、Unreal 系、Gazebo、Webots、CoppeliaSim、PyBullet）均为 CPU 单进程物理，
没有任何公开的批量 RL 吞吐数据。Madrona-MJX 已废弃并指向 MJWarp 渲染器。
Isaac Gym 已停止支持。

**离开 Omniverse 的两种既有模式**：保留 PhysX 但去掉 Kit（ManiSkill3、NVIDIA 自己的
OvPhysX）；连 PhysX 一起换掉（MuJoCo Playground、RoboCasa365、Genesis、Motphys MotrixSim）。
Isaac Lab 3.0 现在在同一套任务 API 下同时提供了这两种。

**国产相关的一个重要先例**：国内头部人形机器人厂商的大型开源仿真项目——智元 Genie Sim 3.0
（CES 2026 开源）、银河通用、上海 AI Lab 的 InternUtopia——目前仍然建立在 Isaac Sim 之上。
真正走厂商独立路线的是 Motphys 的 MotrixSim（Rust 实现的 CPU 核心，Python API，兼容 MJCF，
Apache-2.0 的 MotrixLab），其 v0.6.0 通过鲲鹏认证，并在 2026-10 前后与华为联合发布了
面向国产算力的具身智能仿真底座方案。从报道看，其动机正是非 NVIDIA 算力兼容性。

**没有任何第三方在更换引擎的同时保持 Isaac Lab 任务 API 兼容**，唯一做到这一点的机制是
Isaac Lab 3.0 自己的后端与预设系统。

## 七、建议

### 7.1 不推荐把 O3DE 或 Blender 作为整体底座

Blender 应直接排除，理由见第四节，物理架构上不可能承担。

O3DE 不推荐作为整体底座，但不是因为它不好，而是因为**代价与收益不匹配**。沿这条路要自建的
五项工作，每项都是数个工程师季度的量级，且都没有上游先例可借：

1. GPU 物理：升级 PhysX 包、启用 CUDA 动力学、编写批量 articulation 与刚体视图 API；
   或在 AzPhysics 旁并行接入 Newton/MuJoCo-Warp
2. Python 与张量桥：C++ 应用加 pybind/nanobind，传感器需 CUDA 与 Vulkan 互操作
3. 渲染：GPU 常驻输出的分块多相机管线（RGB、深度、分割、法线、光流）
4. 资产：组合 USD 或 MJCF 导入路径，以及 PhysX schema 映射
5. 多环境实例化与重置，且不能依赖实体创建销毁

而这五项中的第 1、3、4 项，Isaac Lab 的 Newton 后端已经有了可用实现。

### 7.2 推荐路线：沿既有后端接口自建

核心理由在 2.4 节：Isaac Lab 核心已经定义了五组后端接口，并且已有两套独立实现
（Newton 与 OvPhysX/OVRTX）证明这组接口可以被第三方实现。自建后端的工作量显著小于
"引入游戏引擎再自建后端"，且能保持全部任务、管理器、RL 适配层不变。

建议的优先级：

1. **先补齐"去 NVIDIA 二进制"的部分**。当前无 Kit 路径仍依赖 ovphysx、ovrtx、ovstage、
   独立导入器、omniverseclient 五个闭源 wheel。逐个评估：导入器可用开源方案替代
   （urdfpy、mujoco 自带解析器）；资产托管可自建；OVRTX 与 OvPhysX 可暂时用 Newton 侧
   对应能力顶替。这部分收益明确、风险低。
2. **再处理 Warp 依赖**。这是与国产芯片目标的交汇点，见
   [国产芯片生态适配调研](domestic-accelerator-adaptation.md)。
3. **渲染与编辑器按需引入 O3DE**。如果确实需要高保真渲染器与场景编辑器，
   O3DE 是两个候选中更合适的，其 Apache/MIT 许可、Vulkan 跨厂商能力、
   已有的 URDF/SDF 导入器和华为的活跃投入都是正面因素。但应把它定位为
   **渲染与编辑组件**，而不是仿真底座。

### 7.3 如果仍要选一个作为底座

若因为非技术原因必须在两者中选一个，选 O3DE，不要选 Blender。
O3DE 至少具备 articulation、可独立步进、跨厂商 Vulkan 渲染、宽松许可与 URDF 导入；
Blender 在物理这一项上是架构性的不可能，而不是工作量问题。

## 八、风险登记

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| O3DE 贡献者基数薄（22 个月 88 人，单人占 15%） | 上游演进速度不可依赖 | 若采用，按"需自行维护分支"做预算 |
| O3DE 物理非确定性 | sim-to-real 与复现实验受影响 | 需专门工程化，或物理不走 O3DE |
| O3DE 的 USD 仅支持扁平文件 | 约 400 处资产引用需重新编排 | 自写 USD 到 prefab 翻译器 |
| Blender GPL 嵌入 | 对外交付有法律风险 | 仅以独立进程离线使用 |
| 五项自建工作无先例 | 工期估算不可靠 | 设置阶段性中止点 |
| 无 Kit 路径 CI 覆盖薄（482 个测试文件中仅 12 个文件 19 处 kitless 标记） | "已支持"不等于"已验证" | 自建后端前先补该方向测试 |

## 九、待验证事项

关于本仓库：

- [ ] **任务默认物理后端的完整审计。** 调研笔记称 `isaaclab_tasks` 中只有五处
      `default="physx"` 字符串、零处 `default="newton_*"`，据此判断多数任务默认走 Kit。
      但我实际核对 `source/isaaclab_tasks/isaaclab_tasks/core/cartpole/cartpole_common.py:53`
      为 `default: NewtonCfg = newton_mjwarp`，即存在以字段引用（而非字符串）指定的 Newton
      默认值，上述 grep 会漏掉这类写法。**结论尚不确定，需要完整审计后再下判断。**
      仓库文档内部也互相矛盾：`pyproject.toml` 称 Newton 为默认物理引擎，
      而 `docs/source/concepts/visualization.rst:137` 称"多数任务默认 PhysX 后端，需要 Isaac Sim"。
- [ ] 151 处 `gym.register` 中，除环境浏览器收录的 44 个之外，哪些是 `-Play` 副本、
      哪些是真正仅 Kit 可用的任务。
- [ ] 各无 Kit 预设的实际通过状态（仓库未记录）。
- [ ] Newton、PhysX、OvPhysX 三者以及三种渲染器的实测吞吐。仓库内有基准框架
      （`source/isaaclab/isaaclab/benchmark/`）但未提交任何结果，上游也没有公开数字。
      这是所有性能判断缺失的共同根源，建议尽早自测。
- [ ] 五个 NVIDIA 闭源 wheel（ovphysx、ovrtx、ovstage、isaacsim-asset-isolated、
      omniverseclient）各自的可替代性与替代成本。

关于 O3DE：

- [ ] Atom 离屏渲染吞吐（给定分辨率的帧率、多相机扩展性）。无任何公开数据。
- [ ] O3DE 在上千实例规模下的实体与场景开销。无基准。
- [ ] 2025-2026 的付费成员名单，以及 AWS 投入是否下降。只能从提交数据推断，无一手确认。
- [ ] RGL 的确切许可文本（GitHub 显示为 Other），打包前必须确认。
- [ ] "改进 OpenUSD 支持"（2025 下半年路线图）是否已在 25.10 或 26.05 交付。
- [ ] 是否有人把 MuJoCo、Newton、Jolt 或 Bullet 接入过 O3DE。未找到任何 gem。
- [ ] O3DE 人才池规模，特别是国内。无任何公开数据。

关于 Blender：

- [ ] 4.2 至 5.2 期间无显示器运行 EEVEE 的官方表态。只找到间接证据。
- [ ] 低分辨率下的单帧延迟基准。无数据。

## 十、参考来源

本仓库现状：

- `source/isaaclab/isaaclab/app/sim_launcher.py`（`validate_launch`，第 352 至 400 行）
- `source/isaaclab/isaaclab/sim/utils/stage.py`（纯 USD 舞台实现）
- `source/isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py`
- `source/isaaclab/isaaclab/renderers/output_contract.py`（九种输出缓冲契约）
- `docker/Dockerfile.kitless`、`.github/workflows/kitless-docker.yml`
- `docs/source/concepts/renderers.rst`、`docs/source/concepts/backends_and_presets.rst`

O3DE：

- 26.05.0 发布说明 https://docs.o3de.org/docs/release-notes/2605-0-release-notes/
- PhysX 文档（确定性声明）https://docs.o3de.org/docs/user-guide/interactivity/physics/nvidia-physx/
- PhysX5 Linux 包定义
  https://raw.githubusercontent.com/o3de/o3de/development/Gems/PhysX/Core/PhysX5/Source/Platform/Linux/PAL_linux.cmake
- 机器人概览 https://docs.o3de.org/docs/user-guide/interactivity/robotics/overview/
- 部署仿真（无头运行）https://docs.o3de.org/docs/user-guide/interactivity/robotics/deploying-simulation/
- 相机传感器实现
  https://raw.githubusercontent.com/o3de/o3de-extras/development/Gems/ROS2Sensors/Code/Source/Camera/CameraSensor.cpp
- USD 导入 PR #18361 https://github.com/o3de/o3de/pull/18361
- Omniverse 兼容性 issue #15848 https://github.com/o3de/o3de/issues/15848
- 编辑器 Python 绑定
  https://docs.o3de.org/docs/user-guide/gems/reference/script/python/editor-python-bindings/
- 2026 路线图与 2025 回顾 https://o3de.org/roadmap/ ，
  https://o3de.org/a-look-back-at-2025-and-what-to-look-forward-to-in-2026/
- Robotec GPU Lidar https://github.com/RobotecAI/RobotecGPULidar

Blender：

- bpy 模块文档与 PyPI https://pypi.org/project/bpy/
- Blender 5.2 LTS 发布说明 https://www.blender.org/download/releases/
- 物理确定性 bug #83594 https://projects.blender.org/blender/blender/issues/83594
- BlenderProc2 https://github.com/DLR-RM/BlenderProc
- Kubric https://github.com/google-research/kubric
- NVIDIA SimReady Blender 插件（UsdPhysics 关节编写）
  https://docs.omniverse.nvidia.com/

替代栈：

- MuJoCo MJWarp 文档 https://mujoco.readthedocs.io/en/stable/mjwarp/
- MuJoCo Playground 论文 https://arxiv.org/html/2502.08844v1
- ManiSkill3 论文 https://arxiv.org/html/2410.00425v2
- Genesis https://github.com/Genesis-Embodied-AI/Genesis
- Madrona-MJX（已废弃）https://github.com/shacklettbp/madrona_mjx
- 智元 Genie Sim 3.0 https://github.com/AgibotTech/genie_sim
- Motphys MotrixSim https://pypi.org/project/motrixsim/ ，
  Motphys 与华为联合方案报道
  https://autonews.gasgoo.com/articles/news/motphys-and-huawei-jointly-unveil-embodied-intelligence-simulation-base-solution-2102741730670886912
