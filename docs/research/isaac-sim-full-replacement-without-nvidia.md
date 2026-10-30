# Isaac Sim 全功能替代调研：以 OpenUSD 为底、O3DE 或 Blender 为壳、不依赖 NVIDIA 生态

调研日期：2026-10-29（第三轮）
代码基线：本仓库 commit `e1c21cfe6`（Isaac Lab 3.0.0，对应 Isaac Sim 6.2.0.0）
前两轮：[O3DE 与 Blender 的编辑与渲染替代](simulation-backend-o3de-vs-blender.md)、
[国产芯片生态适配](domestic-accelerator-adaptation.md)

## 结论先行

"用 O3DE 或 Blender 实现 Isaac Sim 的全部功能"这个提法里，三个名词都需要先修正，修正之后
问题才可解：

1. **Isaac Sim 不是一个软件，是 13 组能力加一个资产库。** 按 6.2 的源码树与许可文件逐项核对，
   其中约三分之二的 Python/C++ 扩展已是 Apache-2.0，真正闭源的是 Kit 内核、omni.ui、
   OmniGraph、PhysX 二进制、RTX 渲染器、Replicator、RTX 传感器 SDK、XR 运行时、MDL 材质库，
   以及**全部 3D 资产**。闭源部分恰好是"生态"本身。
2. **O3DE 和 Blender 都不能当底座，两轮调研已经证明。** 但它们各自能承担的角色是清楚的：
   Blender 承担资产创作与离线超写实渲染，O3DE 只在需要一个 Apache 许可的实时 Vulkan 渲染器
   与编辑器外壳时有用。底座仍然是 Isaac Lab 自己的后端接口加 Newton 加 OpenUSD。
3. **"不依赖 NVIDIA 生态"必须分级。** 用户的定义是"可以用 N 卡，不用 Omniverse"。按这个定义，
   CUDA、Warp、Newton、NVIDIA 开源的 Apache-2.0 转换器都可以用，于是物理层基本已经完成，
   缺口集中在渲染、传感器、合成数据、资产与工具链。若把 Warp 也算作生态，
   问题就变成另一份文档（国产芯片适配）所讨论的内核层移植，工作量上升一个数量级。

在"可用 N 卡、不用 Omniverse、保留 Warp"这一级定义下，需要做的事情归为 13 条工作线
（第五节）。其中 4 条是纯粹的替换（资产传输、网格转换、部署镜像与基准记录器、物理层的吸盘与预设补齐），
6 条是中等规模的自建（Blender 物理导出插件、合成数据与域随机化、传感器物理模型、ROS 2 桥、
运动生成替换、遥操作与 XR、可视化与检视），3 条是大型自建且没有现成件（覆盖 15 项输出契约的
渲染器、厂商中立的实时光追渲染器、资产库本身）。粗估不含实时光追渲染器约 34 至 64 人月，
无先例可校准。

资产库是最容易被忽视、也最没有技术解法的一条：NVIDIA 的机器人、传感器与环境资产
按"附加软件与材料许可"只能与 Isaac Sim/Isaac Lab 在 NVIDIA GPU 上使用，不得修改与再分发，
脱离生态意味着**资产要重建**。

## 一、问题定义

### 1.1 "不依赖 NVIDIA 生态"的三级定义

| 级别 | 排除 | 保留 | 适用文档 |
| --- | --- | --- | --- |
| **L1：脱离 Omniverse** | Kit 与全部 `omni.*` 扩展、Nucleus/`omni.client`、Replicator、RTX 渲染器与 RTX 传感器 SDK、MDL、OVRTX/OvPhysX/ovstage 专有 wheel、`isaacsim` 系列 wheel、Isaac Capture/CloudXR、cuMotion/Lula、NuRec、NVIDIA 资产库 | NVIDIA GPU 与 CUDA 驱动；NVIDIA 维护但 Apache-2.0 的开源件：Warp、Newton、`urdf-usd-converter`、`mujoco-usd-converter`、`usd-exchange`、cuRobo 源码、PhysX SDK 源码（根目录 BSD-3，`physx/` 自 5.11 起 Apache-2.0） | **本文主线** |
| **L2：脱离 NVIDIA 维护的开源件** | L1 加 Warp、Newton、usd-exchange、NVIDIA 转换器、PhysX SDK | CUDA 驱动、第三方引擎（MuJoCo、Genesis 等） | 本文标注增量 |
| **L3：脱离 CUDA** | L2 加 CUDA 本身 | 国产或 AMD 加速卡 | [国产芯片生态适配](domestic-accelerator-adaptation.md) |

用户给出的约束（"可以使用英伟达的显卡，Omniverse 等都不用，但使用 OpenUSD"）对应 L1。
本文按 L1 展开，凡 L2 会额外增加的工作在对应小节标出。

L1 的边界有两处必须说明：

- **Warp 是 NVIDIA 维护的。** 上游在 2026-10-20 关闭了 AMD 的 HIP 后端 PR，态度明确。
  保留 Warp 意味着接受"物理内核只能跑在 CUDA 或 CPU 上"，这与"可以用 N 卡"一致，
  但与国产芯片目标冲突。两份文档的交汇点在这里，不在本文范围内重复。
- **`usd-exchange` 与两个 NVIDIA 转换器是 Apache-2.0 但由 NVIDIA 发布**，而且 Isaac Lab
  的 `pxr` 运行时目前就是由 `usd-exchange` 提供的（它内置 OpenUSD 26.08）。L1 下可以继续用，
  但要知道它们的发布节奏由 NVIDIA 控制；L2 下要换成 `usd-core` 官方 wheel，
  并且自己维护转换器。

### 1.2 Isaac Sim 6.2 的功能面与许可结构

对 `isaac-sim/IsaacSim` 仓库（`release/6.3` 分支，`VERSION` 6.2.0-rc.14，2026-10-27 最后推送）
的源码树、`deps/*.packman.xml`、`LICENSE` 与文档逐项核对后，许可结构如下：

- **Apache-2.0**：`source/` 下全部内容，含 `source/extensions` 的 142 个扩展目录、
  `source/deprecated` 的 24 个，以及新的无 Kit 层 `source/libraries`。
- **"NVIDIA Isaac Sim 附加软件与材料许可"（闭源）**：Kit 内核（`kit-kernel 110.4.0`，
  packman 标记 `non-redist`）、`carb`、`omni_client_library`、`omni_physics`（PhysX 5.4.2）、
  Replicator 1.13.38、cuMotion、`nvidia_lula_no_cuda`、RTX 传感器 SDK，**以及"3D 模型与纹理"**。
  许可条款：只能在 NVIDIA GPU 系统上使用；"将 Kit 用于 Isaac Sim 或 Isaac Lab 之外的用途不属于
  许可目的"；禁止修改、衍生、再分发、逆向；任一方可提前 30 天通知终止。
- **无 Kit 的专有 wheel**：`ovrtx 0.6.0`、`ovstage 0.3.0`（`LicenseRef-NvidiaProprietary`），
  `ovphysx 0.7.0`、`physx-usd-schemas 25.11.2`（`LicenseRef-NVIDIA-Omniverse`），
  `omniverseclient 2.75.1`。
- **无 Kit 的开源 wheel**：`newton`、`newton-usd-schemas`、`urdf-usd-converter 0.3.4`、
  `mujoco-usd-converter 0.8.0`、`usd-exchange 3.0.1`、`simready-foundation-tier-*`、
  `isaacsim-asset-isolated 6.2.0.1`、`isaacsim-robot-schema`、`isaacteleop 1.5.175`（均 Apache-2.0）。
  PhysX SDK 仓库 BSD-3，但 Kit 内的 `omni.physx` 二进制闭源；cuRobo 源码 Apache-2.0，
  但 `cumotion` wheel 是闭源 packman 产物。

下表按 13 组能力列出 Isaac Sim 提供什么、其中哪些闭源、Isaac Lab 用到多少、无 Kit 路径现状。
"O"为仓库内 Apache-2.0 源码，"C"为闭源二进制，"O→C"为开源包装但离开闭源二进制不能工作。

| # | 能力组 | Isaac Sim 6.2 提供 | 许可 | Isaac Lab 实际使用 | Isaac Lab 无 Kit 现状 |
| --- | --- | --- | --- | --- | --- |
| 1 | 应用框架 | Kit 内核、扩展系统、`.kit` 体验文件、`omni.ui`、`carb`、Fabric/usdrt、OmniGraph | C | 启动器、Fabric 通知、Kit 视口、`omni.ui` 曲线控件 | 已有 `sim_launcher` 无 Kit 启动；Fabric 与 UI 控件为 Kit 专有 |
| 2 | 舞台与创作工具 | Stage/Layer/Property 面板、物理检视器、机器人装配器、增益调参、抓取编辑器、位姿器、自碰撞检测、系统辨识、占据图生成、资产校验、SimReady 浏览器、Stage Recorder、语义编辑器 | 面板 C；机器人工具 O（但建在 `omni.ui` 上） | 文档推荐用 Isaac Sim GUI 调资产；代码不依赖 | 四个无 Kit 查看器均无 prim 编辑 |
| 3 | 物理 | PhysX 5.4.2（GPU 刚体、缩减坐标关节、FEM 体/面可变形、粒子布料与 PBD 流体、SDF 碰撞、车辆、CCD）、张量 API、场景查询、表面吸盘、Newton 后端、Core 实验 API、克隆器 | PhysX 与张量 API C；Newton、克隆器、管理器 O | 刚体、关节、可变形体、接触、表面吸盘、克隆 | **Newton 六种求解器全覆盖刚体与关节**；可变形体仅 VBD；表面吸盘仅 PhysX；粒子流体无 |
| 4 | 渲染 | RTX 实时 2.0/交互式路径追踪/Minimal、MDL 材质、render product 与 AOV、分块相机、多 GPU 渲染、NuRec 高斯泼溅、视口与录屏、WebRTC 推流、RTSP | 全部 C（RTSP 为 O） | 15 项输出契约、分块相机、MDL 材质引用、PPISP | Newton Warp 渲染器 10/15 项，无运动矢量；OVRTX 14/15 但专有 |
| 5 | 传感器 | RTX 激光雷达/雷达/声学（20 余款真实雷达型号的 USD 配置）、RTX 相机（含结构光、立体视差、OpenCV 针孔/鱼眼畸变 schema、ISP 模型、30 余款真实相机型号）、物理传感器（接触、IMU、力矩、关节状态、射线） | RTX 传感器 SDK 与型号资产 C；物理传感器 O→C | 相机、分块相机、接触、IMU、关节力矩、帧变换、PVA、射线投射（网格、Bpearl、通用雷达图样） | **七类非相机传感器在三个后端均有实现**；缺真实雷达型号的物理模型、雷达与超声、相机畸变 |
| 6 | 合成数据 | Replicator 核心（随机化器、分布、触发器、编排器、写入器）、Isaac 写入器（Pose/DOPE/YCB）、行为随机化器、物理域随机化、情节录制器、抓取/移动性/遥操作 SDG、Replicator Agent/Object（生成式） | Replicator 核心与 Agent C；Isaac 侧 O→C | 8 个随机化器中的 2 个，0 个写入器 | 纹理与材质随机化抛 `NotImplementedError`；颜色随机化可用 |
| 7 | 资产导入导出 | URDF、MJCF、资产转换器（扁平化、变体、合并、后端互转）、URDF 导出、CAD（Catia/SolidWorks/AutoCAD/Creo、Onshape）、网格转换（OBJ/FBX/glTF）、高斯泼溅、E57/PTS 点云 | URDF/MJCF/转换器 O（基于 usd-exchange）；CAD 与网格 C | URDF、MJCF、网格转换 | URDF/MJCF 走 `isaacsim-asset-isolated`；**网格转换仍需 Kit** |
| 8 | 资产库 | 240 余个机器人、57 个传感器型号、12 个环境、SimReady 道具、NuRec 场景；HTTPS 资产根与中国镜像 | 代码 O；**资产 C** | 32 个机器人配置模块，472 处 Nucleus 路径引用 | 传输走 `omni.client`；资产本身受许可约束 |
| 9 | 机器人工具链 | ROS 2 桥（35 个 OmniGraph 节点、ros2_control、仿真控制服务）、cuMotion、运动生成实验 API、Pink IK、Lula/RMPflow、轮式与机械臂控制器、策略示例、Holoscan/UCX/ZMQ 桥 | ROS 2 桥与控制器 O；cuMotion、Lula C | 差分 IK、OSC、关节阻抗、Pink IK、RMPflow、cuRobo（Mimic） | **仓库内没有 ROS 2**；RMPflow 依赖闭源 Lula；cuRobo 依赖 CUDA |
| 10 | 遥操作与 XR | OpenXR/CloudXR 运行时、Isaac Capture（Vision Pro、Quest 3、Pico 4 Ultra、Manus、Haptikos）、Replicator 遥操作、`carb.input` 键鼠手柄 | XR 运行时与 CloudXR C；isaacteleop O | 键盘、手柄、SpaceMouse、Haply 无 Kit；头显全部走 Kit XR 加 CloudXR | 桌面设备可用；头显路径全部依赖闭源 |
| 11 | 学习与数据 | Isaac Lab、Mimic、Stage Recorder、情节录制器、Jupyter/VSCode 集成、基准服务 | Isaac Lab BSD-3、Mimic Apache；Stage Recorder C | Mimic 数据生成、基准记录器 | 基准记录器导入 `isaacsim.benchmark.services` |
| 12 | 部署运维 | 无头、容器（需接受 EULA）、云指南与 Isaac Automator、Kit 多 GPU 渲染、推流客户端 | 容器 C；Automator O | Docker、HPC、多 GPU（torchrun）、Ray | 无 Kit 路径只需普通 CUDA 容器 |
| 13 | 脚本与插件 | 内嵌 Python 3.12、扩展模板、`omni.kit.commands`、热重载 | 模板 O；命令与热重载 C | 无 | 不需要复刻 |

两点推论：

- **Isaac Lab 已经替 L1 做掉了一半以上。** 物理、非相机传感器、克隆、URDF/MJCF 导入、
  轻量可视化、训练与多卡都有无 Kit 实现。这是前两轮已确认的事实。
- **剩下的缺口全部在闭源部分**：RTX 渲染与传感器、Replicator、Kit UI、XR 运行时、资产库。
  这些恰恰是"生态"本身，没有 NVIDIA 的替代品，只能自建或采用第三方开源件。

### 1.3 仓库内残余的 Kit 与 NVIDIA 二进制依赖

按本次核验，非测试源码中导入 Kit 专有 API（`omni`、`carb`、`isaacsim`、`usdrt`）的文件共 61 个：
`isaaclab` 23、`isaaclab_physx` 22、`isaaclab_teleop` 8、`isaaclab_tasks` 4、
`isaaclab_visualizers` 3、`isaaclab_mimic` 1；`isaaclab_newton`、`isaaclab_ov`、`isaaclab_rl`、
`isaaclab_assets`、`isaaclab_experimental` 为零。核心包的 23 处按用途分组：

| 用途 | 文件 | L1 处置 |
| --- | --- | --- |
| Kit 启动与版本探测 | `app/app_launcher.py`（已弃用）、`__init__.py`、`utils/version.py`、`sim/utils/extensions.py` | 删除或改为可选 |
| Fabric/usdrt | `cloner/fabric_notices.py`、`sim/views/fabric_xform_selection.py`、`sim/utils/stage.py:391` | 随 PhysX 后端一起放弃 |
| 资产传输 | `utils/assets.py`（`omni.client`，含 S3 与中国镜像配置） | 改为 HTTP/S3 客户端 |
| 网格转换 | `sim/converters/mesh_converter.py`（`omni.kit.asset_converter`） | 换开源转换器 |
| Replicator | `envs/mdp/events.py:31`（`ReplicatorRNG`） | 自建随机数与事件项 |
| Kit UI | `envs/ui/*`、`ui/widgets/*`（`omni.ui` 曲线与图像控件）、`ui/xr_widgets/*` | 已有 Newton GL/viser/rerun 的曲线实现，删除 Kit 版 |
| 运动生成 | `controllers/utils.py`（Lula 路径、`UsdToUrdfConverter`） | RMPflow 需替换 |
| 基准 | `benchmark/benchmark_core.py`（`isaacsim.benchmark.services` 五个记录器） | 自建帧时记录器 |

包级别需要整体放弃的是 `isaaclab_physx`（29,094 行）与 `isaaclab_ov`（28,623 行），
前者绑定 Kit 内的 PhysX，后者绑定 `ovphysx`/`ovrtx`/`ovstage` 专有 wheel。
`isaaclab_teleop`（10,816 行）中头显路径绑定 Isaac Capture 与 CloudXR。
`isaaclab_mimic` 的运动规划器绑定 cuRobo（L1 下可保留，L2 下替换）。

`pyproject.toml` 基础依赖中仍有三个 NVIDIA 包：`omniverseclient==2.74.0`、
`usd-exchange==3.0.1`（提供 `pxr`）、`isaacsim-asset-isolated==6.2.0.0`（URDF/MJCF 导入）。
前者 L1 必须替换，后两者 L1 可保留、L2 替换。

## 二、可以依靠的非 NVIDIA 地基：OpenUSD 生态与 Blender

这一节回答"不用 Omniverse，OpenUSD 这边有什么可用"。结论是：**场景格式与物理 schema 这一层
已经足够成熟，渲染这一层只有零件没有成品，材质这一层有标准但没有迁移工具。**

### 2.1 OpenUSD 核心：UsdPhysics 已经不需要 NVIDIA 的工具链

- 最新版本 v26.11.1（2026-10-21），季度节奏。`usd-core` 官方 wheel 覆盖 CPython 3.11 至 3.14、
  Linux x86-64/ARM64、macOS、Windows，但**不含 Hydra、Storm、usdview、usdrecord**。
  25.08 起为单体构建，校验器库已包含在内。
- **UsdPhysics 解析工具自 25.02 进入核心**（`parseUtils.h`：`LoadUsdPhysicsFromRange` 与描述符
  体系），25.11 实现了嵌套刚体与运动学关节提案，26.03 至 26.11 持续修复并加校验器。
  这意味着一个自建物理后端可以**直接用核心 API 把 USD 舞台解析成刚体、碰撞体、关节、
  驱动的描述符**，不需要 Kit，也不需要 NVIDIA 的 `usd-exchange`。
  Isaac Lab 当前走的是自己的 `sim/schemas/schemas.py` 无代码写入路线，二者兼容。
- 核心里**没有**的：mimic 关节（`PhysxMimicJointAPI` 为 NVIDIA 专有）、腱、执行器、可变形体。
  REP 158（2026-03-03 接受，Robotec/NVIDIA/Ekumen）确认这些继续走 `mjc:`/`newton:` 厂商前缀。
  AOUSD 物理工作组 2024-12 成立，Core Spec 1.0 已于 2026-03 批准，
  **物理领域规范截至 2026-10 仍未发布**；可变形体提案 PR #111（2026-06）与关节摩擦
  PR #119（2026-10）均在草案。
- 语义标注 `UsdSemantics.SemanticsLabelsAPI` 自 24.11 进入核心，Isaac Lab 已在用。
  UsdValidation 框架 24.08 至 26.11 持续建设，`usdchecker` 默认走它。
- Storm 的 Vulkan 后端 `PXR_ENABLE_VULKAN_SUPPORT` **仍默认关闭、标记实验性**；
  Linux GL 构建链接 X11/GLX，EGL 无头 issue #2011（2022 年由 Blender 的 Brecht 提出）至今开放。
  Storm **没有硬件光追，没有运动矢量 AOV**。
- OpenExec 25.08 预览、26.11 仍限于绑定与表达式求值，不是仿真运行时。
- 资产解析：Ar 2.0 支持 URI 解析器插件，但**没有维护中的 S3/MinIO/git-LFS 解析器**
  （`arHttp` 为 2024 年概念验证，`usd-s3-resolver` 停在 2018）。现实做法是 `usd-core`
  加本地路径，或自写一个走预签名 HTTPS 的小型 URI 解析器。这与替换 `omni.client` 的工作量一致。
- NVIDIA 的 `usd-exchange` 3.0.1 为 Apache-2.0，`usdex_core` 只依赖 OpenUSD、TBB、MaterialX，
  可对自有 OpenUSD 构建；PyPI wheel 自带 OpenUSD 并与 `usd-core` 冲突。
  其继任校验器 `usd-validation-nvidia` 1.23.0（2026-10-27，Apache-2.0，纯 Python，跑在 `usd-core` 上）。

### 2.2 Hydra 渲染 delegate：没有一个同时满足"超写实、GPU 常驻、带分割"

对 NVIDIA 之外的 Hydra delegate 逐一核对后的对照：

| Delegate | 许可 | 设备 | 类型 | ID 类 AOV | 运动矢量 | 输出留在 GPU | 2026 维护 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HdStorm | Apache-2.0 | GL 默认，Vulkan 实验 | 光栅 | primId、instanceId、elementId | 无 | 是 | 是 |
| HdEmbree | Apache-2.0 | CPU | 参考路径追踪 | primId、instanceId、elementId、normal | 无 | 否 | 随 OpenUSD |
| hdCycles | Apache-2.0 | CPU/CUDA/OptiX/HIP/oneAPI/Metal | 路径追踪 | primId、instanceId、elementId | 无 | 否 | 是（2026-10-27 有提交），官方标"实验性"，无 MaterialX |
| hdMoonray | Apache-2.0 | CPU 加 XPU（仅 OptiX） | 路径追踪 | 无整数 ID，有 cryptomatte | 有（需 3D 运动模糊） | 否 | 是（2026-07 版本，MaterialX 进行中） |
| hdOSPRay | Apache-2.0 | CPU | 路径追踪 | 有 | 无 | 否 | **2025-08-25 归档** |
| hdRPR（AMD） | Apache-2.0 | OpenCL | 路径追踪 | 未文档化 | — | 否 | 休眠，最后版本 2024-10 |
| HdArnold / Karma / RenderMan XPU | 商业 | GPU 均仅 CUDA/OptiX | 路径追踪 | 有，cryptomatte | 有 | 否 | 是 |
| Falcor（NVIDIA 研究，BSD-3） | BSD-3 | D3D12 加 Vulkan | 硬件光追渲染图 | 任意 | 任意 | 是 | 9.0（2025-09），USD 导入实验性 |
| Autodesk Aurora | Apache-2.0 | DXR 加 Vulkan | 实时路径追踪 | 未文档化 | — | 可能 | 26.08 预览（2026-09） |
| Diligent Hydrogent | Apache-2.0 | D3D/Vulkan/GL/WebGPU | 光栅 PBR | 未文档化 | — | 是 | 是 |

四条横切结论：

1. **语义分割在 Replicator 之外的任何 delegate 里都不存在。** 能拿到的只有整数 primId/instanceId
   或 cryptomatte，语义掩码必须由应用层按 prim 路径查 `SemanticsLabelsAPI` 自己合成。
   这项工作 Isaac Lab 的 Newton Warp 渲染器已经做过一遍，可以复用。
2. **只有光栅类（Storm、Hydrogent、Falcor、Aurora）把输出留在 GPU 上，所有路径追踪器都返回
   主机缓冲区。** 这与 Isaac Lab 的分块相机对 GPU 常驻张量的要求直接冲突。
3. **没有任何 delegate 支持分块多相机渲染**，Hydra 渲染任务按相机划分。
4. **所有带 Hydra 的 GPU 路径追踪器都绑定 CUDA/OptiX**（MoonRay XPU、Arnold GPU、Karma XPU、
   RenderMan XPU、Cycles OptiX）。非 NVIDIA GPU 上的 Hydra 路径追踪实际只剩 Cycles 的
   HIP/oneAPI/Metal 与休眠的 hdRPR。这一点对 L3 很重要。

### 2.3 MaterialX 与 OpenPBR：有标准，没有从 MDL 过来的桥

- OpenUSD 核心内置 MaterialX（26.08 为 1.39.5），`UsdMtlx` 在核心，Storm 能编译 MaterialX 图，
  hdPrman XPU 26.11 起原生渲染 OpenPBR。Adobe 的文件格式插件 2026.07 起默认导出 OpenPBR。
- hdCycles 与 HdEmbree **完全不支持 MaterialX**；hdMoonray 的支持为进行中。
- **MDL 到 MaterialX 没有官方导出器。** NVIDIA MDL-SDK（BSD-3）的 Distiller 只做 MDL 到简化 MDL
  加纹理烘焙，不写 MTLX；MaterialX 仓库历史上的 MDL 代码生成器是反方向。
  现实做法：用普通 `UsdShade` 读 OmniPBR 的输入（反照率/ORM/法线/自发光纹理与标量粗糙度、
  金属度），重新写成 `open_pbr_surface` 或 `UsdPreviewSurface` 网络。
  Isaac Lab 仓库内**没有任何 `.mdl` 文件**，全部引用 Nucleus 上的 `OmniPBR.mdl`、`OmniGlass.mdl`
  与 `/Materials/...` 库，所以这项工作主要落在资产重建上，代码侧只需把 `MdlFileCfg`/
  `PbrMdlCfg`/`GlassMdlCfg` 三个配置类换成 MaterialX 等价物。

### 2.4 Omniverse 之外的物理创作与转换

- **MuJoCo 自带 USD 导入导出**（实验性）：主要使用标准 UsdPhysics，加 `mjcPhysics` 扩展
  （MjcSceneAPI、MjcJointAPI、MjcActuator、MjcTendon、MjcKeyframe 等），刻意不用 `UsdPhysicsDriveAPI`；
  另有 MJCF 的 SDF 文件格式插件，让 MJCF 直接作为 USD 层参与合成。这是 L2 下替代
  `mujoco-usd-converter` 的现成路径，且 Isaac Lab 的 Newton 后端本来就以 MJWarp 为默认。
- Newton 自身导入 USD/URDF/MJCF 到 `ModelBuilder`。gz-usd（SDF 与 USD 互转，2026-10 仍有提交）。
  Drake 在 2025-08 **删除了未完成的 USD 解析器**，原因写明是"被 NVIDIA 放弃的未完成特性"。
- DCC 侧：Unreal 与 Unity 的 USD 导入器都不读 UsdPhysics，Unity 的 USD 包 2024-11 已弃用；
  Maya/3ds Max 的 USD 插件是层编辑器，会保留 UsdPhysics 但没有物理 UI（未复核）。
  **除 Omniverse 与 Windows 版 SimReady Blender 插件外，没有任何带 GUI 的 UsdPhysics 创作工具。**
- 校验：`usdchecker`（含 `usdPhysicsValidators`）、`usd-validation-nvidia`、SimReady Foundation
  规则集（2026.07.1，许可未标明）。

### 2.5 交互式 USD 编辑器

| 编辑器 | 许可与状态 | 能否非破坏地保留未知 applied schema 与变体 |
| --- | --- | --- |
| usdview | Apache-2.0，随 OpenUSD | 是，但编辑只进会话层，无属性编辑器与层栈编辑 UI |
| usdtweak | Apache-2.0，prealpha 2026.07.18，2026-10-19 有提交 | **是**，直接编辑 Sdf 层，支持 prim/属性/变体创作与 schema 应用 |
| USD Manager（DreamWorks） | Apache-2.0，2026-09 有提交 | 是（文本编辑器带合成导航） |
| Houdini Solaris / Maya / 3ds Max | 商业宿主 | 是（未复核） |
| Web：Autodesk WASM 分支、needle usd-viewer、OpenUSD 26.03 官方 wasm | 开源 | 仅查看 |

usdtweak 是唯一同时满足"开源、跨平台、直接编辑层、支持变体与 schema 应用"的桌面工具，
但处于 prealpha。它比在 Blender 里复刻 Property/Layer 面板更接近 Isaac Sim 的三个面板。

### 2.6 Blender 的新增事实

在第二轮结论（无 UsdPhysics 导出、层与变体被压平、`USDHook` 是干净注入点、Cycles 可无头、
EEVEE 不可）之上，本轮补充：

- `bpy` 5.2.3 wheel（2026-10-22）**只支持 Python 3.13**，Linux 需 glibc 2.28；5.x 没有 3.12 wheel。
  Isaac Lab 当前环境是 Python 3.12 与 torch 2.11，**bpy 进程内嵌入与现有环境不兼容**，
  这从工程上坐实了"Blender 必须是独立进程"。
- Blender 5.3（Beta 至 2026-11-04）发布说明：**Vulkan 后端在 Windows 与 Linux 上成为默认**；
  6.0 考虑移除 OpenGL。但没有任何来源说明 Vulkan 使 EEVEE 可以无显示服务器渲染，
  应继续按不支持处理，直到实测。
- hdCycles 在 `main` 持续修补（2026-10-27），但仍标实验性；AMD 的 Blender Hydra 插件停在 2023-11；
  Blender 自带的 Hydra Storm 引擎是唯一的 MaterialX 节点导出路径。
- 机器人插件：Phobos（BSD-3，2026-07 有提交，URDF/SDF/SMURF）；
  **仍然没有任何从 Blender 导出 UsdPhysics 的插件**。

## 三、O3DE 在全栈框架下的评估

前两轮分别从"整体底座"与"编辑与渲染"两个角度否定了 O3DE。本轮补齐其余九个方面，
结论没有变化，但把"为什么不能"落到了具体代码与数字上。基线仍是 26.05.0
（2026-05-27），26.10.0 原定 2026-10-28 发布，核验日只有 `stabilization/26100` 分支与草稿说明。

### 3.1 物理：没有第二后端，没有 GPU，关节接口不可移植

- o3de 与 o3de-extras 中**没有任何 Jolt、Bullet 或 MuJoCo 的一方 gem**。唯一的替代是同一
  谱系的两个社区 Jolt gem（`iamjbq/JoltPhysics` 与其 fork `jorgetbruno/JoltPhysics`，
  后者声称支持 26.05，最后提交 2026-09-18），均不提缩减坐标关节。
- `AzPhysics::SystemInterface` 与 `SceneInterface` 是纯虚接口，刚体层面确实后端中立；
  但**关节（articulation）的 EBus 全部在 `PhysX::` 命名空间下**
  （`PhysX::ArticulationJointRequestBus`），第二后端无法复用。没有任何关于后端替换的 RFC。
- PhysX gem 仍为 5.1.1。唯一的升级努力是 issue #19595（2026-03-02，升到 5.6.1，开放、
  无人认领、无 PR）。**不存在任何 GPU PhysX 的 issue 或 PR**，`PhysXScene.cpp`
  从未设置 `eENABLE_GPU_DYNAMICS`。
- 关节支持仅 `Fix`、`Hinge`、`Prismatic` 三种，**没有球关节**；驱动参数齐全，有力传感器；
  但惯量自动计算默认关闭，issue #20016（2026-08-17）记录了由此导致的"与 Isaac Sim 不一致"，
  26.10 草稿称已修。
- 确定性：无文档与 issue 讨论，唯一开关 `m_enableEnhancedDeterminism` 默认关闭且不在 UI 中。

### 3.2 步进与多场景：有固定步长，无批量状态

`m_autoManageSimulationUpdate` 可关闭以手动驱动，`t_simulationTickScale` 可加速或停止；
最多 64 个场景，但所有场景同一 dt、组件只认默认场景，**没有把 N 个场景当并行 RL
环境的先例**。位姿同步按刚体逐个进行，**不存在任何批量状态读写接口**。
SimulationInterfaces gem（v2.2.0）实现了 ROS 2 仿真接口标准的 spawn/step/reset，
但 step 只在暂停态可用。O3DE 与 Robotec **都没有发布过任何无头吞吐数字**
（实体数、步/秒、实时因子），唯一的第三方对比是一篇付费墙后的 RA-L 2026 论文。

### 3.3 传感器渲染：一切经 CPU，无 ID 通道，无畸变，无 CUDA 互操作

- ROS 2 相机为每个通道建一条 `RenderPipeline`，经 `CapturePassAttachmentWithCallback` 回读。
  `AttachmentReadback` 基于 fence 异步、三帧环形缓冲、一次一个附件，**延迟 1 至 3 帧且无文档**；
  忙时直接失败而非排队；**所有路径终点都是 CPU 缓冲区**。现场报告 issue #1080（2026-08）：
  时间戳抖动四分位距 33 ms，静默丢帧。
- "每多一个相机帧时间大约翻倍"，RTX 4080 上 1080p 请求 60 Hz 实得约 30 Hz（issue #804）。
- 多 GPU 在 25.05 成熟、26.05 支持交替帧渲染，但只能按 pass 级 `deviceIndex` 分配。
- **GPU 到 GPU 的路径不存在**。最接近的是华为的 PR #18450（2024-12 合入）：
  Vulkan/DX12 外部内存与信号量导出钩子，"面向非渲染 API"，**通篇未提 CUDA**。
  对 Atom RHI 的检索（2026-10-28）：无 `cuda`、无 `VK_KHR_external_memory_fd`、无 DLPack。
  没有任何零拷贝到 PyTorch 的先例。
- 相机模型仅针孔，配置结构体没有畸变字段，`camera_info` 的 `d` 为空。
- 可导出的 pass：Forward 输出 Diffuse/Specular/Albedo/F0/Normal 加深度，另有相机与网格运动矢量、
  线性深度、剪影。**没有物体或实例 ID 通道，没有语义通道**；相机分割请求 #955（2025-08）
  开放无 PR。分割只在 RGL 激光雷达（CUDA 加 OptiX）里有。
- 材质：Material Canvas 仍在重构（PR #20058，2026 年多个崩溃 bug）；**MaterialX issue #18873
  无人回应，RFC #51 明确拒绝 MaterialX；OpenPBR 零命中**。着色器链为 AZSL 到 HLSL 到
  DXC/SPIRV-Cross，不接受 Slang 或原生 HLSL。

### 3.4 编辑器与创作：prefab 是且将是原生格式

- JSON prefab 加 JSON patch 覆盖，运行时为 `.spawnable`。**没有任何 RFC 提议以 USD 为原生
  场景格式**；RFC #92 把 prefab 保留为原生并引用 Remedy 原生 USD 的稳定性问题作为理由；
  sig-content 无 USD RFC；"SceneAPI 重构想法"讨论 #19463（2026-01）零回复。
  **O3DF 不是 AOUSD 成员**（华为与字节跳动是）。
- 编辑器 Python 为进程内 3.10，`main` 上仍是 PySide2 5.15.2，`development` 已切到
  PySide6 6.11.2；Python 可添加组件但**不能定义新组件类型或 EBus 处理器**。
- 无头编辑器：`-BatchMode -autotest_mode -rhi=null -runpythontest`；Linux 上仍需 X11。
- 碰撞体：场景设置里有 V-HACD 凸分解（最多 1024 个凸包）与盒/胶囊/球自动拟合；无 CoACD。

### 3.5 机器人生态：全部押在 Robotec，而 Robotec 已多引擎

ROS 2 支持 Humble、Jazzy、26.05 加 Lyrical，仅 Linux；RFC #96 把 ROS2 gem 拆成四个；
Nav2 与 MoveIt2 只在 ROSCon 2023 演示工程里出现；**没有人形机器人工程**。
SIG Simulation 最后一次会议记录是 2022-06-16。Robotec 到 2026-10 仍在提交 o3de-extras，
但其 RoSi 平台**现在同时列出 Isaac Sim、Genesis 与 O3DE**，RAI 仓库已是引擎无关。
中国方面：Gitee 有镜像，信通院与广东数字产业研究院是 O3DF 准成员，
**未找到任何中国机器人企业或高校的 O3DE 使用案例**，无 OpenHarmony 移植。

### 3.6 网络、推流、XR、嵌入

- **没有一方 WebRTC 或像素推流 gem**，只有 Robotec 的 ViewportStreamer
  以 30 Hz 发布 `sensor_msgs/Image`。
- OpenXRVk gem v1.3.0：OpenXR SDK 1.1.41，Linux 2025-09 重新启用，有 Quest 控制器模型，
  **无手部追踪、无透视**，PCVR 无文档。
- **没有 gRPC、ZeroMQ 或共享内存 gem**；Remote Tools 是 Lua 调试链路且发布版编译排除；
  **ROS 2 加 SimulationInterfaces 是唯一的外部控制面**。

### 3.7 构建与部署

Ubuntu 24.04 为主，ARMv8 Linux "实验性"，无 Jetson 文档；源码构建需 100 GB 以上磁盘、
每线程 2 GB 内存；.deb 安装约 17.9 GB；官方 Docker 覆盖 jammy/noble 与 ROS humble/jazzy/lyrical；
`GameLauncher -NullRenderer -rhi=null` 可无 GPU 运行，但相机组件需要真实 RHI，客户端仍需 X11。

### 3.8 项目健康度：提交量较 2022 年下降约 95%

| 年份 | 全部提交 | 合并提交 | 独立作者 |
| --- | --- | --- | --- |
| 2022 | 10,010 | 3,108 | 202 |
| 2023 | 4,285 | 1,370 | 145 |
| 2024 | 1,124 | 576 | 69 |
| 2025 | 536 | 423 | 64 |
| 2026 至今 | 445 | 373 | 57 |

（数据来自子代理对 `development` 分支的本地克隆统计，发布前应复核。）

Amazon 域名作者从 2021 年的 134 人降到 2025 至 2026 年的 3 人；华为提交从 2024 年的 300 次
降到 2026 年的 10 次；Robotec 每年约 20 次。o3df.org 成员页**Premier 一栏显示"暂无成员数据"**，
2022 年的 Adobe、AWS、Epic、华为、Intel、微软等名单已不再出现；General 成员只剩 Red Hat、
Robotec、SideFX。2025 至 2026 年没有 Amazon、华为、腾讯的生产使用公告。

### 3.9 结论

O3DE 在本框架下能提供的只有：Apache/MIT 许可、跨厂商 Vulkan 光栅渲染、一个可用 Python
自动化的编辑器外壳、V-HACD 凸分解、以及 ROS 2 组件集。它不能提供的是本次目标的全部核心：
USD 舞台、GPU 或批量物理、GPU 常驻的传感器输出、ID 与语义通道、MaterialX、
进程内控制面、路径追踪。而且这些缺口没有一项有人在做。

**L1 下不建议引入 O3DE。** 若仍需一个实时 Vulkan 编辑器外壳，其工程量不低于在
Isaac Lab 既有的 Newton GL 查看器上扩展，且会引入一个维护量在萎缩的 C++ 代码库。

## 四、逐能力差距与可用的替代件

本节只覆盖第二、三节没有讨论的能力组：物理、传感器、合成数据、资产导入与资产库、
机器人工具链、遥操作与 XR、推流与可视化、部署。每项先说 L1 下现状，再说可用件，最后说必须自建的部分。

### 4.1 物理：L1 下基本完成，L2 下是另一个问题

L1 保留 Warp，Newton 后端（MJWarp、Kamino、Featherstone、XPBD、VBD、MPM 六种求解器）就是答案。
Isaac Lab 的 47 个注册任务中 44 个训练环境至少有一个无 Kit 物理预设。对照 Isaac Sim 的
PhysX 功能面，L1 下真正丢失的是：

| PhysX 能力 | Newton 现状 | 处置 |
| --- | --- | --- |
| FEM 体可变形与面可变形 | 仅 VBD 求解器，参数体系不同（Lamé 参数而非杨氏模量） | 已有，需为现有任务重新调参 |
| 粒子布料、PBD 粒子与流体 | MPM 求解器覆盖颗粒与部分流体；无 PBD 流体 | 按需评估 MPM，流体类任务可能放弃 |
| 表面吸盘（surface gripper） | `isaaclab_physx` 独有 | 需在 Newton 上以约束或固定关节重做 |
| SDF 碰撞、车辆、CCD | 部分（MJWarp 用凸包，`maxhullvert` 限制） | 非凸网格须上游凸分解（CoACD 已是依赖） |
| GPU 确定性 | MJWarp 因原子操作**不确定**，确定性 PR #1300 关闭未合入、#1422 开放（实测代价约 40%） | 复现实验需接受或等待上游 |

L1 下有价值的第三方引擎只有两个，均作为旁路而非替代：

- **MuJoCo 3.17**：Apache-2.0，自带实验性 USD 导入导出（`mjcPhysics` 扩展 schema），
  3.17 新增 `mujoco.batch` CPU 线程池批处理。Isaac Lab 锁定 `~=3.15.0`，3.16/3.17 有 MjSpec
  破坏性变更。它是 L2 下替代 `mujoco-usd-converter` 的路径，也是 CPU 兜底。
- **Genesis 1.4.5**：Apache-2.0，唯一多厂商 GPU 引擎（经自研编译器 Quadrants 支持 CUDA、ROCm、
  Metal、Vulkan、CPU），读 USD（含 UsdPhysics 关节与驱动）、URDF、MJCF，1.4.4 修复了 GPU
  非确定性。其"4300 万 FPS"宣传数字被第三方复测为约 29 万且比 ManiSkill 慢 3 至 10 倍，
  但多厂商支持已是实际发布的能力而非口号。它是 L3 的候选，不是 L1 的。

需要纠正前一轮文档的一处说法：**MotrixSim 的引擎是闭源二进制**（`motrixsim-core` wheel 是一个
237 MB 的编译模块，仓库只有文档），Apache-2.0 只是元数据；本轮也**未能找到其鲲鹏认证的一手证据**。
前一轮"Rust 实现、鲲鹏认证"的表述应降级为"厂商口径、未验证"。

PhysX 5 SDK 单独使用（SAPIEN/ManiSkill3 的做法：静态链接 SDK、运行时侧载 `libPhysXGpu_64.so`）
在法律上可行（`physx/LICENSE.md` 自 5.11 起 Apache-2.0，根目录 BSD-3），
但它与 Warp 属于同一类——NVIDIA 代码、仅 CUDA。L1 下没有必要为它放弃 Newton。

### 4.2 传感器：非相机类齐全，RTX 类要自建物理模型

Isaac Lab 的七类非相机传感器（接触、IMU、关节力矩、帧变换、PVA、射线投射、射线相机）
在 Newton、OvPhysX、PhysX 三个后端上都有实现，射线投射在 GPU 上直接查询 Newton 场景 BVH。
对照 Isaac Sim 的 RTX 传感器家族，缺的是：

| Isaac Sim 能力 | 开源现状 | 自建内容 |
| --- | --- | --- |
| RTX 激光雷达：20 余款真实型号配置、多回波、反射率强度、光束发散、天气 | 无 OptiX 的 GPU 常驻射线投射只有 Warp BVH 查询；RGL（Apache-2.0）绑定 CUDA 加 OptiX；HELIOS++ 最物理但 CPU 且 GPL；**没有任何开源雷达仿真同时建模光束发散、多回波与天气** | 在 Warp 射线投射上加强度、多回波、噪声与发散模型；按公开规格重建型号库（型号 USD 资产受 NVIDIA 许可约束） |
| RTX 雷达 | **没有成熟开源方案**。RadarSimPy 仓库 GPL-3 但分发的预编译模块收费；CARLA 雷达只是几何点目标 | 几何点目标级雷达可在射线投射上实现；物理级放弃 |
| RTX 声学/超声 | Webots 的多射线锥模型可参考 | 在射线投射上实现锥形多射线，工作量小 |
| 相机畸变 schema（OpenCV 针孔、OpenCV 鱼眼） | OpenCV 4.15 提供 Brown-Conrady、Kannala-Brandt、Mei 模型；光栅器靠超采样加重映射，光线追踪器靠逐像素射线表 | Warp 渲染器可直接接受任意射线方向表，是最自然的实现点 |
| 相机 ISP 模型（`omni.sensors.nv.camera`） | 仓库内 `isaaclab_ppisp` 已是渲染器无关的 HDR 到 LDR 管线 | 已有 |
| 30 余款真实相机型号资产 | 规格公开，资产受许可约束 | 重建配置 |
| IMU 噪声 | Isaac Lab 的噪声工具只有高斯、均匀、加性偏置，**没有随机游走模型**；SDFormat 的 `<noise>`（偏置、动态偏置与相关时间）与 `gnss-ins-sim` 是参考 | 补随机游走与偏置不稳定性模型 |

### 4.3 合成数据与域随机化：Replicator 的使用面窄，替换容易

Isaac Lab 只用了 Replicator 8 个随机化器中的 2 个、0 个写入器。渲染器无关的
`VisualMaterialBatch` 契约三个后端都实现了，缺的只是驱动它做纹理随机化的事件项。

需要自建的：纹理与材质随机化事件项；语义与实例分割的合成（按 primId 查 `SemanticsLabelsAPI`，
Newton Warp 渲染器已实现）；2D/3D 包围盒、关键点、6D 位姿写入器。
可直接用的开源件：`bop_toolkit`（MIT，BOP 格式）、`pycocotools`、FiftyOne、Datumaro。
**不要基于 BlenderProc 构建**（GPL-3 传染）；Kubric（Apache-2.0）自 2021 年无版本；
Infinigen（BSD-3，2026-10 仍活跃）可作为环境生成器，但整场景只导出无关节的 `usdc`；
Unity Perception 已停止、Habitat-Sim 于 2026-10-26 归档。
物理参数随机化 Isaac Lab 的事件管理器本来就是后端无关的，不需要动。

### 4.4 资产导入：L1 下几乎没有缺口，L2 下换一套

| 环节 | L1 方案 | L2 增量 |
| --- | --- | --- |
| URDF/MJCF 到 USD | `urdf-usd-converter 0.3.4`、`mujoco-usd-converter 0.8.0`（Apache-2.0，依赖 `usd-exchange` 与 `newton-usd-schemas`，无专有组件） | 改用 Newton `ModelBuilder.add_urdf/add_mjcf` 直读，或 MuJoCo 自带 USD 导出 |
| 网格（OBJ/STL/FBX/glTF）到 USD | 仓库内 `mesh_converter.py` 仍走 Kit。替换：trimesh/Open3D/Assimp 读入，`usd-core` 写 `UsdGeom.Mesh`；或 Blender 无头进程 | 无 |
| CAD（STEP/IGES） | OCCT 8.0 / `cadquery-ocp` / build123d / FreeCAD 转网格 | 无 |
| Onshape | `onshape-to-robot 1.8.3`（MIT）出 URDF/MJCF | 无 |
| 凸分解与简化 | CoACD（MIT，已是依赖）、`fast-simplification`、xatlas | 无 |
| 资产校验 | `usdchecker`、`usd-validation-nvidia`（Apache-2.0）、SimReady 规则集 | 无 |
| `pxr` 运行时 | `usd-exchange 3.0.1` | 换 `usd-core 26.11`，并解决与 `usd-optimize` 的共存 |

MuJoCo 与 MJWarp 只对网格的凸包做碰撞，非凸几何必须在上游分解；Newton 的 builder 已暴露
`coacd`/`vhacd`/`convex_hull` 重网格选项并带缓存。

### 4.5 资产库：最大的非技术缺口

这是本轮最需要强调的发现。Isaac Lab 的 `isaaclab_assets` 有 32 个机器人配置模块，源码中
`ISAAC_NUCLEUS_DIR`、`ISAACLAB_NUCLEUS_DIR`、`NVIDIA_NUCLEUS_DIR` 共 472 处引用，全部指向
NVIDIA 托管的 USD 资产。这些资产按 Isaac Sim 6.2 的"附加软件与材料许可"：

- 只能"在 NVIDIA GPU 系统上"与 Isaac Sim/Isaac Lab 一起使用；
- 不得修改、不得制作衍生作品、不得再分发；
- 许可文本没有提及与第三方仿真器一起使用的情形，属于未明确授权。

因此 L1 下必须重建资产库。可用来源：

| 类别 | 来源 | 许可 |
| --- | --- | --- |
| 机器人 | MuJoCo Menagerie（逐机器人许可，多为 BSD/Apache/MIT）、厂商官方 URDF（Unitree、Franka、UR、Kinova 等）、Onshape 公开文档 | 逐个核对 |
| 环境与道具 | Infinigen（BSD-3）、Poly Haven（CC0）、ambientCG（CC0）、Objaverse（混合许可，需过滤） | CC0 优先 |
| 材质 | ambientCG、Poly Haven 的 PBR 纹理集，重写为 MaterialX `open_pbr_surface` 或 `UsdPreviewSurface` | CC0 |
| 传感器型号 | 厂商公开规格表重建配置；外观网格自建 | 自有 |

重建的技术路径就是2.6 节 与前一轮文档的 Blender 创作管线加 USD 物理导出插件。
工作量主要是内容制作与许可审核，不是代码。

### 4.6 ROS 2：从零开始，但零件齐全

Isaac Lab 仓库与 Newton 都**没有任何 ROS 2 代码**。社区唯一的先例 `newton_ros`
（93 星，2026-04 最后推送，锁定 Newton 1.0.0）**没有许可证文件**，不能直接用。

可行设计：进程内 `rclpy` 发布 `/clock`、TF、关节状态、IMU；`topic_based_ros2_control`
（PickNik，BSD-3）接 ros2_control；`simulation_interfaces 2.1`（Apache-2.0）提供
spawn/step/reset 服务；图像与点云走 `rclcpp` 旁车进程或 ROS 2 Lyrical 新增的
`rosidl::Buffer` 零拷贝（仅 Fast DDS），因为 `rclpy` 发布 10 MB 点云约 92 ms 而 `rclcpp` 约 2.8 ms。
ROS 2 Lyrical Luth（2026-05-22，LTS 至 2031）默认 RMW 仍是 Fast DDS，`rmw_zenoh` 为一级支持。

### 4.7 运动生成与控制：一项修正与一项替换

- **cuRobo 自 v0.8.0（2026-04-18）起为 Apache-2.0**，此前为 NVIDIA 非商业许可。
  它仍然仅 CUDA，L1 下 `isaaclab_mimic` 的 cuRobo 规划器可以保留。
- **RMPflow 依赖闭源 Lula**（只随 Isaac Sim 发布，无公开源码），L1 下必须替换。
  开源 RMP 实现只有无人维护的 `rmp2`（MIT）与 GPL-3 的 `fabrics`，都不理想；
  实际替代是 cuRobo 的 RMPflow 控制器（L1）或 Pink/mink 差分 IK（已有）。
- `controllers/utils.py` 的 `UsdToUrdfConverter` 来自 Isaac Sim，需基于 OpenUSD 25.02 起的
  UsdPhysics 解析工具自写一个 USD 到 URDF 的导出器。
- CPU 侧可补 MoveIt 2 2.15、OMPL 2.0（`pip install ompl`，VAMP 后端可达微秒级规划）、
  Pinocchio/Crocoddyl、Placo、Ruckig、TOPP-RA；厂商中立的 GPU 规划只有 PyRoki（JAX，无 PyPI 版本）。

### 4.8 遥操作与 XR：桌面设备无事，头显换栈

键盘、手柄、SpaceMouse、Haply 四类设备已无 Kit。头显路径（Vision Pro、Quest 3、Pico 4 Ultra、
Manus、Haptikos）全部经 Isaac Capture 加 CloudXR。替代栈有两条：

- **原生**：Monado（开源 OpenXR 运行时，支持 `XR_EXT_hand_tracking`）加 WiVRn（GPL-3，v26.9，
  串流到 Quest 1/2/3/3S/Pro、Pico 4、Galaxy XR）加 `pyopenxr 1.1.6501`（唯一维护中的 Python
  OpenXR 绑定）。Isaac Lab 的 `OpenXRDevice` 重定向器可以改由 pyopenxr 喂数据。
- **浏览器 WebXR**：`vuer 0.1.6`（MIT）Python 服务，Open-TeleVision 与 Unitree `xr_teleoperate`
  （2026-07，面向 Vision Pro、Pico 4 Ultra、Quest 3）都走这条路；Vision Pro 另有
  `avp_stream`（ARKit 27 关节手部数据经 WebRTC）。

两点损失：Quest Link 仅 Windows；`viser` 的 WebXR 支持两个 PR 至今是草稿，`rerun` 没有 VR 模式，
所以现有查看器都不能直接当头显端。

### 4.9 推流、可视化与检视

- Kit 视口与 `omni.ui` 控件：Newton GL、viser、rerun 已覆盖标记、实时曲线、相机面板、录像，
  Kit 版可以直接删除。
- Kit WebRTC 推流：GStreamer 1.28 `webrtcbin` 加 `nvh264enc`/`vah264enc`，或 `webrtcsink`
  （MPL-2.0，一对多、WHIP/WHEP）；桌面级用 Sunshine/Moonlight（GPL-3）或 Selkies（MPL-2.0）
  抓窗口；`aiortc` 只有软件编码。Isaac Lab 的 `starlette` 依赖只服务 Kit 推流，可去掉。
- 交互式 prim 检视：2.5 节 的 usdtweak 是最接近 Stage/Property/Layer 三面板的开源件；
  Isaac Sim 的机器人工具（增益调参、装配器、抓取编辑器、位姿器）本身是 Apache-2.0 Python，
  但建在 `omni.ui` 上，逻辑可移植、界面要重写。

### 4.10 部署

无 Kit 路径只需 `nvidia/cuda` 基础镜像加 `libvulkan1`/`libegl1` 与 ICD 文件（ManiSkill 的
Dockerfile 是现成模板），`NVIDIA_DRIVER_CAPABILITIES` 需显式列出 `graphics`/`video`。
`nvidia/cudagl` 自 2022 停更、`nvidia/opengl` 过期、GitLab vulkan 镜像归档。
MuJoCo、MJWarp、Newton 都没有官方镜像。多卡多节点走 `torchrun` 与 Ray，本就无 Kit 依赖，
只有 Apptainer 与 Ray 的镜像基于 `isaac-lab-base`，需要换底。基准记录器
`isaacsim.benchmark.services` 需用 `torch.cuda.Event` 与 pynvml 自写。

### 4.11 先例：没有人做过"完整替代"，但每一块都有人做过

| 项目 | 物理 | 渲染 | USD | 脱离 Isaac Sim |
| --- | --- | --- | --- | --- |
| SAPIEN 3 / ManiSkill3 | PhysX 5 GPU（侧载） | 自研 svulkan2，Vulkan 光栅加光追着色包，NVIDIA/AMD/Intel | 不用 | 是 |
| MuJoCo Playground | MJX 加 MJWarp | MJWarp 批量光线追踪（CUDA，RGB 加深度，无分割） | 不用 | 是 |
| RoboCasa365 / robosuite / LIBERO | MuJoCo CPU | MuJoCo GL/EGL | 不用 | 是 |
| Newton | 六种求解器 | GL/Rerun/Viser/USD 查看器，RTX 查看器走 OVRTX | `add_usd` 加 `newton-usd-schemas` | 除 RTX 查看器外是 |
| Genie Sim 3.2（智元）、InternUtopia 2.2、unitree_sim_isaaclab | Isaac Sim | Isaac RTX | Isaac USD | **否** |

值得注意的是：ManiSkill3 论文报告 4090 上 RGBD 加分割 3 万帧/秒以上、Cartpole 128 环境显存
3.5 GB 对 Isaac Lab 14.1 GB（Isaac Lab 侧在光线追踪，条件不等同）。它证明"PhysX GPU 加自研
Vulkan 渲染器、不碰 Kit"这条路有人走通了，而且 svulkan2 是 Apache-2.0，
是自建渲染器时最值得参考的代码库。国内三个大型开源仿真项目至今全部建在 Isaac Sim 上。

## 五、工作分解

按工作线列出。规模为**粗估的工程师人月，无先例可校准**，只用于比较优先级，不作承诺。
"L2 增量"指若连 Warp、usd-exchange 等 NVIDIA 维护的开源件也不用时额外增加的部分。

| # | 工作线 | 现状 | 可用件 | 自建内容 | 粗估 | L2 增量 |
| --- | --- | --- | --- | --- | --- | --- |
| W1 | USD 运行时与资产传输 | `omni.client` 传输，`usd-exchange` 提供 `pxr` | `usd-core`、httpx/boto3、Ar 2.0 URI 解析器 | HTTP/S3 下载与缓存；可选自写 URI 解析器 | 1 至 2 | 换 `usd-core` 并解决 `usd-optimize` 共存：加 1 至 2 |
| W2 | 资产导入 | 网格转换仍走 Kit | trimesh/Open3D/Assimp、OCCT、CoACD、NVIDIA 两个 Apache 转换器 | 无 Kit 网格到 USD 转换器；CAD 管线 | 2 至 3 | 转换器换 Newton 直读或 MuJoCo 导出：加 2 至 3 |
| W3 | 资产库重建 | 472 处引用 NVIDIA 资产 | Menagerie、厂商 URDF、Poly Haven、ambientCG、Infinigen | 机器人、环境、道具、材质、传感器型号的创作与许可审核 | 6 至 12 加内容预算 | 无 |
| W4 | Blender 物理导出插件 | 无 | `USDHook`、`get_prim_map()`、usdchecker、`usd-validation-nvidia` | UsdPhysics 与厂商 schema 写入、分层结构与变体保持、基元补写、非破坏导入 | 3 至 6 | 无 |
| W5a | 分块渲染器补齐 | Newton Warp 渲染器 10/15 项 | 现有 Warp 渲染器代码 | 运动矢量、`simple_shading_*`、更多光源与 MaterialX 子集、畸变射线表 | 4 至 8 | 渲染器随 Warp 一起重写 |
| W5b | 离线超写实管线 | 无 | Cycles/hdCycles、hdMoonray、Blender 无头进程 | USD 到渲染器的桥、AOV 到 15 项契约的映射、分割合成 | 2 至 4 | 无 |
| W5c | Vulkan 光追实时渲染器（可选） | 无 | svulkan2（Apache-2.0）、Falcor（BSD-3）作参考 | 厂商中立、GPU 常驻、外部内存到 CUDA/torch、15 项契约 | 12 至 24 | 这是 L3 的唯一渲染出路 |
| W6 | 传感器物理模型 | 射线投射齐全 | OpenCV 畸变模型、SDFormat 噪声模型、公开规格 | 激光雷达强度/多回波/发散/噪声、雷达几何模型、超声锥、IMU 随机游走、型号库 | 3 至 6 | 无 |
| W7 | 合成数据与域随机化 | 纹理随机化抛异常 | `VisualMaterialBatch` 契约、bop_toolkit、pycocotools | 事件项、写入器、包围盒与位姿标注、轻量编排 | 2 至 4 | 无 |
| W8 | ROS 2 桥 | 无 | rclpy、topic_based_ros2_control、simulation_interfaces 2.1、rclcpp 旁车 | 全部 | 3 至 5 | 无 |
| W9 | 运动生成与控制 | RMPflow 依赖闭源 Lula | cuRobo（Apache-2.0）、Pink/mink、OMPL/VAMP、Pinocchio | USD 到 URDF 导出器；RMPflow 迁移到 cuRobo 或差分 IK | 2 至 4 | cuRobo 换 PyRoki：加 2 至 4 |
| W10 | 遥操作与 XR | 头显全走 CloudXR | Monado、WiVRn、pyopenxr、vuer、avp_stream | 设备层重写，重定向器复用 | 3 至 6 | 无 |
| W11 | 可视化、推流、检视 | Kit 视口与 UI | Newton GL/viser/rerun、GStreamer webrtcsink、usdtweak | 删除 Kit UI；推流接入；检视器集成或 imgui 属性面板；机器人工具界面重写 | 2 至 4 | 无 |
| W12 | 部署与基准 | 容器基于 Isaac 镜像，基准用 Isaac 服务 | `nvidia/cuda` 加 Vulkan ICD、pynvml | 镜像、帧时记录器 | 1 | 无 |
| W13 | 物理补齐 | 表面吸盘、FEM、流体仅 PhysX | Newton VBD/MPM | 吸盘重做、可变形任务调参、剩余任务的 Newton 预设 | 2 至 4 | Warp 移植：见国产芯片文档 |

合计：不含 W5c 与内容预算约 **34 至 64 人月**；含 W5c 约 46 至 88 人月。
W3 的内容制作与许可审核不计入工程人月。L2 再加约 7 至 13 人月，L3 另见国产芯片文档。

需要整体放弃的代码：`isaaclab_physx`（29,094 行）、`isaaclab_ov`（28,623 行）、
`isaaclab_visualizers/kit`、`isaaclab/ui`、`isaaclab/envs/ui`、Fabric 相关三个文件、
`isaaclab_teleop` 的 Kit XR 与 CloudXR 路径。

## 六、推荐路线

### 6.1 定位

- **底座**：Isaac Lab 自己的后端接口加 Newton 加 OpenUSD。不引入游戏引擎。
- **Blender**：资产创作、USD 物理导出插件的宿主、离线超写实渲染。始终独立进程
  （`bpy` 只支持 Python 3.13，与现有 3.12 环境本就不兼容）。
- **O3DE**：L1 下不引入。它能给的（Vulkan 光栅、编辑器外壳）与要付出的（萎缩中的 C++ 代码库、
  无 USD、无 GPU 互操作）不成比例。
- **渲染**：近期 W5a 加 W5b；中期若需要厂商中立的实时超写实渲染，W5c 以 svulkan2 为起点。

### 6.2 阶段

**阶段零：证伪（2 至 4 周）。** 五个快速实验：Linux 上 Blender 应用内 `import pxr`；
hdCycles 对一个 Isaac Lab 分层 USD 资产原地渲染；usdtweak 打开分层资产并往返 `Physics` 变体集
与未注册 applied schema；Newton Warp 渲染器加运动矢量的可行性评估；
Vulkan 外部内存导出到 torch 张量的最小验证（NVIDIA 卡上）。另加一项法务动作：
就 NVIDIA 资产许可对第三方仿真器的适用性取得书面意见。

**阶段一：去 NVIDIA 二进制（一个季度）。** W1、W2、W12、W13 的吸盘与预设部分；
删除 `isaaclab_physx`、`isaaclab_ov` 与 Kit UI；替换 `omniverseclient`。
验收：`uv sync` 不再从 `pypi.nvidia.com` 拉取任何包，44 个训练环境在 Newton 上全部通过冒烟测试。

**阶段二：创作与资产（两个季度，与阶段三并行）。** W4、W3、W7。
验收：至少 5 个机器人与 2 个环境以自有资产在 Blender 中创作、经插件导出、通过
`usdchecker` 与 `usd-validation-nvidia`，并在 Newton 上训练到与 NVIDIA 资产相当的回报。

**阶段三：感知（两个季度）。** W5a、W5b、W6。
验收：Newton Warp 渲染器覆盖 15 项契约中的 13 项以上；离线管线产出带语义与实例分割的
COCO/BOP 数据集；激光雷达模型对照一款真实雷达的公开点云做定性验证。

**阶段四：工具链（与二、三并行）。** W8、W9、W10、W11。
验收：一个任务经 ROS 2 跑通 Nav2 或 MoveIt 2 闭环；一套头显经 Monado 加 WiVRn 完成遥操作采集。

**阶段五（可选）：W5c。** 只有在阶段三证明 Warp 渲染器的画质或吞吐不够、
或 L3 目标被确认时才启动，并以阶段零的 Vulkan 外部内存实验结果为前提。

### 6.3 与另外两份文档的关系

- 前一轮的 Blender 插件、纹理随机化事件项、网格转换、Cycles 验证四项建议原样进入 W4、W7、W2、W5b。
- 国产芯片文档的阶段一（设备抽象改造）与本文阶段一可以合并为同一批 PR。
- 本文的 W5c 是国产芯片文档"渲染层近期放弃"之后唯一的长期出路，两份文档在此汇合。

## 七、风险

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| NVIDIA 资产许可不允许与第三方仿真器共用 | W3 从"建议"变为"前置条件"，影响全部任务 | 阶段零取得法务意见；无论结论如何都启动 CC0 资产替换 |
| Warp 渲染器画质不足以 sim-to-real | 图像观测任务效果下降 | 阶段三做一次与 OVRTX 的对照实验；不足则启动 W5c |
| MJWarp GPU 非确定性 | 数据集复现与回归测试受影响 | 跟踪上游 PR #1422；关键实验用 CPU 或接受容差 |
| hdCycles 不可用 | 离线管线退回 Blender 进程内渲染，架构复杂化 | 阶段零一周内证伪 |
| Blender USD 导出器变更破坏插件 | 维护负担 | 锁定 5.2 LTS；`USDHook` 自 4.0 起六版只有增量变化 |
| OpenUSD Vulkan 后端长期实验性 | Storm 不能作为跨厂商查看器 | 查看器继续用 Newton GL/viser，不依赖 Storm |
| ROS 2 桥与 Isaac Lab 向量化环境的语义冲突 | 单机器人 ROS 工作流与 N 环境训练难以共存 | 桥只服务 `num_envs=1` 的部署与验证场景 |
| 头显生态碎片化 | Vision Pro 只能走 WebXR | 以 Quest/Pico 加 WiVRn 为主线 |
| 工作量估计无先例 | 偏差可能达两倍 | 每阶段末重估，阶段零与一的数据用于校准 |
| 两份调研共用的 Warp 风险 | L1 的全部物理成果在 L3 下要重做 | 决策前先确认目标是 L1 还是 L3，不要同时追求 |

## 八、待验证事项

快速可证伪：

- [ ] Linux 上 Blender 5.2 应用内 `import pxr` 是否可用（决定插件能否进程内操作 stage）。
- [ ] hdCycles 对 Isaac Lab 分层 USD 资产的原地渲染（一周）。
- [ ] usdtweak 对 `Physics` 变体集与未注册 applied schema 的往返保真。
- [ ] Vulkan `VK_KHR_external_memory_fd` 导出的图像在 torch 中零拷贝读取的最小样例。
- [ ] 用 `usdchecker` 与 `usd-validation-nvidia` 跑一遍现有资产的基线通过率。

需要实测：

- [ ] Newton Warp 渲染器在 512×512 与 1024×1024 下的单帧延迟与瓶颈构成。
- [ ] MJWarp 确定性模式（PR #1422）在代表性任务上的吞吐代价。
- [ ] `rclpy` 在 `num_envs=1` 下发布关节状态与图像的延迟。
- [ ] Monado 加 WiVRn 加 pyopenxr 在 Quest 3 上的手部追踪延迟。

需要核对：

- [ ] NVIDIA 资产许可对第三方仿真器的适用性（法务）。
- [ ] MotrixSim 的鲲鹏认证与引擎开源状态（本轮未找到一手证据，与前一轮文档说法冲突）。
- [ ] SimReady Foundation 规则集的许可（站点未标明）。
- [ ] `usd-core` 与 `usd-optimize` 能否共存（L2 前提）。
- [ ] O3DE 26.10.0 是否已发布及其是否改变本文任何结论（原定 2026-10-28）。
- [ ] O3DE 提交统计来自子代理的本地克隆，发布前复核。
- [ ] Isaac Sim 6.2 的 Replicator 完整写入器清单（本轮只确认 Basic/Kitti/Omni 三个）。
- [ ] SAPIEN 光追着色包是否使用 Vulkan KHR 光追及其在 AMD/Intel 上的状态。

## 九、参考来源

本仓库（commit `e1c21cfe6`）：

- `pyproject.toml`（NVIDIA 包与索引配置）
- `source/isaaclab/isaaclab/utils/assets.py`、`sim/converters/mesh_converter.py`、
  `controllers/utils.py`、`envs/mdp/events.py`、`benchmark/benchmark_core.py`
- `source/isaaclab/isaaclab/renderers/output_contract.py`、`sensors/ray_caster/patterns/patterns.py`
- `source/isaaclab_teleop/`、`source/isaaclab_mimic/isaaclab_mimic/motion_planners/`
- `docs/source/concepts/renderers.rst`、`visualization.rst`、`deformables.rst`、
  `docs/source/features/isaac_teleop.rst`、`multi_gpu.rst`、`docs/source/how-to/import_new_asset.rst`

Isaac Sim：

- https://github.com/isaac-sim/IsaacSim（`LICENSE`、`VERSION`、`deps/`、`source/libraries/README.md`）
- https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html
- https://docs.isaacsim.omniverse.nvidia.com/6.2.0/common/license-isaac-sim-additional.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_rtx.html
- PyPI：ovrtx、ovphysx、ovstage、omniverseclient、isaacsim、isaacteleop、urdf-usd-converter、
  mujoco-usd-converter、usd-exchange、physx-usd-schemas、isaacsim-asset-isolated

OpenUSD 与 Hydra：

- https://github.com/PixarAnimationStudios/OpenUSD/releases 与 CHANGELOG（v25.05 至 v26.11.1）
- https://openusd.org/release/api/usd_physics_page_front.html 、`parseUtils.h`
- https://pypi.org/project/usd-core/
- https://github.com/OpenMoonRay/openmoonray 、https://github.com/blender/cycles（`src/hydra`）、
  https://github.com/ospray/hdospray（已归档）、https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderUSD 、
  https://github.com/Autodesk/Aurora 、https://github.com/NVIDIAGameWorks/Falcor 、
  https://github.com/DiligentGraphics/DiligentFX
- https://github.com/cpichard/usdtweak 、https://github.com/dreamworksanimation/usdmanager
- https://github.com/NVIDIA-Omniverse/usd-exchange 、https://github.com/NVIDIA-Omniverse/usd-validation-nvidia
- https://github.com/NVIDIA/MDL-SDK（Distiller）、https://github.com/AcademySoftwareFoundation/MaterialX
- https://aousd.org/working-groups/ 、https://reps.openrobotics.org/rep-0158-2006/
- https://github.com/PixarAnimationStudios/OpenUSD-proposals/pull/111 、/pull/119
- https://github.com/RobotLocomotion/drake/pull/23244（删除 USD 解析器）

Blender：

- https://pypi.org/project/bpy/ 、https://developer.blender.org/docs/release_notes/5.3/
- https://docs.blender.org/manual/en/latest/files/import_export/usd.html
- https://github.com/dfki-ric/phobos 、https://github.com/DLR-RM/BlenderProc

O3DE：

- https://github.com/o3de/o3de/issues/19595 、/issues/20016 、/pull/18450 、/pull/17974 、/issues/15848 、/issues/18873
- https://github.com/o3de/o3de-extras/issues/804 、/issues/1080 、/issues/955
- https://github.com/o3de/sig-simulation/issues/60 、/pull/92 、/pull/96 、/issues/89
- https://github.com/o3de/sig-graphics-audio/issues/51 、/issues/120
- https://github.com/o3de/sig-release/issues/376 、/pull/375
- https://docs.o3de.org/docs/release-notes/2605-0-release-notes/
- https://docs.o3de.org/docs/user-guide/interactivity/robotics/simulation-interfaces/
- https://o3df.org/community/members/ 、https://robotec.ai/ 、https://github.com/RobotecAI/robotec-o3de-tools
- https://github.com/iamjbq/JoltPhysics 、https://github.com/jorgetbruno/JoltPhysics

物理引擎：

- https://mujoco.readthedocs.io/en/stable/changelog.html 、/OpenUSD/index.html 、/mjx.html
- https://github.com/google-deepmind/mujoco_warp（PR #1113、#1300、#1422）
- https://github.com/Genesis-Embodied-AI/Genesis/releases 、https://pypi.org/project/quadrants/ 、
  https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is
- https://github.com/NVIDIA-Omniverse/PhysX（`LICENSE.md`、`physx/LICENSE.md`）
- https://github.com/haosulab/SAPIEN 、https://github.com/haosulab/ManiSkill 、https://arxiv.org/abs/2410.00425
- https://pypi.org/project/motrixsim-core/ 、https://motrixsim.readthedocs.io/ 、https://arxiv.org/abs/2605.30313
- https://github.com/newton-physics/newton（CHANGELOG 1.0.0 至 1.7.0）、https://github.com/newton-physics/newton-usd-schemas
- Jolt、Bullet、Drake、DART、Chrono、Rapier、Gazebo、Webots 的发布页与文档（见各引擎官网）

传感器、资产、ROS 2、运动规划、XR、推流、合成数据、部署：

- https://github.com/RobotecAI/RobotecGPULidar/releases 、https://github.com/3dgeo-heidelberg/helios 、
  https://radarsimx.com/product/radarsimpy/ 、https://raw.githubusercontent.com/gazebosim/sdformat/main/sdf/1.12/noise.sdf
- https://github.com/newton-physics/urdf-usd-converter 、https://github.com/Open-Cascade-SAS/OCCT/releases 、
  https://github.com/kmammou/v-hacd 、https://pypi.org/project/coacd/ 、https://pypi.org/project/onshape-to-robot/
- https://github.com/ros-controls/mujoco_ros2_control 、https://github.com/PickNikRobotics/topic_based_ros2_control 、
  https://github.com/ros-simulation/simulation_interfaces 、https://github.com/vybhav-ibr/newton_ros 、
  https://docs.ros.org/en/rolling/Releases/Release-Lyrical-Luth.html
- https://github.com/NVlabs/curobo/releases/tag/v0.8.0 、https://github.com/ompl/ompl/releases 、
  https://github.com/KavrakiLab/vamp 、https://github.com/chungmin99/pyroki 、https://pypi.org/project/pin-pink/
- https://monado.freedesktop.org/ 、https://github.com/WiVRn/WiVRn/releases 、https://pypi.org/project/pyopenxr/ 、
  https://pypi.org/project/vuer/ 、https://github.com/unitreerobotics/xr_teleoperate 、
  https://github.com/Improbable-AI/VisionProTeleop 、https://github.com/nerfstudio-project/viser/pull/408
- https://gstreamer.freedesktop.org/releases/1.28/ 、https://github.com/LizardByte/Sunshine 、
  https://github.com/selkies-project/selkies 、https://pypi.org/project/aiortc/
- https://github.com/google-research/kubric 、https://github.com/princeton-vl/infinigen 、
  https://github.com/thodan/bop_toolkit 、https://github.com/facebookresearch/habitat-sim（已归档）
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html 、
  https://github.com/haosulab/ManiSkill/blob/main/docker/Dockerfile 、https://github.com/NVIDIA/gpu-operator/releases
