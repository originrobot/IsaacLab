# Isaac Sim 编辑与渲染能力替代调研：O3DE 与 Blender

调研日期：2026-10-29（第二版）
代码基线：本仓库 commit `e1d7e9d4b`（核验时 HEAD 为 `5d0dd9100`，多出两个文档提交）

## 结论先行

需要替换的范围远小于预期。USD 物理 schema 的编写已经不依赖 Kit，超写实渲染也已有无 Kit 的
OVRTX 路径。真实缺口只有五项，最大的一项是交互式场景编辑。两个候选中 Blender 明显更合适：
它缺 UsdPhysics 导出，但有稳定的 Python 注入点可以自建；O3DE 的 USD 支持实质为零、
没有路径追踪器、真值通道只有 RGB 与深度，且这三块都无人维护。

但在动手前必须先回答一个分叉：目标是脱离 Omniverse Kit，还是脱离 NVIDIA。两者的答案完全不同。

## 一、修订说明

第一版把问题理解成"替换整个仿真运行时"，因而把篇幅放在 GPU 批量物理、Warp 内核与张量桥上。
实际需求是替换 Isaac Sim 的**场景编辑与超写实渲染**能力，仿真绝大多数时间 headless 运行，
不开 Isaac Sim。本版按正确框架重写，第一版中以物理和 RL 吞吐为依据的论证与排序已全部删除。

本版还修正了第一版的一处数据错误：渲染输出契约是 **15 种**而非 9 种。

## 二、问题定义

### 2.1 一个必须先回答的分叉

这是本轮调研最重要的发现之一。

`renderer=ovrtx` 在文档中被标为**推荐**的 RTX 渲染器，Isaac RTX 被标为**遗留**
（`docs/source/concepts/renderers.rst:39-58`）。OVRTX 不需要 Isaac Sim，也不需要 Kit。

但 OVRTX 是 NVIDIA 的**专有、预发布、仅二进制**组件，采用 NVIDIA 软件许可协议，
明示"不接受外部贡献"，其 sdist 只是一个从 `pypi.nvidia.com` 拉取平台二进制的存根，
仓库锁定 `ovrtx==0.5.1.385782`（`pyproject.toml:231`）。

因此：

- **若目标是脱离 Kit 这个应用**，超写实渲染已经解决了，不需要替换任何东西。
- **若目标是脱离 NVIDIA**，OVRTX 什么也没解决，必须自建渲染路径，参照物是 15 项输出契约。

考虑到本项目的第二个目标是国产芯片生态适配（见
[国产芯片生态适配调研](domestic-accelerator-adaptation.md)），实际答案大概率是后者，
OVRTX 只能作为过渡而非终点。后文的建议按"最终要脱离 NVIDIA"组织，并标明哪些步骤在
只求脱离 Kit 时可以省略。

### 2.2 已经不是缺口的部分

这部分有明确的代码证据，第一版把其中几项误判为缺口。

**USD 物理 schema 的编写不依赖 Kit。** `sim/spawners/` 下每个文件只导入 `pxr`，
没有任何 `omni.*`、`isaacsim.*` 或 `carb`。`sim/schemas/schemas.py` 只导入
`pxr.{Gf,Sdf,Usd,UsdGeom,UsdPhysics}`，并通过 `prim.AddAppliedSchema(name)` 加命名空间属性的
方式**无代码地**写入 PhysX、Newton、MuJoCo 的 schema。**没有任何生产代码导入
`pxr.PhysxSchema`。** 语义标注用的是上游的 `UsdSemantics.LabelsAPI`，仓库内
`SemanticsAPI` 出现次数为零。截至 2026 年 10 月，`physx-usd-schemas`、`newton-usd-schemas`
与 MuJoCo 的 `mjcPhysics` 均已作为无代码 wheel 发布，可注册进标准 `usd-core`。
2025 年 5 月"PhysxSchema 仅 Kit 可用"的说法已经过时。

**轻量检视已经够用。** 五个可视化后端由同一个选择器驱动（`app/sim_launcher.py:199-205`），
其中四个无需 Kit。标记、实时曲线、相机推流、视频录制、键盘输入、暂停与重置都有无 Kit 实现，
`VisualizationMarkers` 核心没有任何 Kit 导入。

**文档中的 GUI 工作流几乎都可脚本化。** 约 135 处传 `--viz kit` 的文档命令中，
**只有 7 处没有脚本替代**，其中 4 处是 XR 专用。排除 XR 后只剩三项：Stage Recorder 录制与
Layers 回放、USD Composer 创作、一个缓存开关。**没有任何文档让用户在 Isaac Lab 流程中使用
内容浏览器、拖放或 `Create ->` 菜单，也没有让用户在 GUI 里创作材质。**

**Replicator 的使用面很窄。** Isaac Lab 只用了 8 个随机化器中的 2 个，4 个 writer 中的 0 个，
没有用编排器，也没有用散布与位姿随机化器。完整的 `rep.*` 清单是 21 个符号。
Replicator 这个产品本身与 Kit 不可分（只随 5.5 GB 的专有 `isaacsim-extscache-kit` wheel 发布，
硬依赖 OmniGraph 与 `omni.syntheticdata`，没有独立的 PyPI 包），但替换 Isaac Lab
**实际用到的那部分**工作量很小。

### 2.3 五项真实缺口

按"杠杆率"排序，不按难度。

**缺口一：纹理与材质的域随机化。代码里写明了没有替代。**
`isaaclab_physx/envs/mdp/events.py:465-471` 的 `_get_replicator()` 直接抛
`NotImplementedError("Replicator visual events require Isaac Sim (Omniverse Kit).")`。
唯一的无 Kit 项是 `isaaclab_newton/envs/mdp/events.py:278` 的 `randomize_visual_shape`，
它对 `"color"` 以外的任何通道都抛异常。

值得注意的是**渲染器无关的契约已经存在并被三个后端全部实现**
（`VisualMaterialBatch` 与 `BaseRenderer.visual_material_writer`，`renderers/base_renderer.py:30,58`），
缺的只是驱动它做纹理随机化的 MDP 事件项。这是五项里最小的缺口，杠杆率最高。

**缺口二：网格转 USD 的导入（OBJ/STL/FBX）。**
`sim/converters/mesh_converter.py` 是 `sim/` 下最后一个使用 `omni.kit.commands` 与
`omni.kit.asset_converter` 的文件。相比之下 URDF 与 MJCF 已经通过
`isaacsim-asset-isolated` 实现无 Kit。而且 NVIDIA 自家的 `urdf-usd-converter` 与
`mujoco-usd-converter` 是 **Apache-2.0、不依赖 Kit、已在公开 PyPI 上**，
所以导入侧的大部分缺口可以用开源代码补上。

**缺口三：交互式场景创作与检视。这是真正的 Kit 独有能力。**
运行时的 USD prim 检视与编辑，以及属性、图层、舞台三个面板
（`docs/source/concepts/visualization.rst:510-511`）。仓库内与四个无 Kit 查看器都不提供
prim 选择加属性编辑加图层变体管理。

但要注意 NVIDIA 自己的表述是"GUI 中能做的大部分操作都能用 Python 完成"。
也就是说替换 Kit 丢的是**工具而非能力**：碰撞体可视化、质量分布标示、驱动调参滑块、
各类检视器。这决定了这一项的性质是工效问题，不是功能阻断。

**缺口四：MDL 作为材质语言。**
Isaac Lab 用普通的 `UsdShade` 写 MDL 引用（这很容易重写，也存在 `UsdPreviewSurface` 路径），
但解析与着色需要支持 MDL 的渲染器，而仓库内**没有任何 `.mdl` 文件**，全部从 Nucleus 解析。
Replicator 的随机化器写的是 OmniPBR 的着色器输入（`diffuse_color_constant`、
`diffuse_texture`、`texture_rotate`），所以缺口一与缺口四是耦合的。

**缺口五：资产目录与传输。**
走 `omni.client` 而非 HTTP 获取，带 S3 与中国镜像配置
（`utils/assets.py:66-96,125-132`），仓库内只有 12 个 `.usda` 测试夹具。
**仓库里完全没有资产创作工具**，只有转换器与检查器。创作被默认发生在 Omniverse 或其他
DCC 工具中。

### 2.4 两个必须锚定的参考物

**其一：15 项渲染输出契约**（`renderers/output_contract.py:19-39`）。

| 渲染器 | 支持项数 | 缺失 |
| --- | --- | --- |
| Isaac RTX | 15 | 无 |
| OVRTX | 14 | `instance_id_segmentation_fast` |
| Newton Warp | 10 | 运动矢量、`simple_shading_*`、`instance_id_segmentation_fast` 等 |

任何替代渲染器都应以此为验收参照。第一版文档写的 9 种是错的。

**其二：分层多物理资产格式**（`urdf_converter_cfg.py:190-207`）。

导入器产出的结构是：一个接口层 `.usda`，加上 `payloads/{base,instances,geometry,materials}`，
再加上 `payloads/Physics/{physics,physX,mujoco}.usda`，后者挂在一个名为 `"Physics"` 的
变体集之下。

这给外部编辑器设了两条硬要求：**必须能在 payload 层之上往返变体集**，
并且**必须原样保留未注册的 applied API token**。一个会把未知 applied schema
"校验掉"的编辑器，会静默销毁全部 Newton、MuJoCo、PhysX 调参。

这一条直接决定了 Blender 与 O3DE 的命运，见后文。

## 三、Blender 评估

### 3.1 创作往返：物理完全不导出，但注入点干净

**Blender 不导出任何 UsdPhysics，不是往返丢失，是单向也不写。**
5.2 LTS 的 USD 文档把导入导出选项列得很详尽，没有一项涉及物理。逐项确认缺失：
`UsdPhysicsScene`、`RigidBodyAPI`、`CollisionAPI`、`MeshCollisionAPI`、`MassAPI`、
`MaterialAPI`、`Joint` 及 `Revolute`/`Prismatic`/`Spherical`/`Fixed`/`Distance`/D6、
`LimitAPI`、`DriveAPI`、`ArticulationRootAPI`、`CollisionGroup`、`FilteredPairsAPI`。
4.5 LTS、5.0、5.1、5.2 LTS 与开发中的 5.3 发布说明均如此。

**更严重的是层、引用与变体也"不处理"、"不导出"。** 对照 2.4 节的分层资产格式，
这意味着今天的 Blender 导出器不仅丢物理，还会把整个分层结构压平。

**但构建路径比预期好。** `bpy.types.USDHook.on_export` 会在保存前把活的 `pxr.Usd.Stage`
交给 Python，5.1 新增的 `get_prim_map()` 返回 Blender 对象到 USD prim 路径的精确映射。
该 API 自 4.0 起六个版本只有增量变化，`on_material_export` 已经在向 Python 传递
`pxr.UsdShade.Material` 对象。这正是自研插件需要的注入点：**纯 Python，API 稳定，
不需要 fork C++**。对称的 `on_import` 钩子是实现非破坏性往返的唯一途径。

**现有工具都补不上这个缺口。**

- NVIDIA SimReady Blender CORE 插件 v2026.4.0 是唯一在产的 UsdPhysics 创作工具，
  但**仅支持 Windows 10/11**，要求 Blender 5.0 以上。文档化的只有旋转、滑动、固定三种关节
  及其限位；球面与距离关节"可以应用但未文档化"；**D6、articulation root 与质量完全未提**；
  驱动"仍在开发中""不总能正常工作"；物理材质写进 **JSON 旁车文件**而非
  `UsdPhysicsMaterialAPI`。它确实支持 MJCF 的导入导出。
- 其他插件都不写 USD：Phobos 做 URDF/SDF/SMURF 且停在 Blender 3.3 LTS；
  LinkForge 做 URDF/XACRO（GPL-3，Blender 4.2 以上，跨平台，约 7500 次下载，
  带惯量计算与关节标示）；ColliderGen 做 SDF。社区没有 UsdPhysics 创作插件。
- **Hydra 是错误的着力点**：它是渲染抽象，没有 UsdPhysics 表示，且 `BlenderSceneDelegate`
  只有 C++ 实现。
- **上游不会自己解决**：blender.org 的开发板块把"核心 USD 创作"标为
  "需要资金与利益相关方"，没有目标版本。

**碰撞体创作**：Blender 内置只有单一的凸包算子，没有内置凸分解。V-HACD 可通过免费插件使用
但需要外部可执行文件，CoACD 只有付费插件。另外 Blender **不能导出
`UsdGeomCapsule`/`Sphere`/`Cylinder` 基元**，这些必须由钩子直接写入。

**先例**：Wandelbots（德累斯顿）在 2025 年 Blender 大会上作过题为
"An Automation Pipeline for Robotics and Animation with USD" 的报告，
用的是 Blender 核心加 OpenUSD 平台加自研 Blender 扩展，面向工业机器人。
该报告录像未取得，是后续最值得跟进的一条线索。

### 3.2 渲染：Cycles 可用，EEVEE 不可用

**Cycles 真正支持无头运行**，不需要 OpenGL 或 X，5.2 支持 CUDA、OptiX、HIP、oneAPI、Metal。
5.0 到 5.2 恰好补上了数据集生成最需要的三项：Persistent Data 在帧间复用 BVH 与修改器求值
（针对未变化的对象）、5.2 的纹理缓存降低了纹理密集随机化场景的启动开销与内存、
5.2 让几何内存与 Blender 共享以改善同步。

**EEVEE 不可用于本用途**，有两个独立理由：4.2 至 5.2 的全部文档中**没有任何**支持无显示器
渲染的官方表述，最新权威说法仍是 2.8 时代的"必须有显示设备"；
以及 **EEVEE 完全没有索引通道**，无法产出真值。

**真值通道**。Cycles 提供深度、法线、运动矢量、位置、UV、降噪反照率、物体与材质索引、
Cryptomatte（物体/材质/资产）、光组、阴影捕捉。对照 Isaac Lab 的契约，大部分可直接映射，
语义分割、实例分割与到成像平面距离需要自行合成。两个具体的坑：
**开启运动模糊时运动矢量通道会被禁用**；EEVEE 没有索引通道。

**一个可能改变架构的选项：Cycles 的 Hydra delegate。**
Cycles 是 Apache-2.0，可作为 Hydra delegate 对着标准 OpenUSD 25.11 以上构建。
如果可行，Isaac Lab 的 USD stage 就能**原地渲染，不需要导入 Blender，也不涉及 GPL**。
但它没到生产可用：2026 年 8 月的一个 PR 说明 `HdCyclesMaterial`
"从来只处理 UsdPreviewSurface 与原生 cycles_* 节点"，不支持 MaterialX；
最近的用户报告（2025 年 1 月）是 Houdini 崩溃与 usdview 加载插件失败。
**建议用一周做验证性尝试后再决定是否纳入方案。**

**性能证据很薄，必须自测。** 可引用的只有 BlenderProc 的每张 1 至 4 秒与
"5 块 GPU 一夜 5 万张"（**未注明 GPU 型号与分辨率**）、2019 年 BlenderProc 论文的
单 GPU 约每小时 3000 张（**未注明硬件**）、Infinigen 的每对 1080p 立体图约 3.5 小时
（二手来源，硬件未确认）。Blender Open Data 的分数在不同来源间差异巨大
（RTX 5090 有约 9150、16066、17800 三种报告），原因是基准版本不同。
**公开资料中不存在任何注明硬件与分辨率的单帧延迟实测。**

**另外两项空白**：Blender 对渲染确定性没有任何公开表述，没有跨 GPU、跨驱动、跨版本的
位级一致性声明；Cycles 与 Omniverse RTX 在 sim-to-real 上没有任何方向的公开对比。
单向证据倒是很强：BOP 用 35 万张 BlenderProc PBR 图相比此前的合成数据提升了
20.2 与 19.5 个绝对百分点；2026 年一篇基于 BlenderProc 的论文报告纯合成训练在真实图像上
取得 95.1% 至 98.7% 的 mAP@50（RTX 4090）。如果方案需要与 RTX 的对比，只能自己做。

### 3.3 许可：在本用法下基本不是约束

需要区分三种情形：

| 情形 | 义务 |
| --- | --- |
| 美术用 Blender 创作资产 | 无。Blender 许可页明示产出归用户所有 |
| 内部自用插件 | 无。GPL 在"发布"时才触发 |
| 发布到 extensions.blender.org | 要求 GPL-3.0-or-later |
| 把 bpy 嵌入交付二进制 | 真实约束，应避免 |

一个具体的连带约束：**任何基于 BlenderProc 构建的东西在分发时继承 GPL-3.0**，
因此域随机化采样器建议自己实现而非依赖它。

渲染管线保持为独立进程、脚本不公开发布，则整条链路没有分发义务。

## 四、O3DE 评估

### 4.1 USD 互通：实质为零，且被明确排除

这是 O3DE 的硬止步，而且比第一版了解到的更严重。

- 所谓"2024 年 10 月合入的 USD 导入器"实际是 **PR #18361：2 个文件，净增 5 行**，
  内容是把 `.usd/.usda/.usdz/.usdc` 加进 `Registry/sceneassetimporter.setreg` 的扩展名列表。
  文件随后被当作 FBX 一样，经 Assimp 的 TinyUSDZ 读取。
- 25.05.0 的发布说明原话是"**非合成的** USD 几何与动画"。
- **路线图上的"改进 OpenUSD 支持"没有交付。** 25.10.0 与 26.05.0 都没有 USD 条目；
  `Code/Tools/SceneAPI` 自 2024 年 1 月至今的提交记录中**没有任何 USD 功能工作**，
  只有 Assimp 版本升级。
- **导出不存在**，且 sig-simulation PR #92 把它明确排除，原文为："把 O3DE prefab 转换为
  USD 文件，本次初始工作不包括从 O3DE 内容创建 USD 文件的能力……这是长期可以再看的事。"
- 该 RFC 刻意选择 Assimp 而非 Pixar 的 USD 库。**Assimp 没有物理模型**，
  因此两个方向上都没有任何 UsdPhysics 或 PhysxSchema 映射。o3de-extras 中不含任何 USD 文件。
- Issue #15848 自 2023-04-30 开启至今未关闭，2026-03-13 的最后一条评论说导入"仍然很粗糙"。

对照 2.4 节的分层资产格式要求，O3DE 连扁平 USD 都只能当几何读，谈不上往返变体集与
保留未注册 applied schema。

### 4.2 Atom：没有路径追踪器

全引擎 36,557 个路径中**没有一个匹配 `pathtrac`**，也没有降噪器。
实际存在的是光追加速的光栅化：一个 TLAS pass、
`ReflectionScreenSpaceRayTracing.pass`（屏幕空间反射链中用作离屏回退的真实 RayTracingPass）、
以及 DiffuseProbeGrid gem（DDGI）。

华为的光追材质重构 issue（sig-graphics-audio#152）**自 2023-09-20 开启至今未关闭**，
其中对现状的自述是：光追"假设只存在一种材质类型……使用单一命中着色器，
重新实现了一个几乎但不完全是 BasePBR 的材质类型"。也就是说 **Material Canvas 创作的材质
根本不参与光线追踪**。25.10 与 26.05 只交付了铺垫性工作。

与 Cycles 这样的生产级路径追踪器、与 Omniverse RTX 相比，Atom 的设计目标本就不同，
不适合承担超写实渲染角色。

### 4.3 真值通道与域随机化

**真值通道只有 RGB 与深度。** `CameraChannelType` 枚举恰好两个值。
分割能力只存在于激光雷达侧（`ROS2Sensors/Lidar/ClassSegmentationBus.h`）。
法线与运动矢量作为内部目标存在，但没有任何接口把它们暴露出来。没有物体或实例 ID。

**域随机化完全不存在。** 不是功能弱，而是这个类别在两个仓库里都不存在。
`MachineLearning` gem 是一个手写的多层感知机加 MNIST 加载器，与此无关。

**扩展路径倒是干净**：JSON 格式的 `.pass` 模板加 `AttachmentReadback`
（RGBD 传感器已经在从 `DepthPrePass` 回读 `"DepthLinear"`）。

### 4.4 两个意外的正面发现

- **ScriptAutomation gem 是现成的批量渲染骨架**，提供 `LoadLevel`、`SetCamera`、
  `ResizeViewport`、`CaptureScreenshot`，以及关键的 **`CapturePassAttachment`**
  （可导出任意 pass 附件），外加基线图像比对。配合自定义 `.pass` 模板，
  这就是一条**不需要写 C++ 的 AOV 导出路径**。
- 编辑器的无头自动化参数已从 O3DE 自己的测试框架中得到验证：
  `-runpythontest <file> -BatchMode -autotest_mode [-rhi=null]`。

### 4.5 维护风险

按子系统区分：

| 子系统 | 维护状况 |
| --- | --- |
| Atom | 基本由华为在做，四个账号自 2025-01 起合计约 109 次提交，约占 Atom 活动的一半 |
| SceneAPI | 约 5 名无机构归属的个人，只做维护 |
| USD | **bus factor 为零**。AWS 那位作者提完 7 行 PR 后无后续，跟进者是一名志愿者做构建清理 |

没有找到任何中国机器人或工业领域的 O3DE 使用者。

## 五、逐能力对比

| 能力 | Isaac Sim / Kit | Blender | O3DE |
| --- | --- | --- | --- |
| UsdPhysics 写入 | 原生 | 无，但有稳定的 Python 注入点可自建 | 无，且 Assimp 路线上不可能 |
| USD 层、引用、变体往返 | 原生 | 不处理、不导出（必须自建） | 只能当几何读，无合成 |
| USD 导出 | 原生 | 有（几何与材质） | **不存在，且被 RFC 明确排除** |
| 超写实渲染 | RTX 路径追踪 | Cycles，生产级路径追踪器 | **无路径追踪器** |
| 无头渲染 | 支持 | Cycles 支持，EEVEE 不支持 | GameLauncher 支持，编辑器未验证 |
| 真值通道 | 15 项契约全支持 | 大部分可映射，分割需合成 | **仅 RGB 与深度** |
| 域随机化 | Replicator（Isaac Lab 只用 2/8 随机化器） | BlenderProc 等成熟范式（GPL-3 需注意） | **不存在** |
| 交互式 prim 编辑 | 属性、图层、舞台面板 | 有编辑器，但无 USD 物理概念 | 有编辑器，但无 USD 概念 |
| 批量渲染自动化 | Replicator 编排器 | `--background --python`，成熟 | ScriptAutomation gem，含 pass 附件导出 |
| 碰撞体生成 | GUI 工具齐全 | 仅内置凸包，分解需外部工具 | 有碰撞体组件 |
| 许可 | NVIDIA 专有 | GPL，但本用法下基本无约束 | Apache-2.0 或 MIT，干净 |
| 相关子系统维护 | NVIDIA | 基金会，但 USD 创作"需要资金" | Atom 靠华为，**USD bus factor 为零** |

## 六、建议与自研范围

### 6.1 候选选择

**选 Blender，不选 O3DE。**

O3DE 在本次框架下的三项核心能力上全部为零或接近零：USD 合成读写、路径追踪、真值通道。
而且这三块恰好是维护最薄弱的部分，USD 方向甚至没有维护者。它唯一的优势是许可干净和
ScriptAutomation 这条批量渲染骨架，不足以抵消前述缺口。

Blender 的缺口集中在一处（UsdPhysics 与分层结构的导出），而且有稳定、纯 Python 的
注入点可以自建。渲染侧 Cycles 直接可用。许可在"美术创作加独立进程渲染加内部插件"的用法下
没有实质约束。

### 6.2 自研范围划定

按优先级，每项都对应前文的具体缺口：

1. **Blender USD 物理导出插件**（对应 2.4 的两条硬要求与 3.1）。
   基于 `USDHook.on_export` 与 `get_prim_map()`，在 stage 保存前写入 UsdPhysics 与
   PhysX/Newton/MuJoCo 的 applied schema 与命名空间属性；补写 Blender 无法导出的
   胶囊、球、圆柱基元；保持分层结构与 `"Physics"` 变体集。
   对称实现 `on_import` 钩子以保证非破坏性往返，关键是**原样保留未注册的 applied API token**。
   验收可直接用 Apache-2.0 的 `simready-validate`，它跑在标准 `usd-core` 上、不需要 Kit。
2. **纹理与材质域随机化的 MDP 事件项**（对应缺口一）。
   渲染器无关的 `VisualMaterialBatch` 契约已经存在且三个后端都实现了，
   只需补上驱动它的事件项。这是最小的改动、最高的杠杆，且可以合入上游。
3. **网格转 USD 的无 Kit 路径**（对应缺口二）。
   先评估 NVIDIA 自家 Apache-2.0 的 `urdf-usd-converter` 与 `mujoco-usd-converter`
   能覆盖多少，剩余的 OBJ/STL/FBX 转换用开源方案补齐。
4. **合成数据渲染管线**（对应 2.1 的分叉）。
   先做一周的 Cycles Hydra delegate 验证：若能对 USD stage 原地渲染，
   则不需要把资产导入 Blender，架构显著简化。不可行则退回 Blender 进程内渲染。
   域随机化采样器自己实现，不依赖 GPL-3 的 BlenderProc。
5. **交互式检视工具**（对应缺口三）。
   这一项是工效问题不是功能阻断，建议最后做，且按需裁剪：
   优先补碰撞体可视化与驱动调参，而非复刻整个属性面板。
6. **资产传输去 `omni.client` 化**（对应缺口五）。优先级最低，改动明确。

若目标**只是脱离 Kit 而非脱离 NVIDIA**，第 4 项可以直接用 OVRTX 跳过，
第 1 项仍然需要（因为那是创作侧），第 2、3 项不变。

### 6.3 两个可直接用的现成件

- **SimReady 现在是 Apache-2.0 的版本化规范，并有可 pip 安装的校验器 `simready-validate`**，
  跑在标准 `usd-core` 上，不需要 Kit。这是替代创作工具现成的机器可验收标准。
- **ScriptAutomation 的 `CapturePassAttachment`**（若最终仍要用 O3DE 做某些渲染）。

## 七、风险

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| Cycles Hydra delegate 不可用 | 需把资产导入 Blender，架构复杂化 | 先做一周验证再决定 |
| Blender 导出器未来变更破坏插件 | 维护负担 | `USDHook` API 自 4.0 起六版只有增量变化，风险中等；锁定 LTS |
| 渲染确定性无保证 | 数据集复现与回归测试受影响 | 需自建位级一致性测试 |
| 无 Cycles 与 RTX 的 sim-to-real 对比 | 画质是否够用无依据 | 方案中应包含一次对比实验 |
| OVRTX 为预发布专有二进制 | 作为过渡方案存在供应风险 | 不作为终点，按 2.1 的分叉推进 |
| AOUSD 物理工作组无产出 | 不能指望标准化解决 schema 问题 | 该组 2024-12-16 成立至今未发布任何内容，方案不应依赖它 |

## 八、待验证事项

可以快速证伪的：

- [ ] **Linux 上标准 Blender 应用内 `import pxr` 是否可用。** 这决定插件能否在进程内直接
      操作 stage。`bpy.utils.expose_bundled_modules()` 只在"Blender 作为 Python 模块"
      的语境下有文档。五分钟可验证，应最先做。
- [ ] Cycles Hydra delegate 能否加载并渲染一个 Isaac Lab 的 USD 资产（一周验证）。
- [ ] 用 `simready-validate` 跑一遍现有资产，确认基线通过率。

需要实测的：

- [ ] Cycles 在 512×512 与 1024×1024 下的单帧延迟与瓶颈构成，注明硬件。公开资料中没有。
- [ ] 三个渲染器的实测吞吐。仓库内有基准框架但未提交结果，文档只给了序数排名。
- [ ] OVRTX 的分割输出：**它的文档描述的是遗留的 `SemanticsAPI` 元数据，而 Isaac Lab 只写
      `UsdSemantics.LabelsAPI`，SimReady 的 SL.003 又要求新的那种。** 依赖它之前必须直接测试。

需要核对的冲突与未知：

- [ ] SimReady 的 RV.001 至 RV.011 规定三个变体集 `PhysX`/`Newton`/`MuJoCo`，
      取值 `Disabled`/`Enabled`；而 Isaac Lab 的转换器产出单个 `"Physics"` 变体集，
      取值 `physics`/`physx`/`mujoco`。意图相同，布局不同，需确定以哪个为准。
- [ ] `physx-usd-schemas` 的 wheel 元数据写 `LicenseRef-NVIDIA-Omniverse`，
      其源码却写 Apache-2.0。
- [ ] `docs/source/concepts/visualization.rst` 内部自相矛盾，
      表格（:959-1010）与正文（:392-396）对 Newton RTX 的能力描述不一致。
- [ ] Wandelbots 在 2025 年 Blender 大会的报告录像，是目前唯一已知的同类先例。
- [ ] O3DE 的 Qt 编辑器能否在带 GPU 的无头 Linux 服务器上运行（GameLauncher 可以）。
- [ ] 本轮核验未安装 `ovrtx`、`ovstage`、`ovphysx`、`isaacsim-asset-isolated`、
      `physx-usd-schemas`，相关结论均来自文档与包元数据，未实际运行。

## 九、参考来源

本仓库（第二轮核验，含行号）：

- `source/isaaclab/isaaclab/renderers/output_contract.py:19-39`（15 项输出契约）
- `source/isaaclab/isaaclab/renderers/base_renderer.py:30,58`（`visual_material_writer` 契约）
- `source/isaaclab_physx/isaaclab_physx/envs/mdp/events.py:465-471`（Replicator 不可用的抛错）
- `source/isaaclab_newton/isaaclab_newton/envs/mdp/events.py:278`（仅支持 color 通道）
- `source/isaaclab/isaaclab/sim/schemas/schemas.py`（无代码 schema 写入）
- `source/isaaclab/isaaclab/sim/converters/mesh_converter.py`（最后的 Kit 依赖）
- `source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:190-207`（分层资产格式）
- `source/isaaclab/isaaclab/utils/assets.py:66-96,125-132`（资产传输与中国镜像）
- `source/isaaclab/isaaclab/app/sim_launcher.py:199-205`（可视化后端选择器）
- `docs/source/concepts/renderers.rst:39-58`（OVRTX 推荐、Isaac RTX 遗留）
- `docs/source/concepts/visualization.rst:510-511`（Kit 独有的面板）
- `pyproject.toml:231`（`ovrtx==0.5.1.385782`）

Blender：

- Blender 5.2 LTS USD 导入导出文档 https://docs.blender.org/manual/en/latest/io/scenes/usd.html
- `bpy.types.USDHook` https://docs.blender.org/api/current/bpy.types.USDHook.html
- Blender 开发板块"核心 USD 创作" https://www.blender.org/lab/
- NVIDIA SimReady Blender 插件 https://docs.omniverse.nvidia.com/
- Cycles 渲染通道 https://docs.blender.org/manual/en/latest/render/layers/passes.html
- Cycles Hydra delegate https://github.com/blender/cycles
- BlenderProc2 https://github.com/DLR-RM/BlenderProc
- Blender 许可页 https://www.blender.org/about/license/

O3DE：

- PR #18361（USD 扩展名注册）https://github.com/o3de/o3de/pull/18361
- sig-simulation PR #92（USD in the Scene Pipeline RFC）https://github.com/o3de/sig-simulation/pull/92
- Issue #15848（Omniverse 兼容性）https://github.com/o3de/o3de/issues/15848
- sig-graphics-audio #152（光追材质重构，未关闭）https://github.com/o3de/sig-graphics-audio/issues/152
- 26.05.0 发布说明 https://docs.o3de.org/docs/release-notes/2605-0-release-notes/
- ScriptAutomation gem https://docs.o3de.org/docs/user-guide/gems/reference/

标准与规范：

- SimReady 规范与 `simready-validate` 校验器（Apache-2.0）
- OpenUSD UsdPhysics https://openusd.org/release/api/usd_physics_page_front.html
- AOUSD 物理工作组（2024-12-16 成立，至今无产出）https://aousd.org/
