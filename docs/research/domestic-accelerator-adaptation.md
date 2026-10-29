# Isaac Lab 国产芯片生态适配调研与方案

调研日期：2026-10-29
代码基线：本仓库 `main` 分支，commit `e1d7e9d4b`，VERSION 3.0.0

## 结论先行

推荐"分层解耦、分阶段推进"：先做厂商无关的设备抽象改造与 PyTorch/RL 层适配（昇腾最成熟），
物理层近期走 ARM CPU 上的原生 MuJoCo 兜底，中期在海光 DCU 上尝试 Warp 的 HIP 后端分支，
渲染层近期放弃或下沉到独立 GPU。不要把"一次性全栈移植到某一款国产芯片"当作可交付目标。

## 一、文档范围与可信度说明

本文回答的问题是：把 Isaac Lab 迁移到国产芯片生态上，技术上要做什么、代价多大、先做哪一步。

三点必须在阅读前说明，它们直接影响本文结论的强度：

1. **没有任何公开先例。** 中英文检索均未发现 Isaac Lab、Isaac Gym、Newton、MuJoCo-Warp
   或任何刚体物理引擎跑在国产加速卡上的公开报告。本文给出的是首次尝试的路径设计，不是
   对成熟做法的总结。
2. **厂商一手文档大量抓取失败。** 调研过程中 hiascend.com、gitcode.com/Ascend、寒武纪与
   昆仑芯开发者门户、部分海光与沐曦文档页多次返回错误。相关结论来自搜索摘要或二手中文
   技术媒体，文中逐条标注。
3. **厂商规格互相矛盾的情况普遍。** 同一款卡的 FP32 算力、显存带宽、功耗在不同来源相差
   一倍以上的例子不止一处。本文保留冲突、标注口径，不做静默取舍。

凡标注"未经独立验证"或"厂商口径"的数字，不得作为选型决策的唯一依据。

## 二、现状盘点：Isaac Lab 的 NVIDIA 依赖面

### 2.1 Warp 是数据层，不是加速插件

这是整个适配工作中最重要的事实。统计自本仓库 `source/` 目录，"非测试"指排除 `*/test/*`：

| 指标 | 数量 |
| --- | --- |
| `@wp.kernel` 内核定义（全部 / 非测试） | 593 / 523 |
| 含内核的非测试文件数 | 106 |
| 上述文件总行数（含非内核代码，为上界） | 约 48,900 |
| `wp.launch` 调用点 | 777 |
| `wp.array` 引用 | 约 6,953 |
| `wp.to_torch` / `wp.from_torch` | 637 / 488 |
| import warp 的非测试文件 | 316 |
| `ProxyArray` 引用 / 涉及文件 | 2,968 / 125 |

按包统计非测试内核数：`isaaclab_newton` 130、`isaaclab` 114、`isaaclab_ov` 92、
`isaaclab_experimental` 69、`isaaclab_physx` 68、`isaaclab_tasks_experimental` 59、
`isaaclab_tasks` 36。

关键在于 `ProxyArray`（`source/isaaclab/isaaclab/utils/warp/proxy_array.py`）。它是
Warp 优先的双访问数组包装，内部持有 `wp.array`，对外提供零拷贝的 torch 视图。仿真状态
本身就分配在 Warp 的内存与设备模型之上。因此**仅替换 PyTorch 后端无法完成移植，目标设备
上必须存在 Warp 或与 Warp API 兼容的运行时**。

内核的指令构成以逐元素与索引类为主（`wp.tid()` 555 处、原子操作 30 处、随机数 76 处），
真正难以在 Warp 之外重建的是 BVH 与网格查询（`mesh_query_*` 约 24 处、`wp.Mesh` 构造 9 处）
与 HashGrid（6 处）。Warp 的 tile API 在 Isaac Lab 自身代码中基本未使用，但 Newton 上游的
VBD 与 tiled camera 用到了。

### 2.2 CUDA 图与流交互

`NewtonCfg.use_cuda_graph` 默认为 `True`。`wp.ScopedCapture` 捕获点 29 处，其中 18 处集中在
`source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`。该文件还调用了 Warp 的
私有接口 `core.wp_cuda_context_check` 来清理图内存释放失败后的 CUDA 错误状态，属于强 CUDA
驱动耦合。`source/isaaclab/isaaclab/renderers/render_context.py` 通过
`wp.stream_from_torch(torch.cuda.current_stream(...))` 桥接 torch 与 Warp 的流，前提是两个
运行时共享同一 CUDA 上下文。

### 2.3 torch 侧耦合反而很薄

| 指标 | 数量 |
| --- | --- |
| `torch.cuda.*` 调用行 / 涉及文件 | 119 / 57 |
| 硬编码 `"cuda:0"` / `"cuda"` 字面量 | 约 115 |
| 自定义 CUDA C++ 扩展 | 0 |
| cupy / numba.cuda / triton 使用 | 0 |

`torch.cuda.*` 以 `is_available`、`device_count`、`synchronize`、`set_device` 为主，都有
`torch.accelerator` 或 PrivateUse1 的等价物。`torch.backends.cudnn` 仅出现在测试中。
`pynvml` 与 `nvidia-smi` 只在基准测试记录器、遥操作系统自检和 ray 脚本中出现。

多卡训练走 `torchrun`，Isaac Lab **自身从不调用** `init_process_group`，进程组由 rsl_rl /
rl_games / skrl 建立，后端选择（NCCL）在 RL 库侧。Isaac Lab 侧只有
`source/isaaclab/isaaclab/app/sim_launcher.py` 把设备固定为 `cuda:{local_rank}`。

### 2.4 依赖分类

| 类别 | 依赖项 | 处置 |
| --- | --- | --- |
| NVIDIA 专有二进制 | `isaacsim`、`ovphysx`、`ovrtx`、`ovstage`、`isaaccapture` | 必须放弃 |
| 开源但仅 CUDA/CPU 后端 | `warp-lang==1.17.0`(Apache-2.0)、`newton`、`mujoco-warp` | 需新后端或 fork |
| 可换厂商构建 | `torch==2.11.0`(cu130 源)、`torchvision`、`jax[cuda13]` | 换 wheel |
| 厂商中立 | `usd-exchange`、PyOpenGL、imgui-bundle、rerun、viser、gymnasium、RL 库 | 保留 |

三个物理后端中只有 Newton 在原理上可移植。`isaaclab_physx`（约 29,100 行）与
`isaaclab_ov`（约 28,600 行）依赖专有二进制，合计约 5.8 万行需要放弃。这两个包同时也是
Kit / Fabric 耦合最重的部分。

渲染同理：`IsaacRtxRenderer` 与 OVRTX 为 NVIDIA 专有，`NewtonWarpRenderer` 在 Warp 可移植的
前提下可移植，`newton_gl`（PyOpenGL）、`rerun`、`viser` 三个可视化后端厂商中立。

### 2.5 已存在的设备抽象缝

设备标识是一个普通字符串，流转路径为：`SimulationCfg.device` → `sim_launcher.py` 解析 →
`SimulationContext.__init__` 同时调用 `torch.cuda.set_device` 与 `wp.set_device` →
按 `class_type` / `launcher_type` 构造后端管理器。

障碍是约 20 处 `"cuda" in device` 或 `device.startswith("cuda")` 的判断，分布在
`envs/manager_based_env.py`、`envs/direct_rl_env.py`、`envs/direct_marl_env.py`、
`physics/physics_manager.py`、`utils/math.py`、`renderers/render_context.py`、
`terrains/trimesh/mesh_terrains.py`、`newton_manager.py` 等处。新设备名（如 `npu:0`、
`musa:0`）会落到 CPU / numpy 分支。例如 `PhysicsManager.get_backend()` 的实现就是
`"torch" if "cuda" in self._device else "numpy"`。

`source/isaaclab/isaaclab/utils/device.py` 目前只有两个函数，是引入加速器抽象的自然位置。
测试设备掩码 `ISAACLAB_TEST_DEVICES` 是位置式的（0=cpu，1=cuda:0），也需要扩展。

Newton 的 CPU 模式是受支持的（`--device cpu`，另有 `MJWarpSolverCfg.use_mujoco_cpu` 切换到
经典 MuJoCo C 实现），但**仓库内与上游文档均无 CPU 与 GPU 的吞吐对比数据**，上游只说明
CPU 用于开发调试。任何关于 CPU 兜底性能的判断都必须自行实测。

## 三、关键结论

### 结论一：移植的对象是 Warp，不是 PyTorch

见 2.1。仿真状态分配在 Warp 之上，523 个内核与 BVH / HashGrid 查询都是 Warp 原语。
PyTorch 换厂商构建是必要条件，远非充分条件。

### 结论二：NVIDIA 已明确拒绝在 Warp 上游接纳非 CUDA 后端

NVIDIA/warp 的 PR #1932（HIP/ROCm 适配层，AMD 方贡献者于 2026-09-09 提交）于 **2026-10-20
被关闭且未合入**，维护者说明 NVIDIA 出于上游测试、打包与支持承诺的考虑，"暂时把 HIP 后端
留在独立 fork 中"。

配套的 RFC 讨论 #1960 记录了该后端在 AMD gfx942（MI300X/MI325X）、ROCm 7.2 上的真实状态：
NVRTC 调用通过翻译头映射到 hipRTC；图支持只有部分可用（`wp.capture_while()` 靠静态展开模拟，
`wp.capture_if()` 在 HIP 上直接拒绝，`capture_save()` 不可用，部分图测试疑似受
`hipGraphAddMemFreeNode` 缺陷阻塞）；纹理与确定性模式被刻意排除；核心测试套件一次完整运行
为 8869 项测试、54 项失败、2977 项错误（其中大部分错误归因于一次捕获污染事件的连锁反应）。

对我们的含义：任何国产芯片的 Warp 移植都是"分支的分支"，且会继承上述已知缺陷。而
Isaac Lab 默认开启 CUDA 图、Newton/MJWarp 大量依赖图捕获，正好撞在最弱的一环上。
Warp 是 Apache-2.0，fork 在法律上没有障碍；代价在于 Isaac Lab 精确锁定 Warp 版本
（当前 1.17.0），Newton 与 MJWarp 又紧跟 Warp 特性，fork 必须长期跟进上游。

### 结论三：摩尔线程的选择反证了 MUSA 缺少运行时编译

摩尔线程于 2026-03-30 开源了 `mujoco_warp_musa`，这是目前**唯一**一个国产 GPU 上的
MuJoCo-Warp 物理实现，要求 MUSA SDK 4.3.4、MTT S4000/S5000，Apache-2.0。

但其实现方式是：用 CMake 构建一个**手写的、预编译的** MUSA C++ 共享库
`libmujoco_musa_shared.so`，通过 `import mujoco_warp.musa_api as mjw` 调用。它**不是**给
Warp 的 Python JIT 增加 MUSA 后端。仓库只有 3 次提交，README 未给出任何速度声明，也未提及
Isaac Lab、Newton 或 MJX；"相比 CPU 最高 40 倍"的说法来自媒体转述的厂商口径，未经独立验证。

推论是：如果 MUSA 存在可用的 NVRTC 等价物，移植 Warp 的代码生成层会是更划算的路线，而且能
免费带走整个 mujoco_warp。选择重写，说明这条路当时走不通。检索 `mtrtc` / `musaRTC` 等名称
未找到任何 MUSA 运行时编译库，`musart` 是普通运行时而非编译器。

### 结论四：没有任何公开案例

跨中英文检索，未发现 Isaac Lab、Isaac Gym、Newton、MuJoCo-Warp 在任何国产加速卡上运行的
报告。最接近的三个先例是：Isaac Gym 在 AMD 卡上以 CPU 仿真模式运行（论坛个例，RL 示例不可用）；
MJX-JAX 在 TPU 与 AMD 上运行；Genesis 自带 ROCm/HIP 后端——但 Genesis 拥有自己的内核编译器，
Isaac Lab 不拥有 Warp。

## 四、厂商评估

### 4.1 CUDA-like 国产 GPU

| 厂商 | 编程模型 | 运行时编译（NVRTC 等价物） | PyTorch | 集合通信 | 图形栈 | Warp 移植可行性 |
| --- | --- | --- | --- | --- | --- | --- |
| 海光 DCU | DTK，ROCm/HIP 衍生，DTK 26.10 称 HIP 对齐 ROCm 7.2 | hipRTC 在 ROCm 上游存在；**海光无一手文档确认随 DTK 发布** | 实测环境见到 torch 2.10.0(FlagOS, Beta) 与 2.4.1+dtk25041(生产镜像) | RCCL 2.22.3 / HCCL | 无产品文档，仅见招聘信息 | **最高**。但 warp_size=64 |
| 摩尔线程 | MUSA，MUSIFY 源码翻译，torchada 走 PrivateUse1 | **未找到**（`musart` 非编译器） | torch_musa 跟到 PyTorch 2.11（2026-08） | MCCL | **唯一有真实图形栈**：S4000 支持 DX/Vulkan/OpenGL；桌面驱动 OpenGL 4.6 + Vulkan 1.3 实验性；硬件光追在未发布的庐山架构 | 中。已有 MuJoCo-Warp 的 AOT 重写版 |
| 沐曦 | MXMACA，`cu-bridge`/`cucc` 可替代 nvcc 且无需 CUDA Toolkit | 未找到库级 RTC API | 2.4 / 2.6 / 2.8，Py 3.10/3.12，Ubuntu 20/22/24 + CentOS 8/9，x86+Arm（本组最宽） | 文档未获取 | 无信息 | 中。可考虑把 `cucc` 当子进程调用 |
| 天数智芯 | IXUCA，DayCompile（Clang）源码级重编译；NVIDIA cubin 不能直接运行 | 未找到 | 2.7.1+corex.4.4.0 与 2.4.1 两种说法冲突 | ixCCL | 无信息 | 低。warp_size=64，且有 float64 返回全零的缺陷报告 |
| 壁仞 | BIRENSUPA，CUDA 风格 C/C++ | 未找到 | 支持，无版本号 | 未找到 | 无信息 | 无可行路径。且不在 FlagTree 后端列表中 |
| 燧原 | TopsRider，**明确非 CUDA 兼容**，靠 `transfer_to_gcu` 钩子改写 `.cuda()` | 未找到 | torch-gcu 2.6.0 | 未确认 | 无信息 | 无可行路径 |

硬件数字（均需独立验证，口径冲突已标注）：

| 型号 | 显存 / 带宽 | FP32 | 备注 |
| --- | --- | --- | --- |
| 海光 BW1000 | 64GB HBM2e / 1.6 TB/s | 60 TFLOPS | 第三方来源，海光未正式公布；无 FP8 |
| 海光 K100-AI | 64GB HBM2e | **49 与 98 TFLOPS 两种说法** | 疑为 vector 与 matrix 口径差异 |
| 摩尔线程 S5000 | 80GB / 1.6 TB/s | **32 TFLOPS** | 1000 TFLOPS 头条数字是 FP8 稠密算力 |
| 沐曦 C500 | 64GB HBM2e | **15 至 54 TFLOPS 区间** | 取决于 vector/matrix 口径，无权威数据表 |
| 天数 BI-V150 | 64GB HBM2e | ≥48 TFLOPS | 带宽 1.2 TB/s 与 420 GB/s 两种说法冲突 |
| 壁仞 BR100 | 64GB HBM2E | 256 TFLOPS | Hot Chips 34 材料，TDP 550W；聚合站 300W 说法矛盾 |

注意对照：A100 FP32 为 19.5 TFLOPS，H100 为 67 TFLOPS。就 FP32 而言，这些卡并不处于劣势，
问题从来不在峰值算力，而在软件栈。

### 4.2 NPU 类

| 厂商 | 编程模型 | 运行时编译 | PyTorch | CUDA 图等价物 | 集合通信 | 图形栈 | 可行性 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 昇腾 | AscendC，**非 SIMT**，cube/vector 分块模型；另有 AscendNPU IR 开源并提供 Triton 转换接口 | triton-ascend 经 `bishengir-compile` 实现首次调用编译，说明路径存在；无公开的通用 RTC API | torch_npu 跟到 2.7.1 至 2.11，Py 3.12，Ubuntu 22.04 + openEuler 24.03 官方镜像 | `torch.npu.NPUGraph` / ACLGraph，vLLM-Ascend 生产验证；**仅静态形状，不支持动态形状与条件分支** | HCCL，128 卡级生产案例 | **无** | 最高，但仅限 PyTorch/RL 层 |
| 寒武纪 | BANG C | 未找到（无 `cnrtc` 类 API 证据） | Torch-MLU 走 PrivateUse1，含 Inductor 集成 | 未找到 | CNCL | 无 | 形式地位最好（PyTorch 基金会白金会员），但 MLU590/690 规格是公开文档真空 |
| 昆仑芯 | XPU，XTDK | 未找到 | XPyTorch 为二进制 `.run` 安装包，torch 2.5.1 / Py3.10 / Ubuntu 20.04 | 未找到 | BKCL（未获文档） | 无 | 最低。栈最封闭，PaddlePaddle 为一等框架 |

三点补充：

- **昇腾没有公开的 FP32 算力数据。** 910B/910C 只有 FP16/BF16 峰值（400 / 800 TFLOPS）。
  整个 910 系列唯一能查到的 FP32 数字是 910A 的 16 TFLOPS，且来自聚合站。Isaac Lab 物理
  计算以 FP32 为主，这意味着宣传算力对本负载的参考价值很低。
- **命名陷阱：** 上游 PyTorch 的 `torch.xpu` 指 Intel GPU，与昆仑芯的 XPU 无关。
- 昇腾上所有"强化学习"材料都是面向大模型的 RLHF/RLVR（verl、GRPO、DeepSeek-R1），
  不是环境步进式 RL。华为的 `cann-recipes-embodied-ai` 仓库是策略推理（Pi0 走 OM 格式），
  不含仿真器。一篇流传较广的 CSDN 文章声称该仓库含 MuJoCo/PyBullet 配方并给出 FPS 数字，
  但其示例代码与真实 MuJoCo API 不符，调研中已判定为不可靠并排除。

### 4.3 跨厂商中间层

- **FlagTree（智源）** 主分支后端包含 NVIDIA、AMD、x86 CPU、天数、摩尔线程、昆仑芯(KLX)、
  沐曦、海光(hcu)。**昇腾与寒武纪不在其中**，二者各自维护 triton-ascend 与 triton-linalg。
  更重要的是其仓库自述："各后端基于不同版本的 Triton，因此位于不同的受保护分支。"
  也就是说它不是一个支持 N 个后端的编译器，而是 N 个不同 Triton 版本的分支共处一仓。
  "一次编写、跨厂商运行"在当前结构下是愿景而非现实。
- **FlagGems** 是基于 Triton 的跨芯片算子库，注册进 PyTorch 的 ATen 分发器，算子数量在不同
  材料中为 180 余个与 207 个两种说法，已进入 PyTorch 基金会生态项目体系。
- **PaddlePaddle CustomDevice** 覆盖昇腾、寒武纪、燧原、天数、沐曦、海光、昆仑芯等，但它服务
  的是 PaddlePaddle，不解决我们的问题。
- **PJRT / OpenXLA：** 现有第三方插件只有 Intel GPU、Apple Metal、AWS Trainium。
  **昇腾、寒武纪、昆仑芯均无 PJRT 插件**，因此 MJX（纯 JAX 实现）这条绕行路目前是断的。

必须指出中间层的共同局限：它们都工作在**算子层**（替换 ATen 算子，或编译 Triton 内核），
而 Isaac Lab 的依赖在**内核层**——Warp 内核是用户编写的标量代码，含逐线程分支、BVH 遍历、
射线与三角形求交、原子散射写入。没有任何一层能接收 Warp 生成的 CUDA C 并在 NPU 上运行。
而且 Triton 自身是面向分块的模型，擅长 GEMM、逐元素、归约与注意力，对 BVH 遍历与稀疏约束
求解则力不从心。即便昇腾的 Triton 后端做到完美，也不能让 MJWarp 变得可移植。

## 五、路线对比与取舍

| 路线 | 适用厂商 | 优势 | 代价与风险 | 评价 |
| --- | --- | --- | --- | --- |
| A. fork Warp 增加厂商后端 | 海光优先，其次摩尔线程、沐曦 | 523 个内核与整个 Newton/MJWarp 原样复用，架构不变 | 需移植 Warp 的 native 运行时与内建函数（BVH、HashGrid、基数排序、原子、图、内存池）；继承 AMD 分支的已知缺陷；需长期跟进 Warp 版本 | **中期主攻方向**，但须先验证 hipRTC |
| B. 用 Triton 重写内核 | 昇腾、寒武纪及 FlagTree 覆盖的厂商 | 借力现有中间层，逐内核推进，可先易后难 | 只能覆盖逐元素与索引类内核；BVH、碰撞检测、约束求解无法用 Triton 表达；**MJWarp 仍然不可移植** | 作为路线 A 的补充，不能单独成立 |
| C. PJRT 插件 + MJX | 昇腾 | 内核由 XLA 生成，不需手写；一次投入解锁整个 JAX 生态 | 目前无任何厂商的 PJRT 插件；MJX-JAX 在接触密集场景弱于 MJWarp；与 Isaac Lab 的 torch + ProxyArray 架构不匹配 | 杠杆最高的长期投入，但属于另起炉灶 |
| D. CPU 物理 + NPU 训练 | 昇腾等全部 | 零移植成本，立刻可跑；MuJoCo 原生 C 在 CPU 上成熟，随鲲鹏核数扩展 | 吞吐未知，上游明确 CPU 仅供开发调试；大规模并行环境 RL 很可能不可行 | **近期唯一确定可交付的兜底** |

**推荐组合：D 打底 → 厂商无关改造 → A 在海光上验证 → 视结果决定是否投入 C。**

理由有三。其一，路线 D 不依赖任何未验证的厂商能力，能在最短时间内给出"能跑"的结论和真实的
性能基线，而这个基线正是后续所有决策所缺的数据。其二，厂商无关的设备抽象改造（见阶段一）
无论最终选哪条路都必须做，且可以合入上游，是沉没成本最低的一段。其三，路线 A 的成败几乎
完全取决于一个可在一天内验证的事实——hipRTC 是否随 DTK 发布，应当尽早证伪。

不推荐一开始就投入路线 C：它实质上是更换物理引擎，被移植的将是 Isaac Lab 的管理器与任务层，
而非其仿真核心，与"适配"的目标已经不同。

## 六、分阶段实施方案

### 阶段零：事实核验（1 至 2 周）

**目标**：用真机数据替换本文中标注为未验证的关键假设。

**工作内容**：取得海光 DCU 与昇腾各一台可用环境；核验 hipRTC 是否随 DTK 发布；
实测 Newton CPU 模式在目标 ARM 服务器上的吞吐；核验 torch_npu 在目标 CANN 版本上的实际可用性。

**验收标准**：产出一份真机核验报告，逐条回答第八节清单。

**风险**：拿不到硬件。应对：优先走厂商适配计划或公有云，调研中未找到这些芯片的公有云实例与
报价，可能需要厂商直接支持。

### 阶段一：厂商无关的设备抽象改造（4 至 6 周）

**目标**：让 Isaac Lab 在不引入任何厂商代码的前提下，能够接纳非 CUDA 设备名。

**工作内容**：
- 在 `source/isaaclab/isaaclab/utils/device.py` 建立加速器抽象，提供 `is_accelerator(device)`、
  `accelerator_module(device)` 等接口。
- 替换约 20 处 `"cuda" in device` 判断与约 115 处 `"cuda:0"` 字面量。
- 把 `torch.cuda.*` 调用改为 `torch.accelerator` 等价物，保留 CUDA 专有路径的显式分支。
- 扩展 `ISAACLAB_TEST_DEVICES` 掩码语义，使位置 1 表示"默认加速器"而非固定 `cuda:0`。
- 修正 `PhysicsManager.get_backend()` 这类以 `"cuda"` 判定张量后端的逻辑。

**验收标准**：在 NVIDIA 卡上全部现有测试不回归；引入一个假设的设备名后，设备解析路径不再
静默回落到 CPU/numpy。此阶段成果应具备向上游提 PR 的质量。

**风险**：改动面广但单点简单，主要风险是与上游冲突。应对：拆成多个小 PR，尽早与上游沟通。

### 阶段二：PyTorch 与 RL 层落地（4 至 8 周，与阶段一部分并行）

**目标**：RL 训练部分在昇腾上跑通。

**工作内容**：安装 torch_npu 构建；验证 rsl_rl / rl_games / skrl 的设备无关性；
确认三个 RL 库是否硬编码 `backend="nccl"`（调研未验证，需实测），需要时替换为 HCCL；
处理 `sim_launcher.py` 中 `cuda:{local_rank}` 的设备命名。

**验收标准**：以离线数据或 CPU 物理驱动，在昇腾上完成一次完整的 PPO 训练，损失曲线与
NVIDIA 基线一致。

**风险**：ACLGraph 仅支持静态形状。Isaac Lab 的步进循环在固定环境数下本就是定形的，
概念上兼容，但 MJWarp 内部的动态接触数量是真正的难点（该问题在 CUDA 上同样存在）。

### 阶段三：物理层兜底与基线测量（4 周）

**目标**：拿到 CPU 物理的真实吞吐，判断其是否构成可用的训练路径。

**工作内容**：在目标 ARM 服务器（鲲鹏/飞腾）上以 `--device cpu` 与 `use_mujoco_cpu` 两种
模式运行代表性任务；按核数扩展测量；与 NVIDIA GPU 基线对比。

**验收标准**：产出 steps/s 对比数据。仓库与上游都没有这类数据，这份测量本身就有价值。

**风险**：结果可能是"慢两到三个数量级，不可用于训练"。这同样是有价值的结论，它会把决策
推向路线 A 或 C。

### 阶段四：Warp 后端验证（8 至 16 周，取决于阶段零结论）

**目标**：在海光 DCU 上让 Warp 跑起来，哪怕功能不全。

**工作内容**：基于 AMD 的 HIP 分支，针对 DTK 调整；优先打通无图捕获的基础内核路径；
逐步处理 warp_size 64 带来的 warp 级原语问题；评估图捕获缺陷的影响面。

**验收标准**：Warp 自带测试套件的通过率；Isaac Lab 一个最简任务（如 Cartpole）在
`use_cuda_graph=False` 下完成步进。

**风险**：最高。AMD 分支本身在 MI 系列上仍有数十项失败与近三千项错误，移到 DTK 只会更差。
建议设置明确的中止条件，例如八周内若基础内核路径不能跑通则终止。

### 阶段五：渲染与多卡（视前序结果）

**目标**：确定渲染方案。

**工作内容**：近期明确放弃 RTX 与 tiled camera 类传感器，只支持状态观测；
`newton_gl`、`rerun`、`viser` 在有 OpenGL 驱动的主机上保留；
若需要图像观测，评估摩尔线程作为独立渲染卡的可行性（本组唯一有真实图形栈的厂商）。

**风险**：Isaac Sim 的渲染器通过 Vulkan/DirectX 枚举设备并要求 RTX 能力，无软件回退路径。
因此这条路只能靠放弃该功能或换卡解决，没有第三种办法。

## 七、风险登记

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| hipRTC 未随 DTK 发布 | 路线 A 直接不成立 | 阶段零优先验证；备选为把编译器当子进程调用 |
| CPU 物理吞吐不足以训练 | 近期无可交付的完整方案 | 以阶段三数据驱动决策，必要时转向路线 C |
| Warp fork 维护成本失控 | 长期技术债 | 设置中止条件；评估锁定 Warp 版本不跟进上游的可行性 |
| 渲染能力缺失 | 图像观测类任务全部不可用 | 明确范围，只承诺状态观测任务 |
| 厂商文档与实际不符 | 计划基于错误前提 | 一切以真机核验为准，不采信规格页 |
| HBM 短缺推高价格 | 2026-09 报道昇腾与寒武纪涨价 20% 至 50% | 采购侧提前锁定 |

## 八、待验证事项清单

硬件与工具链：

- [ ] **hipRTC 是否随 DTK 发布。** 无任何海光一手文档确认，ROCm 7.0 起 hipRTC 是需显式链接的
      独立库。真机执行 `ls $ROCM_PATH/lib/libhiprtc*` 即可证实或证伪。本项为路线 A 的前置条件。
- [ ] 除海光经 ROCm 继承外，**没有任何厂商公开文档描述 NVRTC 等价物**。需向摩尔线程、沐曦、
      昇腾逐一确认是否存在可从源码字符串运行时编译的公开 API。
- [ ] 海光 DTK 上 PyTorch 的最高可用版本（实测见到 2.10.0 为 Beta、2.4.1 为生产镜像，
      无一手来源确认 2.5/2.7/2.11）。
- [ ] 昇腾 910B/910C 的 FP32 实际算力，以及不规则分支型 FP32 负载的实测性能。
      公开资料完全没有这个数据。
- [ ] triton-ascend 的基线 Triton 版本、CANN 与 torch_npu 最低要求、逐算子支持矩阵，
      特别是 atomics 的数据类型覆盖与内存空间支持。调研中相关页面均抓取失败。
- [ ] 寒武纪 MLU590/690 的规格。这是公开文档真空而非检索失败，需走厂商渠道。
- [ ] Warp 的 AMD HIP 分支与上游的版本对应关系，以及 CUDA 图与内存池在其上的实际可用程度。

代码与框架：

- [ ] rsl_rl 5.5.1、rl_games、skrl 是否硬编码 `backend="nccl"`（调研未验证，本地未安装）。
- [ ] Newton 与 mujoco_warp 两个包自身的内核规模（本地虚拟环境未同步，未能统计）。
      这是路线 A 工作量的主要部分，当前无数。
- [ ] `newton_gl` 是否使用 CUDA-GL 互操作（如 `wp.RegisteredGLBuffer`）。
- [ ] `isaaccapture`、`omniverseclient`、`usd-optimize` 是否存在 CUDA 链接。
- [ ] `isaaclab_tasks`（657 个文件）与 `isaaclab_teleop` 中是否还有未统计到的 `"cuda"` 字面量。

采购与合规：

- [ ] 这些芯片的公有云实例与报价。调研在阿里云、华为云、移动云、天翼云均未找到。
- [ ] 2025-11 关于国家资助数据中心须使用国产 AI 芯片的指引，其层级与现行状态。
      所有来源均为对路透报道的转述，未能获取一手文件。
- [ ] 信创认证是否覆盖仿真类软件。已查到的目录覆盖操作系统、CPU、数据库、办公软件，
      未发现仿真工具链类别。

## 九、参考来源

Warp 与上游态度：

- NVIDIA/warp PR #1932 "Add WIP HIP/ROCm shims"（2026-09-09 提交，2026-10-20 关闭未合入）
  https://github.com/NVIDIA/warp/pull/1932
- NVIDIA/warp discussion #1960 "RFC: A HIP backend for AMD GPUs"
  https://github.com/NVIDIA/warp/discussions/1960
- ROCm/warp（AMD 维护的 HIP 移植分支，Apache-2.0）https://github.com/ROCm/warp
- Warp 代码生成文档 https://nvidia.github.io/warp/codegen.html
- NVIDIA/warp 主仓库与许可 https://github.com/NVIDIA/warp

物理引擎：

- Newton 安装指南（CPU 模式限制）https://newton-physics.github.io/newton/latest/guide/installation.html
- google-deepmind/mujoco_warp https://github.com/google-deepmind/mujoco_warp
- MuJoCo MJX 文档（MJX-JAX 与 MJX-Warp 的后端差异）https://mujoco.readthedocs.io/en/stable/mjx.html
- MooreThreads/mujoco_warp_musa https://github.com/MooreThreads/mujoco_warp_musa

厂商：

- 海光光合开发者社区 DTK https://developer.sourcefind.cn/dtk
- AMD HIP RTC 文档（ROCm 7.0 起 hipRTC 为独立库）
  https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_rtc.html
- MooreThreads/torch_musa releases https://github.com/MooreThreads/torch_musa/releases
- MooreThreads/torchada https://github.com/MooreThreads/torchada
- MUSA SDK 5.1.0 安装指南 https://docs.mthreads.com/musa-sdk/version-5.1.0/install_guide/
- 沐曦 mcPyTorch 用户指南
  https://developer.metax-tech.com/api/client/document/preview/1066/split_files/%E6%A6%82%E8%BF%B0.html
- torch-npu PyPI（PyTorch 与 Python 版本矩阵）https://www.pypi.org/project/torch-npu
- vLLM Ascend 图模式指南（ACLGraph）
  https://docs.vllm.ai/projects/ascend/en/main/_sources/user_guide/feature_guide/graph_mode.md
- CANN Ascend C 算子开发指南
  https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html
- triton-ascend 文档 https://triton-ascend.readthedocs.io/zh-cn/latest/
- AscendNPU IR 的 Triton 转换接口
  https://ascendnpu-ir.gitcode.com/en/sources/developer_guide/conversion/triton_interface.html
- PyTorch 基金会关于寒武纪的公告（Torch-MLU 走 PrivateUse1）
  https://pytorch.org/blog/cambricon-joins-the-pytorch-foundation-as-a-platinum-member/

中间层：

- flagos-ai/FlagTree（后端列表与各后端分支独立的说明）https://github.com/flagos-ai/FlagTree
- PyTorch 加速器集成文档（PrivateUse1 / OpenReg）https://docs.pytorch.org/docs/main/accelerator/hooks.html
- PyTorch 自定义后端分布式集成 https://docs.pytorch.org/docs/main/accelerator/distributed.html
- OpenXLA PJRT 集成 https://openxla.org/xla/pjrt/pjrt_integration
- PaddlePaddle 自定义设备文档
  https://www.paddlepaddle.org.cn/documentation/docs/en/dev_guides/custom_device_docs/index_en.html

其他：

- Genesis（自带 ROCm/HIP 后端的对照案例）https://genesis-world.readthedocs.io/
- Awkward Array 的 ROCm/HIP 移植研究（不规则内核 5 至 10 倍性能损失）https://arxiv.org/html/2609.24628v1
- AMD HIP 硬件实现说明（CDNA 波前 64 宽）
  https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/understand/hardware_implementation.html
