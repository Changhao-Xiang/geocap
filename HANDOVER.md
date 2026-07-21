# GeoCap 项目交接文档

> **日期**: 2026-06-02
> **项目**: geocap
> **负责人**: 岳逸帆

---

## 目录

1. [模块一：eval/ — 评估模块](#1-模块一eval--评估模块)
2. [模块二：data/draw/ — 图形绘制模块](#2-模块二datadraw--图形绘制模块)
3. [模块三：stage3/get_angles_and_slopes.py — 形态学分析](#3-模块三stage3get_angles_and_slopepy--形态学分析)
4. [模块四：stage2_diffusion/ — 扩散纹理化管道](#4-模块四stage2_diffusion--扩散纹理化管道)
5. [已知问题与注意事项](#5-已知问题与注意事项)
6. [数据集调优](#6-数据集调优)
7. [InternVL 推理](#7-internvl-推理)

---

## 1. 模块一：eval/ — 评估模块

负责部分：**`eval.py`** + 两份 system prompt。

### 1.1 涉及文件

```
eval/
├── eval.py                          # ★ 主评估入口
└── prompts/
    ├── extract_system_prompt.txt    # 特征提取提示
    └── eval_system_prompt.txt       # 评分提示
```

### 1.2 eval.py 评估流程

三阶段：**提取 → 评分 → 规则覆盖**。

```
输入: JSON 数组，每项 {img, output, reference}
                │
    ┌───────────┴───────────┐
    ▼                       ▼
output 描述             reference 描述
    │                       │
    ▼                       ▼
Evaluater.extract()    Evaluater.extract()
(LLM + extract_system   (LLM + extract_system
_prompt.txt)            _prompt.txt)
    │                       │
    ▼                       ▼
19 个结构化 JSON 特征   19 个结构化 JSON 特征
    │                       │
    └───────────┬───────────┘
                ▼
      Evaluater.evaluate()
      (LLM + eval_system_prompt.txt)
                │
                ▼
   逐特征评分 {reason, rating: 0-10}
                │
                ▼
      rule_based_eval()
      8 个特征用确定性规则覆盖 LLM 评分
                │
                ▼
   输出: detailed_score_list.json
         → cal_score.py 统计 → feature_statistics.csv
                              → species_statistics.csv
```

**`Evaluater` 类关键方法**：

| 方法 | 功能 |
|---|---|
| `load_llm_generator()` | 从 `common.llm.generator_mapping` 初始化 LLM，加载 extract prompt |
| `reload_eval_mode()` | 切换到 eval prompt（因为 extract 和 eval 用不同 prompt） |
| `extract(entry_batch)` | 对一批条目调用 LLM，将自由文本转为 19 个特征的 JSON。解析失败时记录错误 |
| `evaluate(outputs, references)` | 构建逐特征比较字符串（跳过 rule-based 特征），调用 LLM 评分 |
| `make_eval_prompt(eval_pair)` | 将一对 output/reference 特征格式化为比较文本 |

**`rule_based_eval()`** — 8 个特征不用 LLM，改为确定性规则覆盖：

| 特征 | 规则 |
|---|---|
| `length`, `width`, `ratio`, `number_of_volutions`, `proloculus`, `tunnel_angles` | 数值比较（范围匹配、相对误差） |
| `axis_shape` | 弯曲类别匹配（0 或 10） |
| `chomata` | 三维度综合（发育程度/高度/宽度） |

**配置**（来自 `common.args.fossil_eval_args`）：

| 参数 | 说明 |
|---|---|
| `eval_origin_file` | 输入 JSON，每项 `{img, output, reference}` |
| `eval_reference_file` | 缓存参考特征 JSON（避免重复提取） |
| `eval_result_dir` | 输出目录 |
| `eval_llm` | LLM 模型名 |
| `eval_start_pos` / `eval_end_pos` | 切片范围 |
| `extract_only` | 仅提取不评分 |

### 1.3 extract_system_prompt.txt 详解

**用途**：将一段自由文本化石描述 → 结构化 JSON。eval.py 对 output 和 reference 各调用一次。

**结构**（117 行）：

```
Task Description  ── 任务说明：从化石描述中提取特征，输出纯 JSON
    │
Requirements  ── 提取 19 个特征（case-sensitive keys）
    │            + 6 条提取准则（不重复提取、数字转换、冲突保留精确值、
    │              保留原文、单位换算、同义词映射）
    │            + 关键禁令（不加注释、不编造、不修改测量值、不重排 key 顺序）
    │
Output Format  ── 纯 JSON，缺失字段用空字符串 ""
    │
Example 1  ── 一段长自由文本 → 完整 JSON（19 个字段）
    │
Example 2  ── 带 XML 标签的文本 → 完整 JSON（19 个字段）
    │
{input} 占位符  ── 运行时替换为实际描述
```

**关键设计点**：

1. **19 个特征 key 固定顺序**，不允许重排。`{input}` 占位符在末尾，运行时替换
2. **同义词映射**写在 prompt 里而非代码里：proloculus=initial chamber, spirotheca=wall, whorls=volutions
3. **单位换算**由 LLM 自行处理（如 1 mm = 1000 microns），`utils.py` 中也有代码层面的换算作为兜底
4. **冲突处理策略**："保留数据范围更小的那个"——将取舍权交给 LLM
5. **两个示例覆盖不同输入格式**：Example 1 是纯自然语言，Example 2 带 XML 标签

### 1.4 eval_system_prompt.txt 详解

**用途**：逐特征比较 output 和 reference 的结构化特征，输出 0-10 评分 + 推理理由。

**结构**（160 行）：

```
Task Description  ── 比较生成描述 vs 参考描述，逐特征 0-10 评分
    │
Instructions  ── 独立客观分析每个特征
    │
Response Format  ── {"category_name": {"reason": "...", "rating": N}}
    │
Scoring Guidelines  ── 5 类评判标准：
    │   · 数值与范围：精确匹配→10，范围包含→10，否则按差异程度
    │   · 描述内容：语义相似度 + 完整性
    │   · 空字段：output 或 reference 为空 → 0
    │   · 一致性：关注语义而非字面
    │
Example 1  ── 12 个特征的逐项比较（含数值、描述、矛盾等各种情况）
    │         每个特征给出 reason + rating
    │
Example 2  ── 11 个特征的逐项比较（更多边缘情况）
    │
{input} 占位符  ── 运行时替换
```

**关键设计点**：

1. **评分是 0-10 的整数**（prompt 中明确 `integers only`）
2. **两个示例覆盖了评分指南的全部 5 种情况**：精确数值匹配、语义等价、部分匹配、矛盾、空字段
3. **评分指南区分了数值型和描述型**：数值型看范围和误差，描述型看语义相似度和完整性
4. **不评 rule-based 特征**：`eval.py:make_eval_prompt()` 在构建比较文本时会跳过那 8 个特征，因此 eval_system_prompt 不需要处理它们
5. **两个示例的输入格式一致**：都是 `特征名\nGenerated:...\nReference:...` 的格式，由 `make_eval_prompt()` 生成

---

## 2. 模块二：data/draw/ — 图形绘制模块

### 2.1 目录结构

```
data/draw/
├── draw.py                      # ★ 主入口：选择后端并批量生成图形
├── pil_backend.py               # PIL 渲染后端（简单版）
├── plt_backend.py               # Matplotlib 渲染后端（丰富版）
├── diffusion_backend_new.py     # 扩散模型渲染后端（MimicBrush 纹理迁移）
└── diffusion/
    ├── cpu_get_sim.py           # CPU 参考图像搜索
    ├── gpu_get_sim.py           # GPU 参考图像搜索
    ├── shape_filter.py          # 形状匹配库（OpenCV 轮廓匹配）
    ├── run_gradio3_demo.py      # MimicBrush 扩散推理 + Gradio 演示
    ├── cpu_test.sh              # CPU 搜索测试脚本
    ├── gpu_get_sim_test.sh      # GPU 搜索测试脚本
    └── gpu_test.sh              # 4-GPU 并行扩散处理脚本
```

### 2.2 三阶段绘制管道

#### 阶段 1：几何图形绘制

```
JSON 规则 → draw.py → pil_backend.py 或 plt_backend.py → 基础图形图像
```

| 后端 | 支持的形状 | 特效 |
|---|---|---|
| **PIL** (`pil_backend.py`) | polygon, line, ray, segment, ellipse, spiral | 高斯噪声、白线、重绘、灰度化、stylish 渐变 |
| **Matplotlib** (`plt_backend.py`) | 以上 + spindle, fusiform_1, fusiform_2, curves, sector, star | xkcd 风格、渐变线条、Perlin 噪声 |

**入口**：`draw.py:draw_figure(rules, path, backend, random_seed, randomize)`

#### 阶段 1.5：参考图像匹配

```
基础形状 → cpu_get_sim.py / gpu_get_sim.py → shape_filter.py → 最相似参考图像
```

使用 OpenCV 的 ShapeContextDistance 或 HausdorffDistance 在参考图像池中搜索最匹配的真实化石图像。

#### 阶段 2：扩散纹理化

```
基础形状 + 参考图像 → diffusion_backend_new.py → MimicBrush → 逼真纹理化石图像
```

管道：`基础形状 → 掩码生成 → 扩散纹理迁移 → 极点掩码 → 二次扩散 → 隔膜叠加 → 混合 → 保存`

### 2.3 关键类与函数

| 名称 | 文件 | 功能 |
|---|---|---|
| `draw_figure()` | `draw.py` | 中央调度：选择后端，渲染并保存图形 |
| `process_single()` | `draw.py` | 单样本并行包装器 |
| `Figure` | `pil_backend.py` | PIL 渲染器：`draw()` + `save_release()` |
| `Figure` | `plt_backend.py` | Matplotlib 渲染器：更丰富的形状 + Perlin 噪声 |
| `Figure_Engine` | `diffusion_backend_new.py` | 简化的 Matplotlib 引擎（无随机化、无 xkcd） |
| `generate_one_img()` | `diffusion_backend_new.py` | 完整扩散管道 |
| `getMostSimilarImages()` | `shape_filter.py` | 形状相似度搜索（ShapeContext/Hausdorff 距离） |
| `inference_single_image()` | `run_gradio3_demo.py` | MimicBrush 单图推理 |

### 2.4 支持的形状类型

在 `plt_backend.py` 中实现（`__handle()` 分发）：

```
polygon, line, ray, segment, ellipse, spiral,
spindle, fusiform_1, fusiform_2, curves, curve, sector, star
```

纺锤形（fusiform）是化石图像的核心形状。`fusiform_1` 和 `fusiform_2` 使用正弦调制来模拟生物形态。

### 2.5 使用方式

```bash
# 通过统一入口
./run -m data.draw.draw --rules_path dataset/rules.json --figure_dir dataset/figures --backend plt
```

### 2.6 配置参数

绘制配置通过 `common.args.DrawArgs` 控制：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `backend` | 渲染后端 | `"plt"` |
| `random_seed` | 随机种子 | — |
| `randomize` | 是否随机化 | — |
| `size` | 图像尺寸 | — |
| `dpi` | 分辨率 | — |
| `line_weight` | 线宽 | — |
| `line_style` | 线条样式 | — |
| `color` | 颜色 | — |
| `Gaussian_mean/var` | 高斯噪声参数 | — |
| `Perlin_lattice/bias/power` | Perlin 噪声参数 | — |
| `stylish` | 风格化渲染 | — |
| `inline_noise` | 内联噪声 | — |
| `n_white_line` | 白线数量 | — |

---

## 3. 模块三：stage3/get_angles_and_slopes.py — 形态学分析

### 3.1 模块定位

这是第三阶段化石描述管道的**核心形态学分析模块**。从化石 RGBA 图像中提取几何角度和凸性分数，用于分类赤道形状、极点形状和侧坡形状。

### 3.2 在项目中被广泛引用

该模块在 **4 个不同模块** 中被导入使用：

| 调用者 | 用途 |
|---|---|
| `stage3/generate_dataset.py` | 数据集生成：提取赤道/极点/坡度形状分类 |
| `stage3/distribution_figures.py` | 生成角度和凸性分布图 |
| `eval/eval_vis_tools.py` | 评估管道中的特征提取 |
| `visualize_consistency/angles_slopes_statistics_img.py` | 一致性可视化统计 |

### 3.3 函数详解

共 **18 个函数**，分类如下：

#### 轮廓处理

| 函数 | 行号 | 功能 |
|---|---|---|
| `find_contour(img)` | 10-25 | 从 RGBA alpha 通道提取最大外轮廓 |
| `find_extremes(contour)` | 41-55 | 找 4 个极值点（最左/右/上/下） |
| `find_extrema_points(points)` | 58-88 | 通过一阶导数符号变化检测局部极值点 |
| `classify_points(contour, mid_x, n, center)` | 28-38 | 按 x 坐标选最近 n 个点，分为上下半部分 |

#### 角度计算

| 函数 | 行号 | 功能 |
|---|---|---|
| `get_start(extremas, mid_x, contour)` | 91-122 | 为角度计算找鲁棒起始点 |
| `recheck(start, neighbor_points, contour)` | 125-135 | 验证起始点未漂移，必要时扩展邻域 |
| `get_angle(start, side, neighbor_points)` | 138-237 | **核心**：对称窗口 + 向量投影 + 反余弦 → 角度 |

#### 曲率/凸性分析

| 函数 | 行号 | 功能 |
|---|---|---|
| `separate_slopes(slope)` | 240-301 | 按极角将轮廓点分为 4 个象限区域 |
| `deriv2(region)` | 304-310 | lexsort 排序后委托给 `check_curvature` |
| `check_curvature(x, y)` | 313-364 | 去趋势 → 旋转坐标 → 3 次多项式拟合 → 系数 |
| `conv_scoring(coeff, side, t3_threshold=0.5)` | 367-380 | 从多项式系数计算单一凸性标量分数 |
| `extend(exclusion, contour)` | 383-397 | 沿轮廓扩展排除点集 |
| `get_convexity(contour, extremes, ...)` | 400-425 | 整体凸性分析：移除极值邻域 → 分区域 → 曲率评分 |

#### 主入口

| 函数 | 行号 | 功能 |
|---|---|---|
| `get_angles_and_slope(path, debug=False)` | 428-541 | **主入口**：完整管道，返回 8 个浮点数 |
| `main()` | 544-597 | 独立测试：对硬编码图像运行分析 |

### 3.4 处理流程

```
输入：RGBA PNG 图像路径
    │
    ▼
find_contour()  ──► 从 alpha 通道提取最大轮廓
    │
    ▼
find_extremes() ──► 4 个极值点（最左/右/上/下）
    │
    ▼
classify_points() ──► 上下半部分点集
    │
    ▼
find_extrema_points() + get_start() ──► 鲁棒起始点
    │
    ▼
get_angle() × 4  ──► left_angle, right_angle, upper_angle, lower_angle
    │
    ▼
extend() + get_convexity() ──► 4 个凸性分数
    │
    ▼
输出：(left_angle, right_angle, upper_angle, lower_angle,
       convex_score[0], convex_score[1], convex_score[2], convex_score[3])
```

### 3.5 输出分类阈值

这些阈值在 `generate_dataset.py` 和 `eval_vis_tools.py` 中使用：

| 特征 | 计算方式 | 分类 |
|---|---|---|
| **赤道形状** | `(upper_angle + lower_angle) / 2` | `< 2.95` 凸形 · `2.95-3.15` 直线形 · `≥ 3.15` 凹形 |
| **极点形状** | `(left_angle + right_angle) / 2` | `< 2.35` 尖锐 · `≥ 2.35` 钝圆 |
| **侧坡** | `sum(4 凸性分数)` | `< -2.3` 凸形 · `-2.3 ~ -0.8` 直线形 · `≥ -0.8` 凹形 |

### 3.6 输入要求

- **RGBA PNG 图像**，alpha 通道（第 4 通道）包含化石轮廓
- 如果是 numpy 数组输入，会被 resize 到 `(128, 128)`（`size_` 常量，第 7 行）

### 3.7 关键算法细节

**角度计算**（`get_angle`）：
1. 以起始点为中心构建对称窗口（每侧最多 15 个点）
2. 计算从起始点到窗口边缘的向量
3. 将向量投影到方向向量上
4. 使用反余弦计算两个角度
5. 对各对角度取平均 → 返回

**凸性分析**（`check_curvature`）：
1. 归一化 (x, y) 坐标
2. 拟合直线找趋势
3. 旋转坐标系使该直线成为新 x 轴
4. 在旋转坐标上拟合 3 次多项式
5. 返回 4 个系数

**凸性评分**（`conv_scoring`）：
- 上半部分：`coeff[1] * 3`（线性系数 × 正向）
- 下半部分：`coeff[1] * -3`（线性系数 × 反向）

---

## 4. 模块四：stage2_diffusion/ — 扩散纹理化管道

负责部分：**`stage2_diffusion/`** 全部三个文件。

### 4.1 目录结构

```
stage2_diffusion/
├── figure.py          # 图形渲染引擎（Matplotlib）
├── diffuser.py        # MimicBrush 扩散模型接口
├── process.py         # ★ 主编排入口
└── README_IN_FOSCAP_PROJECT.md
```

### 4.2 模块定位

该模块将**几何图形 → 逼真纹理化石图像**。从 JSON 规则渲染基础形状，利用 MimicBrush 扩散模型将参考图像纹理迁移到合成形状上，最终输出灰度化石图像。

**该模块不能在本项目（geocap）中运行**，必须整体复制到外部 MimicBrush 项目根目录下执行。

### 4.3 处理流程（process.py）

```
JSON 规则 + best_match 参考列表
    │
    ▼
figure.py: Figure_Engine
    │  渲染 fusiform_1, fusiform_2, ellipse, curves
    │  记录 volution 角度数据（用于后续掩码生成）
    ▼
基础 RGBA 图像 + volution_memory
    │
    ├──► generate_basic_mask() ──► axial_main 掩码
    │         │
    │         ▼
    │    diffuser.py: diffuse(mode="axial_main")
    │         │  随机选取 ref_axial_*.png 参考图
    │         ▼
    │    轴向主体纹理
    │
    ├──► generate_basic_mask() ──► axial_ext 掩码
    │         │
    │         ▼
    │    diffuser.py: diffuse(mode="axial_ext")
    │         │  随机选取 ref_axial_ext_*.png 参考图
    │         ▼
    │    轴向延伸纹理 → 与 axial_main 合成
    │
    ├──► redraw_basic_shapes()  重绘形状轮廓线
    │
    ├──► generate_basic_mask() ──► poles 掩码
    │         │
    │         ▼
    │    diffuser.py: diffuse(mode="poles")
    │         │  使用 best_match 中最匹配的真实化石图像
    │         │  经 processRefImage() 中线对称处理
    │         ▼
    │    极点纹理 → 合成
    │
    ├──► generate_septa()  绘制隔膜（Bezier 曲线 + 射线填充）
    │         │
    │         ▼
    │    隔膜叠加（alpha 混合）
    │
    ▼
灰度化 → 保存 {keyword}/{img_path}.jpg
```

### 4.4 输入数据来源

**JSON 规则**（`--rules`）由 `data/rule/generate.py` 生成。该模块按化石形态学规则生成结构化的几何描述 JSON，定义每个样本包含的形状类型（fusiform_1、fusiform_2、ellipse、curves）、位置、大小、volution 参数、隔膜（septa）和轴向填充（axial_filling）等。

**best_match**（`--best_match`）由形状匹配管道生成：

```
data/draw/diffusion/cpu_get_sim.py 或 gpu_get_sim.py
    │
    ▼
data/draw/diffusion/shape_filter.py
    │  getMostSimilarImages()
    │  OpenCV ShapeContextDistance / HausdorffDistance
    ▼
参考图像池（pics_12xx/）中搜索最相似的真实化石图像
    │
    ▼
best_match.txt（每行一个参考图像路径）
```

流程：将规则渲染为基础形状 → 在参考图像池中按轮廓相似度匹配 → 输出每个样本最匹配的真实化石图像路径，供扩散阶段作为极点纹理参考。

### 4.5 关键文件详解

#### figure.py — `Figure_Engine` 类

Matplotlib 渲染引擎，12.8×12.8 英寸 100 DPI → 1280×1280 像素画布。

| 方法 | 功能 |
|---|---|
| `draw(shape)` | 分发：fusiform_1 / fusiform_2 / ellipse / curves |
| `__handle_fusiform_1()` | 抛物线纺锤形 + 正弦调制 |
| `__handle_fusiform_2()` | 幂律纺锤形 + 左右不对称 |
| `__keep_memory(index, x, y)` | 按极角存储 volution 点数据（供 mask 生成使用） |
| `transfer_to_cv2()` | Matplotlib → BytesIO → OpenCV BGRA 数组 |

支持 xkcd 风格随机化。颜色随机生成（范围 [50,256) 每通道）。`fill_mode` 控制透明度：`"no"`/`"border"` → 透明，`"white"` → 白色，`"black"` → 黑色。

#### diffuser.py — MimicBrush 接口

模块加载时即初始化完整 MimicBrush 管道：

```
DDIMScheduler → VAE → UNet2DConditionModel(in_channels=13)
    → MimicBrushPipeline → DepthGuider → ReferenceNet
    → MimicBrush_RefNet（组合以上全部）
    → DepthAnything（深度估计）
```

| 函数 | 功能 |
|---|---|
| `diffuse(img, mask, best_ref_poles, ref_path, num_refs, mode)` | **主入口**：按 mode 选取参考图 → MimicBrush 推理 → 合成 |
| `infer_single(ref, target, mask, ...)` | 核心推理：正方形填充 → 深度提取 → `mimicbrush_model.generate()` → 裁剪还原 |
| `processRefImage(ref_image)` | 极点参考图处理：画白色中轴线 → 双侧对称效果 |
| `pad_img_to_square()` | 非正方形图像白边（mask 黑边）填充 |
| `crop_padding_and_resize()` | 逆操作：恢复原始宽高比 |

三种扩散模式：

| mode | 参考图来源 | 说明 |
|---|---|---|
| `"axial_main"` | `ref_path/ref_axial_{index}.png` 随机选取 | 轴向主体纹理 |
| `"axial_ext"` | `ref_path/ref_axial_ext_{index}.png` 随机选取 | 轴向延伸纹理 |
| `"poles"` | `best_ref_poles["best_ref"]` 最匹配图 | 极点纹理（经中线对称处理） |

#### process.py — 主编排

| 函数 | 功能 |
|---|---|
| `generate_basic_shape(shapes, ni)` | 渲染所有形状 → 图像 + volution 数据 |
| `generate_basic_shape_separately(shapes, ni)` | 先渲染最大 volution（用于参考），再渲染全部 |
| `generate_basic_mask(volution_memory, filling, mode)` | 从 volution 角度数据生成扩散掩码 |
| `generate_septa(septas)` | 绘制隔膜：Bezier 曲线 + 射线填充封闭区域 |
| `redraw_basic_shapes(dif_pic, shapes)` | 在扩散图上重绘形状轮廓 |
| `separate_axial_filling(axial_filling)` | 拆分轴向填充为 main 和 extension 两组 |
| `generate_one_img(idx, sample, ...)` | **完整单图生成**：基础形状 → 3 阶段扩散 → 隔膜 → 灰度 → 保存 |
| `main()` | CLI 入口 |

### 4.6 使用方式

```bash
# 需在 MimicBrush 项目根目录下运行
CUDA_VISIBLE_DEVICES=1 python stage2_diffusion/process.py \
  --rules <rules.json> \
  --best_match <best_match.txt> \
  --kwd <output_dir> \
  --start_pos 0 --end_pos 100
```

两阶段执行流程：
1. **最佳匹配搜索**：`sh gpu_get_sim_test.sh <keyword>` → 生成 `best_match.txt`
2. **扩散生成**：`sh gpu_test.sh <keyword>` → 生成图像到 `pics/`

### 4.7 与 data/draw/ 的关系

`stage2_diffusion/` 是基于 `data/draw/` 模块独立构建的扩散纹理化管道。它复用了 `data/draw/` 中的几何渲染能力，但作为独立模块运行于外部 MimicBrush 项目中，不依赖 geocap 项目内的其他组件。

### 4.8 硬编码配置

| 位置 | 值 | 说明 |
|---|---|---|
| `process.py:366` | `ref_path="fos_data/reference_aug_14th"` | 参考图像目录 |
| `process.py:367` | `ref_poles_pool="pics_8xx/"` | 极点参考池 |
| `process.py:368` | `num_refs=10` | 随机参考数量 |
| `diffuser.py:32` | `sys.path.append("/home/nfs03/xingsy/MimicBrush/depthanything")` | 绝对 NFS 路径 |
| `diffuser.py:26` | `OmegaConf.load("./configs/inference.yaml")` | MimicBrush 模型配置（不在本仓库） |
| `process.py:375` | `np.random.seed(0)`, `random.seed(0)` | 固定随机种子 |

---

## 5. 已知问题与注意事项

### 5.1 路径问题

脚本中存在一些硬编码路径（如 `/home/nfs04/`、`/home/nfs05/` 等 NFS 路径，以及个别 Windows 绝对路径），跨环境运行时需要自行检查和替换。

### 5.2 外部运行依赖

**`stage2_diffusion/`** 模块依赖 MimicBrush 扩散模型，**不能在本项目中运行**。需要切换到外部 MimicBrush 项目目录下执行，包括其中的 `run_gradio3_demo.py`、`diffusion_backend_new.py` 等依赖 MimicBrush 管道的脚本。

### 5.3 代码冗余

**两个渲染后端**：PIL（`pil_backend.py`）和 Matplotlib（`plt_backend.py`）功能重叠。**默认使用 Matplotlib**，后者支持更丰富的形状类型（fusiform、spindle、sector、star、Bezier 曲线等）和特效（Perlin 噪声、xkcd 风格）。除非有特殊场景需求，否则不需要使用 PIL 后端。

### 5.4 算法注意事项

1. **`get_angles_and_slopes.py` 对 alpha 通道高度依赖**：轮廓提取完全依赖 RGBA 的第 4 通道，普通 RGB 图像无法使用。
2. **`get_convexity()` 递归回退**：区域划分失败时会以递减 `sample_rate` 递归重试，`sample_rate <= 2` 时抛异常。
3. **`find_extrema_points()` 的非确定性**：`argpartition` 可能导致非确定性排序，`recheck()` 引入了补偿逻辑。
4. **`eval.py` 评估依赖 LLM 质量**：评分可靠性取决于 LLM 的提取和比较能力。8 个特征使用规则覆盖来缓解此问题。

---

## 6. 数据集调优

第三阶段模型训练中，数据集构造经历了三轮迭代。基底模型为 `dec_18_stage2_base`。

### 6.1 数据集构造

数据集基于地科院于 10 月 7 日前后提供的修正后数据（下称**地科数据**）进行构造。

#### 方法 1：完全基于地科数据

将地科数据与旧数据集按特征合并，策略为：
- 某项仅存在于一个数据集 → 取该数据集内容
- 某项同时存在 → 以地科数据为准

**问题**：
1. 部分描述与测试集风格差异大（如 Tunnel shape），导致评分下降。通过修改 `eval/eval.py` 使评分更关注特征来缓解
2. 地科数据与旧数据的人类输入存在出入（如 shape、equator），地科数据可能更准确，但大模型应转述专家工具结果，而非地科数据

#### 方法 2：地科数据 + 数据替换

在合并策略中引入**替换列表**（包含需保留旧数据的特征），修改合并规则：
- 某项仅存在于一个数据集 → 取该数据集内容
- 某项在替换列表中 → 使用旧数据集内容
- 否则 → 使用地科数据内容

同时发现旧数据的数值信息（如 length 的范围值）与人类输入不对齐，因此对旧数据也做了改造，用与人类描述对齐更好的数据替换地科数据对应项。

**问题**：
1. 数值+描述混合特征（如 spirotheca、proloculus）需要更好的数值植入策略
2. 旧数据中部分描述已被 paraphrase 过，再次 paraphrase 会导致效果下滑

#### 方法 3（最终采用）：地科数据 + 数据替换 + 复杂数值替换 + 风格化

针对方法 2 的问题做两项改进：

**复杂数值替换**：对数值+描述混合特征，使用 LLM 将专家工具提供的数值植入定性描述中。例如：

> 原文：*very thin in tightly coiled inner volutions, about 0.008-0.01 mm thick; thickening outwards; the spirotheca on the eighth volution about 0.07 mm thick*
>
> 专家数值：*average 42 microns thick and 25, 51, 43, 42, 45, 46 microns by volutions*
>
> LLM 编辑后：*very thin in tightly coiled inner volutions, thickening outwards; average 42 microns thick and 25, 51, 43, 42, 45, 46 microns by volutions.*

**使用 raw 数据合并**：raw 数据是未被 paraphrase 过的内容，避免多次 paraphrase 导致效果下滑。

### 6.2 基底模型

`dec_18_stage2_base`，相比之前版本：
1. 优化了部分描述项的数据分布
2. 修正了部分数据值

### 6.3 训练经验

1. **数据集影响最大**：超参数调优对总分提升有限，且常伴随某些特征性能下滑
2. **特征并非独立**：修改 B 特征可能导致 A 特征评分剧烈波动，即使 A 特征本身未做任何修改
3. **推理超参数**：temperature、top_p 对结果有一定影响但较小，推荐 `temperature=0.5` 或 `0.6`

---

## 7. InternVL 推理

InternVL 推理在外部 `internvl_chat/` 目录下进行，不在 geocap 项目内运行。

**流程**：

```
1. 修改配置
   internvl_chat/shell/full_ft.sh
   注意 META_PATH 指向的数据集路径
        │
        ▼
2. 运行推理
   python internvl_chat/run_internvl_temperature.py
   环境变量：
     MODEL_NAME  模型目录
     TEMP        推理温度
        │
        ▼
3. 获取结果
   internvl_chat/outputs/
   复制到 geocap 项目下
        │
        ▼
4. 运行评估
   eval_data/outputs/                ← 放置推理结果
   bash stage3_foscap_qwen3_judge.sh  ← 运行评估管道
```

**关键文件**：

| 文件 | 说明 |
|---|---|
| `internvl_chat/shell/full_ft.sh` | 训练/微调脚本，`META_PATH` 指向训练数据集 |
| `internvl_chat/run_internvl_temperature.py` | 推理脚本，通过 `MODEL_NAME` 和 `TEMP` 环境变量控制 |
| `internvl_chat/outputs/` | 推理结果输出目录 |
| `eval_data/outputs/` | geocap 评估输入目录（从 internvl_chat 复制结果至此）

