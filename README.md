# Hadwiger–Nelson / 几何图染色实验项目

本项目围绕“在平面上随机或优化构造单位距离图，并研究其色数”展开，
大致包含三条实验线：

1. **随机几何图 + 精确染色**：在平面上撒结构化点（菱形 / 正三角形 / 单位圆），按距离≈1 连边，
   用 DSATUR 风格回溯算法计算精确色数，并画出带颜色的几何图（入口：`main.py`）。
2. **soft coloring + ML 打分**：把 k-染色 Relax 成可导的概率分布，用 PyTorch 训练“软染色”模型，
   通过 3 色 / 4 色损失差值来度量图的“3 色困难度”，用于筛选更值得精确搜索的图（核心：`soft_coloring.py`, `ml_search.py`）。
3. **抽象图嵌入到几何图**：先在抽象层面随机生成图并算精确色数，再用优化算法把其嵌入到平面，
   得到满足单位距离约束的几何 realization，并验证其色数（入口：`abstract_embed/main.py`）。

这些有限几何图可以看作平面单位距离图的随机或优化子结构，是探索 Hadwiger–Nelson 问题（平面色数）的一个小型实验平台。

---

## 环境与依赖

在项目根目录（包含 `hadwiger_project` 文件夹的地方，例如 `D:/code/ml`）下，建议新建虚拟环境，然后安装依赖：

```bash
pip install -r hadwiger_project/requirements.txt
```

当前 `requirements.txt` 中的主要包：

- `torch`：soft coloring 模型与梯度下降优化；
- `networkx`：抽象图的生成与基本图操作；
- `numpy`：几何构造、随机数与距离计算；
- `matplotlib`：几何图可视化与保存图片。

> 若只想跑最基础的随机几何图 + 精确染色，可在 CPU 上直接运行，soft coloring 与 embedding 也默认优先使用 GPU（若可用）。

---

## 1. 随机几何图 + 精确色数（`main.py`）

**运行方式**（在 `D:/code/ml` 下）：

```bash
python -m hadwiger_project.main
```

主流程（`main()`）：

1. 设定全局参数：
   - `epsilon`：单位距离带宽，边在 `[1-eps, 1+eps]` 内连边；
   - `side`：工作区域 `[0, side]^2`；
   - `strategy`：几何图构造策略（1：菱形+三角形+单位圆；2：高斯场；3：链式 motif）；
   - `n_trials`：独立采样次数。
2. 对每个 trial：
   - 根据 `strategy` 调用 `graph_generation.py` 中的构造函数生成几何图：
     - `random_parallelogram_field()`：撒入若干由两个公边正三角形组成的 **菱形 motif**；
     - `add_random_triangles()`：在基础图上额外撒入正三角形 motif；
     - `grow_on_unit_circles()`：在已有点的单位圆上继续撒点；
     - 或使用 `random_gaussian_field()`、`random_chain_field()` 生成其它风格几何图；
   - `merge_duplicate_points()` 合并数值上重复的点并去重边；
   - 使用 `greedy_dsat_coloring()` 得到近似色数 `approx_k` 作为快速打分。
3. 在所有试验中，挑出最有希望需要更多颜色的若干 candidate，对它们调用
   `chromatic_number_exact()` 做 **精确 DSATUR 回溯染色**，得到真正的最小色数 `exact_k`。
4. 对最佳候选图调用 `plot_colored_graph()` 画图并保存到：

   - `hadwiger_project/outputs/parallelogram_unitcircle_colored.png`

图像说明：

- 顶点：使用 `tab10` 调色板按颜色编号着色；
- 边：用较细的黑色线条展示单位距离约束下的连接关系；
- 标题中包含：`n`、`|E|`、`approx_k`、`exact_k`、`eps_band` 等信息。

---

## 2. Soft Coloring + ML 打分（`soft_coloring.py`, `ml_search.py`）

### 2.1 Soft Coloring 模型

文件：`soft_coloring.py`

- `SoftColoringModel`：
  - 每个点有一组可学习 logits，经 softmax 得到对 `k` 个颜色的概率分布；
  - 损失：对每条边 `(i,j)`，惩罚 `⟨p_i, p_j⟩`，即两端颜色分布越相似惩罚越大；
  - 若图可 k-染，期望通过梯度下降把所有边的颜色“拉开”，令 loss 接近 0。
- `train_soft_coloring(...)`：
  - 支持多次随机初始化（`n_restarts`），返回 **loss 最低的一次结果**；
  - 无边图特殊处理为 `loss = 0`；
  - 返回 `SoftColoringResult(logits, probs, loss)`。

### 2.2 用 ML 给图“打难度分”

文件：`ml_search.py`

- `evaluate_graph_ml(graph, n_soft_steps=800)`：
  - 分别在 3 色、4 色下跑 soft coloring，得到 `loss_3`, `loss_4`；
  - 定义 `score = loss_3 - loss_4`：
    - 若 `score` 偏大，说明“3 染困难、4 染相比容易”；
  - 同时计算：
    - 贪心近似色数 `approx_k`；
    - 精确色数 `exact_k` 及对应染色 `exact_colors`；
  - 打包成 `MLGraphScore` 返回。
- `evaluate_graph_soft_greedy(...)`：
  - 只跑 soft 3/4-coloring + 贪心近似色数，不做精确回溯；
  - 适合作为大量 trial 中的快速筛选指标，
    最终对少数“高分”候选再调用 `chromatic_number_exact` 做精确验证。

> 你可以在自己的几何图生成循环中，调用上述函数，为每张图记录 `(loss_3, loss_4, score, approx_k, exact_k)`，
> 从而用“soft + greedy” 给图的 3 色困难度打一个 ML 风味的分数。

---

## 3. 抽象图 → 几何 realization（`abstract_embed/`）

文件夹：`abstract_embed/`

- `abstract_graph.py`：定义抽象无向图 `AbstractGraph` 以及 `random_erdos_renyi_graph`、`compute_chromatic_number` 等；
- `embedder.py`：实现从抽象图到平面点集的嵌入优化（使边长≈1、非边远离 1）；
- `main.py`：入口脚本 `run_experiment()`，整体流程：
  1. 抽象层面：
     - 多次采样 Erdos–Renyi 抽象图（`n_vertices`, `p_edge`, `n_graph_samples`）；
     - 调用 `compute_chromatic_number` 计算精确色数 `chi`；
     - 只保留 `chi >= target_chi` 的候选抽象图。
  2. 嵌入优化：
     - 对每个候选抽象图，调用 `embed_abstract_graph(...)` 在 `[0, side]^2` 里做连续优化；
     - 损失项包含：
       - 边长度接近 1 的约束；
       - 非边远离 `[1 - nonedge_margin, 1 + nonedge_margin]` 的约束；
     - `EmbeddingResult` 中记录最终 loss、约束违规数等信息。
  3. 几何图与色数：
     - 用 `realized_geometric_graph(...)` 得到真实的点集 `pts` 与单位距离边 `edges_real`；
     - 再次用 `chromatic_number_exact` 计算几何 realization 的精确色数 `chi_real`；
     - 选出 `chi_real` 最大的实例，调用 `plot_colored_graph` 画出并保存到：

       - `hadwiger_project/abstract_embed/outputs/abstract_embed_best.png`

**运行方式**（在 `D:/code/ml` 下）：

```bash
python -m hadwiger_project.abstract_embed.main
```

---

## 4. 代码结构一览

- `main.py`：随机几何图 + 精确染色 + 可视化主入口；
- `graph_generation.py`：各种几何随机图构造函数与 `GeometricGraph` 数据结构；
- `geometry.py`：距离计算与基于距离 ≈ 1 的连边规则；
- `exact_coloring.py`：贪心 DSAT + DSATUR 回溯，给出近似色数与精确色数；
- `soft_coloring.py`：可导的 soft k-染色模型及训练函数；
- `ml_search.py`：基于 soft coloring 的图“难度打分”工具；
- `abstract_embed/`：从抽象图（networkx 风格）到具体几何单位距离图的嵌入与实验脚本；
- `outputs/`, `abstract_embed/outputs/`：运行脚本后生成的带颜色图像输出目录。

---

## 5. 报告 / 项目说明撰写提示（可选）

若这是课程或项目作业的一部分，可以围绕以下问题展开：

1. 几何随机图（epsilon 图 / 单位距离图）如何作为平面色数问题的有限近似？
2. 贪心 DSAT 与 DSATUR 回溯在图染色问题上的实验表现与复杂度差异；
3. soft coloring 在 3/4 色下的 loss 与图的实际色数之间的关系，score 如何帮助筛选“难 3 染”的图；
4. 抽象图嵌入阶段的损失设计：边和非边的约束如何平衡，epsilon 和 nonedge_margin 对结果的影响；
5. 如何基于现有代码进一步扩展：
   - 换用其它随机图模型（几何图 / 小世界图 / 规则图等）；
   - 引入强化学习或进化算法在参数空间中搜索高色数几何图；
   - 系统收集实验数据，对 Hadwiger–Nelson 问题的下界给出“数值实验”视角的讨论。

欢迎根据自己的需求继续精简 / 翻译为英文版 README。
