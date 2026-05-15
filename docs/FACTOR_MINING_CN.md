# 因子挖掘方法论

AI 驱动的自主因子研究循环：读取笔记 → 设计因子 → 回测 → 分析 → 更新知识库 → 迭代。

---

## 概述

QuantGPT 包含一套系统化的因子挖掘框架，将 LLM 驱动的因子设计与严格的统计验证相结合。整个流程分为 6 个阶段，内置研究纪律和交叉审查机制。

核心工具位于 [`scripts/factor_miner.py`](../scripts/factor_miner.py) —— 一个无状态的工具库，用于批量提交因子表达式到 QuantGPT 服务器并解析结果。

---

## 研究循环（6 阶段）

### 阶段 0：环境检查与上下文加载

1. 验证回测服务器健康状态
2. 加载目标研究笔记本（如 `research_notes/archive/factor-mine-reversal.md`）
3. 回顾：当前基线（表达式 + 指标）、关键发现、已完成实验（避免重复）、下一步方向
4. 加载知识库（`research_notes/knowledge/INDEX.md`）：规则（必须遵循）、发现（参考）、失败（不可重复）
5. 确定起始点：第一个未完成的方向

### 阶段 1：因子设计

基于研究上下文和知识库设计 1–3 个因子表达式：

- **陈述假设**（一句话）
- **使用有效算子** —— 50+ 可用（见下方[算子参考](#算子参考)）
- **约束条件**：最大 300 字符，最大 8 层嵌套，禁止使用保留变量名
- **设计原则**：
  - 比值 > 乘法 > 加法：`rank(A/(B+eps))` > `rank(A)*rank(B)`
  - 非线性压缩：`sign_power`、`tanh`、`sigmoid` 处理极端值
  - 条件门控：`where()`、`trade_when()` 用于状态依赖行为
  - 简洁性：超过 4 层嵌套通常会降低表现

### 阶段 2：批量回测提交

```python
from scripts.factor_miner import batch_evaluate

results = batch_evaluate(
    server="http://localhost:8003",
    expressions=[
        "rank(ts_delta(close, 5) / ts_shift(close, 5))",
        "rank(close / ts_mean(close, 10))",
        # ... 每批 10-20 个表达式
    ],
    params={
        "universe": "hs300",
        "holding_period": 5,
        "n_groups": 5,
        "benchmark": "hs300",
        "start_date": "2021-01-01",
        "end_date": "2024-12-31",
    },
    max_concurrent=10,
)
```

- 两阶段并发：先全部提交（3 波重试），再全部轮询
- 结果按 fitness 降序排列
- 若 hs300 fitness < 0.1，跳过 csi500 验证

### 阶段 3：四步分析法

**每个结论必须通过全部四步 —— 无例外。**

**第 1 步 —— 事实收集**：提取指标，与基线对比，暂不下结论。

| 指标 | 基线 | 当前 | 变化 |
|------|------|------|------|
| Fitness | — | — | — |
| Sharpe | — | — | — |
| Returns | — | — | — |
| Turnover | — | — | — |
| IC | — | — | — |
| Rating | — | — | — |

**第 2 步 —— 独立判断**：基于事实形成结论。假设是否被验证？若 fitness 较低，诊断原因：Sharpe 不足、收益太低、还是换手率过高？

**第 3 步 —— 交叉审查**：将事实 + 判断提交给第二个 LLM（DeepSeek Reasoner）进行独立评估。对于包含行动性词汇（采纳、拒绝、推荐、下一步、终止方向）的结论，此步骤为强制要求。

**第 4 步 —— 共识**：若双方一致 → 输出共识结论。若存在分歧 → 展示双方立场及证据，采纳更保守的结论。

### 阶段 4：更新研究笔记

1. 追加实验记录（表达式、参数、指标、分析、结论）
2. 若发现新最优，更新基线
3. 若出现跨实验洞察，更新关键发现
4. 标记已完成方向（删除线）
5. 更新知识库（跨会话洞察）：
   - 稳定规则 → `knowledge/rules/`
   - 经验发现 → `knowledge/findings/`
   - 已证伪路径 → `knowledge/failures/`

### 阶段 5：继续或停止

停止条件（满足任一即停止）：
1. 达到轮次上限
2. 达到时间上限
3. 收敛：连续 N 轮无改善（默认：5）
4. 所有方向耗尽 + 已探索 2 轮自动生成的方向

### 阶段 6：总结报告

输出所有 A/B 级因子、关键发现、新知识库条目及建议的未来方向。

---

## 研究纪律

1. **控制变量**：每次实验只改变一个维度
2. **不重复实验**：先检查笔记和知识库
3. **标注不确定性**：将分析结论标记为"假设，数据驱动"
4. **记录失败**：失败实验同样有价值 —— 记录原因
5. **简洁优于复杂**：干净的表达式胜过 6 层嵌套
6. **知识库是资产**：每个有意义的发现都应持久化

---

## 算子参考

### WQ BRAIN 兼容算子

这些算子生成的表达式可直接提交到 WorldQuant BRAIN 进行独立验证。

| 类别 | 算子 |
|------|------|
| 截面 | `rank(x)`, `zscore(x)`, `scale(x)`, `group_rank(x, group)`, `group_zscore(x, group)` |
| 一元数学 | `abs(x)`, `sign(x)`, `log(x)`, `sqrt(x)` |
| 幂运算 | `power(x, e)`, `sign_power(x, e)` |
| 时间序列 | `ts_mean`, `ts_std`, `ts_max`, `ts_min`, `ts_sum`, `ts_shift`, `ts_delta`, `ts_rank`, `ts_argmax`, `ts_argmin`, `ts_av_diff`, `ts_corr`, `ts_cov`, `decay_linear`, `product` |
| 条件 | `where(cond, t, f)`, `trade_when(cond, alpha, hold)` |
| 二元 | `max(a, b)`, `min(a, b)` |
| 特殊 | `adv20`, `returns`, `vwap`, `cap` |
| 变量 | `open`, `high`, `low`, `close`, `volume`, `market_cap` |

### 仅限本地算子

可用于本地研究但不被 BRAIN 接受。提交 BRAIN 时使用 WQ 替代方案。

| 算子 | WQ 替代 |
|------|---------|
| `tanh(x)` | `sign_power(x, 0.5)` |
| `sigmoid(x)` | `rank(x)` |
| `exp(x)` | `power(2.718, x)` |
| `clip(x, lo, hi)` | `max(lo, min(hi, x))` |
| `ts_zscore(x, N)` | `(x - ts_mean(x,N)) / ts_std(x,N)` |
| `ema/sma/wma` | `decay_linear(x,N)` / `ts_mean(x,N)` |
| `rsi/macd/obv/atr` | 用基础算子手动实现 |

### 已验证的表达式模板

以下结构已被验证能产生高 fitness 因子：

```
rank(ts_delta(close, 5) / ts_shift(close, 5))
rank(close / ts_mean(close, 10))
rank(ts_corr(rank(close), rank(volume), 10))
rank(decay_linear(ts_delta(close, 5) / ts_shift(close, 5), 10))
where(volume > ts_mean(volume, 20),
      rank(close / vwap),
      rank(ts_delta(close, 5) / ts_shift(close, 5)))
```

---

## Fitness 公式

```
Fitness = Sharpe × sqrt(|Returns| / max(Turnover, 0.125))
```

### WQ BRAIN A 级阈值（CN D1 五分组）

| 指标 | 阈值 |
|------|------|
| Sharpe | ≥ 1.625 |
| \|Returns\| | ≥ 6.3% |
| Fitness | ≥ 1.0 |
| Turnover | 1% – 70% |
| 子宇宙 Sharpe | 两半均 ≥ 1.19 |

---

## 文件结构

| 路径 | 用途 |
|------|------|
| `scripts/factor_miner.py` | 提交/轮询/解析工具库 |
| `research_notes/TEMPLATE.md` | 研究笔记本模板 |
| `research_notes/knowledge/` | 跨会话知识库（规则、发现、失败） |
| `research_notes/archive/*.md` | 按方向分类的研究笔记本 |

---

## 批量评估 API

```python
from scripts.factor_miner import batch_evaluate, evaluate

# 单因子评估
result = evaluate(
    server="http://localhost:8003",
    expression="rank(ts_delta(close, 5) / ts_shift(close, 5))",
    params={"universe": "hs300", "holding_period": 5, "n_groups": 5},
)
# 返回: {"expression": "...", "fitness": 0.xxx, "sharpe": ..., "rating": "B", ...}

# 批量评估（10-20 个表达式，并发执行）
results = batch_evaluate(
    server="http://localhost:8003",
    expressions=["rank(...)", "rank(...)", ...],
    params={...},
    max_concurrent=10,
)
# 返回: 因子字典列表，按 fitness 降序排列
```
