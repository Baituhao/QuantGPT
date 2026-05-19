
# WorldQuant BRAIN Combined Alpha Performance 核心笔记
## 一、指标定义与计算
**Combined Alpha Performance（CAP）**：GENIUS顾问最重要的名片指标，衡量所有提交过的alpha和super alpha（SA）的整体表现。
- 计算逻辑：所有地区已提交的alpha和SA**等权组合** → 模拟**2023-2025年（semi-OS）**表现 → 扣除手续费 → 得出**费后Sharpe值**
- 本质：反映提交的alpha整体"靠谱程度"，每一个alpha都会影响最终结果

## 二、提升CAP的三大核心关键词
### 1. 分散（Diversity）：降低单一风险，提升稳定性
- 核心逻辑：避免单一因素（地区、模板、信号）表现不佳拖累整体
- 实操方法：
  - **地区分散**：覆盖不同市场环境、手续费、波动性的地区
  - **降低同一地区自相关性（SC）**：使用不同类型模板和数据类别（主信号多样性，而非技术优化或附带信号的"假分散"）
  - 加强风险中性化处理

### 2. 质量：基石，永远优先于数量
- **量化影响**：一个负OS Sharpe的alpha需要多个正alpha才能弥补
  - 示例：1个Sharpe=-1 + 2个Sharpe=+1 → 组合Sharpe仅**0.33**（下降**66%**）
  - 规律：**1个-1 Sharpe需要2个+1 Sharpe才能弥补回整体+1**
  - 推论：2个-1 Sharpe需要4个+1 Sharpe才能弥补
- 提升方法：
  - 确保alpha有清晰的经济学或统计学核心逻辑，不为提交而提交
  - 利用AI和MCP验证alpha合理性，优化至可提交状态

### 3. 手续费：容易被忽视的致命因素
- 注意：提交时显示的Sharpe均为**不计手续费**，CAP和真实交易均为费后
- **量化控制标准**：
  - **ASI指标**：亚洲市场≥**15**（**20以上为佳**），其他地区≥**10**
  - 优化交易频率：调整decay或信号，降低换手率
  - 提交前检查margin，避免低利润空间的alpha

## 三、Super Alpha（SA）：双刃剑的正确使用
### 常见误区与风险
- 误区：IS Sharpe高的SA一定OS表现好（IS与OS表现无直接关系）
- 风险：质量差的原始alpha（RA）会让SA成为"双倍伤害"（Garbage In, Garbage Out）

### 正确使用方法
1. 过滤IQC表现差的RA，**剔除product_correlation>0的RA**，保证组合分散性
2. 优先选择OS表现稳定且为正的RA
3. 优化组合逻辑，降低RA之间的相关性

## 四、量化核心总结
提升CAP的本质是**"少即是多，精益求精"**，严格遵循以下量化标准：
1. **分散标准**：多地区、多模板、多数据类别，加强风险中性化，降低单一alpha对整体的影响
2. **质量红线**：绝不提交OS Sharpe为负的alpha，因为**1个坏alpha需要2个好alpha才能弥补**
3. **手续费底线**：亚洲市场ASI≥15（20+最佳），其他地区ASI≥10，避免低margin高换手alpha
4. **SA使用原则**：只用OS表现稳定为正的RA组SA，剔除相关性>0的RA，防止"垃圾进垃圾出"

# 其他总结
1.Performance Comparison加分越多，sc越低，fit越高，对组合越是有益，钱越多
2.要做有经济学含义的alpha,必须让ai 解释 alpha 的经济学含义
3.fitness有四个区间，最好提交fitness大于2.5以上alpha
4.margin建议提交万分之30以上的
5.pnl中每一年的sharpe和fitness要平稳
