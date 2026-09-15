# Planner：从原始 Llama 3 8B 开始的条件模型

## What / Why / How

What：学习训练材料中组成与粗粒度材料属性的联合分布，产出可以交给晶体生成器的条件 P。Why：de novo 生成需要合理的条件来源，不能默认给定组成总是可行。How：把每条训练晶体转成七行 Plan，用普通 Llama 3 8B 的因果语言建模目标进行 LoRA 微调。

模型是 Meta-Llama-3-8B **基础版**，不是 Instruct。这里“从0”指从原始预训练基础模型开始本任务微调，不是重新预训练整个 Llama。

## 数据与序列

原始 train/val/test 保持各自划分。预处理先用原 H1A2 晶体量化方法得到数组，再计算 formula、anion_framework、charge_bucket、lattice_system、spacegroup_bucket、volume_per_atom_bin 和结束标志。Plan 条件来自该条结构；采样评估 Plan 则是独立随机输出，不能按行与训练晶体硬配对。

实际训练使用 rich/noid/direct_plan：每来源一条监督，sample_weight=1，不在提示中嵌入 sample ID。基础模型没有 chat template，文本格式采用 `System: ... User: ... Assistant:`。只对 answer 加 EOS 后的 token 计算损失。

## 优化的数学对象

给定提示 c，七行 Plan 序列 p，其自回归概率为

$$
p_\omega(p\mid c)=\prod_{t=1}^{|p|}p_\omega(p_t\mid c,p_{<t}).
$$

loss 为 answer token 的交叉熵均值。提示 token 的 label 为 -100，因此提示作为条件而非重建目标。每个线性变换通过 LoRA 更新 $W=W_0+(\alpha/r)BA$，基础参数固定。[Llama 3](https://arxiv.org/abs/2407.21783)、[LoRA，ICLR 2022](https://openreview.net/forum?id=nZeVKeeFYf9)

实际 rank=16、alpha=32、dropout=.05，目标 q/k/v/o/gate/up/down_proj；可训练参数为41,943,040。IO表沿用原普通文本词表，不添加晶体坐标 token。

## 两阶段训练为何有区别

stage1 用新 LoRA 遍历 TRAIN 一轮；stage2 仅加载该次 stage1/final，重新创建 AdamW 和 cosine，并重新 seed17。stage2 的动量、二阶矩和暖身从零开始。

连续两轮只有一条 scheduler 曲线；历史两阶段的第二轮会重新升到目标学习率，因此并不等价。当前发布入口显式采用两个进程完成两阶段，避免把阶段间 adapter 加载和阶段内状态恢复混在一起。

每阶段27136训练例、有效batch8，3392更新。LR2e-5、weight_decay0、warmup100、clip1、最大长度768；microbatch1/累积8，BF16及梯度检查点。每500步用VAL前50batch诊断并保存。恢复保存模型、优化器、scheduler、采样器顺序/游标及随机状态。

## 推理

采样在同一 rich prompt 下逐token生成，temperature=.9、top-p=.95、top-k50、max_new_tokens96。解析后保存计划文本、组成条件和 G/F 的随机种子。生成器只读取保存的条件，不需要 Planner 常驻显存。

## 代码和资产配置

- [prepare.py](../src/dlm_iclr/planner/prepare.py)：原 H1A2 direct-plan 数据构造。
- [trainer.py](../src/dlm_iclr/planner/trainer.py)：原训练器及其阶段内保存/恢复支持。
- [workflow.py](../src/dlm_iclr/planner/workflow.py)：两个独立阶段。
- [sampling.py](../src/dlm_iclr/planner/sampling.py)：参数化 Plan 采样。
- 基座由 `models.planner_base` 指定，默认 `hf:meta-llama/Meta-Llama-3-8B`。
- 最终 adapter 由 `models.planner` 指定，默认 `@run/planner/epoch2/final`；`@run/` 表示配置的输出目录。

训练和采样有独立入口，可以完成两阶段训练后再按需要采样 Plan。

## 实证与身份

本次原 TRAIN CSV 的历史文件差异已定位为换行；恢复CRLF后与历史源文件匹配。新 prepared JSONL 含后续审计字段，缺少完整旧 prepared 对照文件，因此公开说明使用“已知流程和配方复现”，不声称逐字节复现整个历史训练输入或历史随机环境。当前两阶段训练的效果不能由训练正常启动直接推断。
