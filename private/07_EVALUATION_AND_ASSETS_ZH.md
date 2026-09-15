# 评价与结果到实际资产的对应

## 统计对象

指定的锚点是旧工作目录 `docs/model_led/results/C2_UNIFIED/RESULT_ZH.md`。该表统计固定1050条Plan中共同标签完整的1005条；因此表内分母1005，完整1050逐条结果是另一层记录。发布评测默认保留完整请求序列及每个谓词状态，输出passed/pending/count_bounds。

本次选择的默认链路为F评价后保留确认SUN，其余采用一次修订C2的最终选择。它对应统一表中的下列数值：

| 指标 | F800 | 默认C2输出 |
|---|---:|---:|
| Stable | 111/1005 | 113/1005 |
| MetaStable（含Stable） | 570/1005 | 548/1005 |
| SUN | 89/1005 | 105/1005 |
| MSUN | 493/1005 | 518/1005 |
| V | 911/1005 | 911/1005 |
| U | 1003/1005 | 1005/1005 |
| N | 930/1005 | 975/1005 |
| VUN | 838/1005 | 883/1005 |

SUN和MSUN分别增加16、25个；MetaStable总数降低22个，而新颖性增加45个。这说明MSUN改善包含新颖性收益，不能统一解释为稳定性提升。

## 物理与结构谓词

F是生成式扩散精修，CHGNet FIRE则是评价弛豫。后者固定模型0.3.0，软件包0.4.2，允许cell变形，fmax=.1 eV/Å、stress tolerance=.5 GPa、max_steps1000。终态重新计算能量，并检查周期平移下的一致性。

每个组成的hull能量由MP的GGA/GGA+U校正后竞争相构成PhaseDiagram得到。评价量为CHGNet弛豫终态每原子能量减去该组成的MP hull能量。稳定阈值0，亚稳阈值.1 eV/atom。[CHGNet](https://doi.org/10.1038/s42256-023-00716-3)、[Materials Project API](https://docs.materialsproject.org/downloading-data/using-the-api)

N/U使用提交结构，与物理能量所属的弛豫终态分开记录；N比较训练参考，U按当前输出完整原序做later-to-earlier同formula比较。没有把U当成普通单样本训练标签。

comp_valid沿SMACT3.1的氧化态表/Pauling规则，并加入允许指定元素混合价态的补充。组合搜索在同元素内部按无序价态组合合并排列，要求整体电中性。形式价态解表明通过筛选规则，并不确定真实氧化态；CHGNet能运行也不等于通过化学筛选。[SMACT v4混合价态实现思想](https://github.com/WMD-group/SMACT/blob/v4.0.0/smact/screening.py)

本定义只用于本次评测，Planner/B0训练数据的历史charge字段继续使用原规则。定义变化不能算成模型提升。

## 资产角色与配置入口

| 角色 | 配置项、默认输出或来源 |
|---|---|
| LLaDA | `models.dlm`，默认 `hf:GSAI-ML/LLaDA-8B-Instruct` |
| B0 | `models.b0`，默认 `@run/b0/final` |
| C1 | `models.c1`，默认 `@run/c1/best.pt`；参考选择step800 |
| F | `models.diffusion`；使用指定兼容权重，或新训练的 `@run/diffusion/last.pt` |
| C2初始化 | 在新B0上创建几何与状态模块 |
| C2热身后 | `dlm train c2 --stage warmup` 产物 |
| C2原协议训练 | `dlm train c2 --stage editor` 产物 |
| C2轻量更新 | `models.c2`，默认 `@run/c2/light/checkpoint` |
| value | `models.value`，默认 `@run/c2/value/autonomous_value.pt` |
| risk | `models.risk`，默认 `@run/c2/risk/model.json` |
| 一次修订设置 | `c2.revision`，采用实际校准参数 |
| Planner | `models.planner`，默认 `@run/planner/epoch2/final` |

大权重不复制到Git，发布入口从使用者配置的资产加载。原实验资产始终保留在原目录，发布整理只在独立目录工作。

## 配方与验证记录

本说明从原C2训练、轻量更新、value、风险与校准记录，以及B0运行配置中提取实际配方。发布整理后的[参考配方](../docs/reference.md)记录阶段与参数，[验证记录](../docs/validation.md)记录数据、模型和接口检查；[默认配置](../src/dlm_iclr/defaults.json)提供可执行设置。

训练记录给出的关键量为：原editor970来源/488updates，light970来源/61updates，value7424行/1984updates；risk8640端点/1000来源。风险惩罚实际校准为.5，不能使用草案中的.25冒充参考默认。

## 发布与研究结论的关系

新仓库提取可复用内核、统一接口和默认配置。已有结果仍对应既有检查点和既有数值执行。新用户重新训练、换数据或改变batch时会产生新的模型和输出，需重新测量。公开代码会记录各阶段输出，使以后定位差异只需比较对应模块，而不需要在大量旧试验目录中寻找来源。

前置F物理信息改变了C2的信息条件，其收益应按这个执行流程解释。保护机制、编辑器学习、一次修订与最终选择是可以分开的作用；不能把整体表格都归因于某一个神经网络更新。
