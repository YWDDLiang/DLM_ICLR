# KEEP/EDIT：实现与复现说明

本页记录原条件重构方法的实际执行规则和证据口径。[Master Story](PHYSICAL_FEEDBACK_MASTER_STORY_ZH.md) 聚焦周期构造与物理反馈学习；本页说明其既有 pipeline 的具体条件。这次只调整表达，不修改主实验的数据、代码、默认配置、训练资产或结果。

## 完整阶段与确认 SUN 后的保护

学习型核心为 `Plan → B0+C1 → F → 条件提案/连续 patch → 相对 KEEP 选择`。原完整 `dlm run` 按 `c1 → diffusion → hull → physics → c2 → evaluate` 执行。

启用 `c2.protect_sun` 时，pipeline 读取 F 阶段已测评分，核对评分与输入结构绑定，保留已确认 SUN 的请求；其他请求进入条件编辑器。这使用真实物理标签，不能归为 learned value 的预测能力。

C2 读取保存的 F 连续结构及其 token 视图；前置 CHGNet 评价没有用弛豫终态替换编辑器输入。最终输出仍按原请求顺序重算 Unique。实现见 [pipeline.py](../src/dlm_iclr/runtime/pipeline.py)，标签定义见 [evaluation](evaluation.md)。

移出故事正文不表示删除或关闭该规则。复现报告须写明实际配置，旧保护版结果不能直接当成关闭保护后的结果。默认值以 [defaults.json](../src/dlm_iclr/defaults.json)、本次配置快照和 [reference profile](reference.md) 为准。

## 候选身份、连续 patch 与模型特征

每条候选保留提案 token、动作字段、实际结构及提交状态。物理缓存按结构与协议绑定；value/risk 还依赖动作范围与模型输入视图，不能仅凭相同结构 key 复用读出。

仅回写真正变化的数值字段，未改字段保留原 F 连续值；周期别名和无效 patch 按固定规则处理，晶格改变时笛卡尔坐标随之重算。[连续 patch](../src/dlm_iclr/_core/continuous_keep_edit.py) · [value 特征](../src/dlm_iclr/c2/value.py)。

## 一次修订的适用条件

原赢家必须是单 site 的 `local_xyz`，实际改动也限于一个 site，且晶格未变。预算允许时检查最多三个变化坐标，选择预期风险下降最大的一个字段，至多采样一次。

未通过适用性、基线风险或预期风险下降门槛时，保留基础 value 选择；通过后才按相对价值/风险重排 KEEP、原赢家及可用附加候选。新候选使用自己的 token/动作视图重新评分。C2 后没有第二次 diffusion。[revision.py](../src/dlm_iclr/c2/revision.py)。

## 反馈训练与参数路径

原方法训练 `warmup → collect → label → compile → editor → light → value → risk`。有限候选教师影响编辑 DLM 目标，最终编辑器表示用于拟合 value。首次 B0+C1 使用独立资产，未被这条流程自动更新。[训练编排](../src/dlm_iclr/c2/workflow.py)。

另一条 `python -m dlm_iclr.feedback` 配方将完整教师和实际前缀监督学回 draft DLM，其材料、损失和证据见[独立方法说明](C1_C2_METHOD_ZH.md)。不得混用两条路径的效果或检查点。

## 结果归属、未知与成本

[1050 历史报告](results/C2_UNIFIED/RESULT_ZH.md)、[3000 面板](results/C2_HISTORY1000_TOP2_3000/RESULT_ZH.md) 和 [5000 面板](results/C2_HISTORY1000_TOP4_5000/RESULT_ZH.md) 保留各自标签使用、选择/回退规则、历史筛选来源及分母。

某结果若使用额外真实标签回退，必须保留其定义；候选或标签缺失时，不补造另一配置的成绩。失败与未知保留在完整请求分母中，子面板重算自身 Unique，物理标签只按相同结构与协议复用。

成本比较计入实际发生的生成、F、编辑、前置物理计算和最终评价，并标记缓存命中。训练经验被参数复用，不意味着部署 F 或物理计算已被消除。
