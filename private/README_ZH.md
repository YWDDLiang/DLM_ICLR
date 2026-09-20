# DLM 模块化实现与技术论证

> **故事入口：** [物理反馈 Master Story](../docs/PHYSICAL_FEEDBACK_MASTER_STORY_ZH.md) 将原方法组织为周期构造与物理反馈学习，C2 的教师训练和验证机制见本目录。main 的主实验配置、数据和结果保持原绑定。

这里收录原私人说明的公开整理版，逐模块解释科学问题、训练和推理过程、优化对象、数学与计算实现，以及对应论文依据。目录名 `private` 沿用原说明的名称，内容的可见性与仓库一致。

[此前技术说明快照 ZIP](technical_notes_zh.zip) · [项目首页](../README.md) · [训练与推理默认配方](../docs/reference.md) · [实现与复现](../docs/keep-edit-implementation.md)

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [科学问题与整体模型](01_SCIENTIFIC_TASK_ZH.md) | What / Why / How、两项贡献、完整生成分布与集合指标 |
| [Planner](02_PLANNER_ZH.md) | 基础版 Llama 3、Plan 数据、两阶段 LoRA、条件生成 |
| [B0](03_B0_ZH.md) | 晶体 token、输入输出表、掩码去噪目标与初始化 |
| [C1](04_C1_ZH.md) | 周期关系势、树消息传递、联合 NLL、实际揭示过程 |
| [连续扩散](05_DIFFUSION_ZH.md) | CrysLLMGen / DiffCSP、晶格与坐标去噪、F800 精修 |
| [C2](06_C2_ZH.md) | 物理反馈训练条件重构、连续 patch、教师、相对验证与一次修订 |
| [评价与资产对应](07_EVALUATION_AND_ASSETS_ZH.md) | Direct / SUN / MSUN、结果口径、检查点角色与配置入口 |

## 与代码一起阅读

各章节对应 [`src/dlm_iclr/`](../src/dlm_iclr/) 中的同名模块。训练与推理超参数集中在 [`defaults.json`](../src/dlm_iclr/defaults.json)，数据与模型位置通过[配置文件](../docs/configuration.md)指定。

文档保留实际配方、已有测量和结论的适用条件；机器路径、访问凭据及显卡管理配置保存在使用者自己的本地配置中。ZIP 保留此前技术说明快照，本次叙事以在线 Markdown 和完整 Master Story 为准。离线阅读代码链接时，可结合完整仓库使用。
