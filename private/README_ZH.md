# DLM 模块化实现与技术论证

> **当前方法入口已更新：** 请先读 [C1/C2 完整方法、数学与教师故事](../docs/C1_C2_METHOD_ZH.md) 和 [当前 README](../README.md)。本目录及 ZIP 保留原模块化说明，其中整体管线和 C2 章节描述历史 post-F 编辑器。当前研究采用完整教师与真实前缀反馈学回 DLM、固定 C1 的路线。

这里收录原私人说明的公开整理版，逐模块解释科学问题、训练和推理过程、优化对象、数学与计算实现，以及对应论文依据。目录名 `private` 沿用原说明的名称，内容的可见性与仓库一致。

[下载完整说明压缩包](technical_notes_zh.zip) · [项目首页](../README.md) · [训练与推理默认配方](../docs/reference.md)

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [科学问题与整体模型](01_SCIENTIFIC_TASK_ZH.md) | What / Why / How、两项贡献、完整生成分布与集合指标 |
| [Planner](02_PLANNER_ZH.md) | 基础版 Llama 3、Plan 数据、两阶段 LoRA、条件生成 |
| [B0](03_B0_ZH.md) | 晶体 token、输入输出表、掩码去噪目标与初始化 |
| [C1](04_C1_ZH.md) | 周期关系势、树消息传递、联合 NLL、实际揭示过程 |
| [连续扩散](05_DIFFUSION_ZH.md) | CrysLLMGen / DiffCSP、晶格与坐标去噪、F800 精修 |
| [C2](06_C2_ZH.md) | 几何条件编辑、连续补丁、轻量物理教师、value 与一次修订 |
| [评价与资产对应](07_EVALUATION_AND_ASSETS_ZH.md) | Direct / SUN / MSUN、结果口径、检查点角色与配置入口 |

## 与代码一起阅读

各章节对应 [`src/dlm_iclr/`](../src/dlm_iclr/) 中的同名模块。训练与推理超参数集中在 [`defaults.json`](../src/dlm_iclr/defaults.json)，数据与模型位置通过[配置文件](../docs/configuration.md)指定。

文档保留实际配方、已有测量和结论的适用条件；机器路径、访问凭据及显卡管理配置保存在使用者自己的本地配置中。压缩包包含本索引及全部 7 篇说明，与本目录的 Markdown 正文一致。离线阅读时，指向项目代码的链接可结合完整仓库使用。
