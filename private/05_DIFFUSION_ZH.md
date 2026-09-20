# 连续扩散精修：沿用 CrysLLMGen

> F 将 C1 draft 转为连续参照；参照及条件候选的物理后果进一步成为 DLM 的训练监督。完整故事见[物理反馈 Master Story](../docs/PHYSICAL_FEEDBACK_MASTER_STORY_ZH.md)。本页保留原连续模型与训练配方。

## What / Why / How

What：在固定组成下改善DLM提供的连续晶体几何。Why：离散网格有量化误差，token模型也不直接等同于连续几何模型。How：复用CrysLLMGen的连续晶格/分数坐标联合扩散网络，从DLM结构启动其反向精修。

这部分是外部模型的使用与工程适配，必须引用CrysLLMGen和其DiffCSP基础，不能写成我们新提出的扩散原理。[CrysLLMGen，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f789a628fca473e922c806657512a20f-Abstract-Conference.html)、[DiffCSP，NeurIPS 2023](https://arxiv.org/abs/2309.04475)

## 训练变量

L为3×3晶格，R为N×3分数坐标，A为原子种类。网络在给定A及时间t时预测晶格噪声和分数坐标的wrapped score。训练并不是先生成DLM错误晶体再监督纠错；参考模型从原始结构的加噪过程学习。

晶格采用Gaussian forward process：

$$
L_t=\sqrt{\overline\alpha_t}L_0+\sqrt{1-\overline\alpha_t}\epsilon_L.
$$

坐标采用周期加噪：

$$
R_t=(R_0+\sigma_t\epsilon_R)\bmod1.
$$

wrapped normal的梯度同时考虑周期像，避免普通欧氏高斯在0/1边界产生不连续目标。具体代码调用`d_log_p_wrapped_normal`并除以训练schedule的score归一化因子。总loss为晶格噪声MSE与坐标score目标MSE之和。

## 网络和默认参数

使用上游CSPNet：hidden512、6层、SiLU、128个周期距离频率、fully-connected边、LayerNorm、lattice inner-product条件，时间embedding256。实际构造器与所用checkpoint对应，不能仅用上游config.py里未被该构造器使用的hidden256解释网络。

噪声步数1000，lattice cosine schedule，coordinate sigma从.005到.5。训练MP20上游命令为500epochs、batch512；优化器Adam1e-3，梯度value clip1，ReduceLROnPlateau按训练loss调整，factor.6/patience30/threshold1e-4。上游seed1234，验证batch32。

上游保存“训练loss最好”的检查点，但final使用最后迭代的模型对象。新模块分别保留best与last，默认last作为新训练模型的F输入，并明确保存epoch及optimizer/scheduler/RNG。

## 推理究竟做了什么

CrysLLMGen采样从提供的DLM晶格和坐标开始，而不是从全随机噪声重新生成。这里起始反向时间为800，执行800次predictor/corrector，两次几何decoder调用对应一个时间步。

corrector用当前坐标score做局部Langevin式更新；predictor按相邻sigma与lattice扩散schedule推进。原子种类不更新；坐标每步取模，最终输出连续R和L。F不是CHGNet弛豫，后者是独立的物理评价。

## 为什么能加速而不换方法

fc图的连接关系由原子数决定，反向过程内不必重复构造。`reuse_fixed_geometry`复用固定拓扑及与当前噪声坐标无关的计算；依赖当前R/L的网络计算继续每步更新。并行在不同请求之间进行，每个进程保持该请求自己的随机流。

结构元素顺序、800步schedule、同一权重和随机种子构成比较条件。C1更好的raw初态可能让F落入更好盆地；这一因果链需要端点实验，不由几何有效率单独决定。

## MP hull 的位置

F输出后先查询各化学体系的MP竞争相数据；随后CHGNet按共同协议弛豫并估算能量。hull查询本身不能给一个新结构算出能量，它提供的是同组成的最低竞争相组合能量。用户自己的MP API key在运行时读取，查询结果按数据库版本缓存。

## 代码和资产

新模块`diffusion/trainer.py`复用上游损失与优化规则；`diffusion/refinement.py`处理请求、种子和连续输出；数值网络位于`_vendor/crysllmgen`并保留MIT许可。

参考结果使用固定的CrysLLMGen `model_final.pt`。配置项 `models.diffusion` 可指向兼容的已有权重；新训练的默认输出为 `@run/diffusion/last.pt`。
该共享权重的全部原训练日志不在本次资产中；新训练入口按上游配方提供，不把补写入口称作已重新训练了这份权重。
