# 当前 C1 / C2：方法、数学与教师反馈的完整故事

本文对应研究分支 `codex/verified-feedback-train-20260920` 的注册物理反馈方法，按当前实现说明“生成时怎样构造、教师怎样产生、参数怎样学习、结果怎样验证”。[README](../README.md) 是简要入口，[运行与材料契约](registered-feedback.md) 给出公开接口。历史 post-F 编辑器见 [C2 历史部分](modules/c2.md#historical-c2-editor-retained-for-reproduction)，不计入这里的反馈学习配方。

**核心故事：C1 把 DLM 的位置预测组织成周期相容的坐标联合候选；C2 把连续精修与物理核验得到的改善，转成同条件的完整几何教师和真实前缀监督；DLM 学习后，再通过固定的 C1 生成新的 raw。** 目标是提高生成器自身的物理质量，并检验这种改善能否传递到后续精修质量或精修成本。

## 阅读顺序

1. [整体流程与模块分工](#story)
2. [符号与晶体表示](#notation)
3. [C1 的概率模型、训练与实际提交](#c1)
4. [老师从哪里来，完整几何如何注册](#teacher)
5. [真实前缀如何构成第二类老师](#prefix)
6. [C2 的实际学习目标，以及它怎样影响 C1](#learning)
7. [评价、证据与 F400 的边界](#evidence)
8. [实现对应关系](#implementation)

<a id="story"></a>
## 1. 整体流程与模块分工

单个位置的 token 预测，并不直接约束同一次候选中的多个原子坐标是否相容。原 DLM 已经通过“提交后再次前向”表达跨步依赖；C1 在此基础上，增加当前轴候选内部的周期关系。几何相容又不直接等于低能量、低力或低应力，C2 因而引入外部物理反馈，把找到的更好几何学回生成器。

~~~mermaid
flowchart TB
    subgraph Material["离线构造训练材料：保留原 Plan 与 prompt"]
        P["原 Plan / prompt"] --> G["保留的 DLM+C1 raw 与实际构造轨迹"]
        G --> F["冻结 F 连续精修 / 允许的物理弛豫候选"]
        F --> R["量化为 token + 等价几何注册"]
        R --> V["解码确切 token 结构，再做物理核验"]
        V --> T["完整注册教师"]
        G --> S["真实学生前缀：晶格、可见坐标、提交位置"]
        T --> C["只补全缺失坐标"]
        S --> C
        C --> H["完整补全结构的独立物理核验"]
        G --> H
        H --> B["已知正增益 + 提交目标改变的前缀监督"]
    end
    T --> L["75% 完整教师 + 25% 前缀监督，更新 DLM LoRA"]
    B --> L
    L --> D["更新后的 DLM"]
    subgraph Generate["冻结模型后的生成与评价"]
        NP["固定评价 Plan / prompt / 随机种子"] --> D
        D --> C1["固定参数的 C1，读取新的 logits / hidden"]
        C1 --> RAW["新的 raw"]
        RAW --> RF["可选 F800 / F400：各自从同一 raw 出发"]
        RAW --> E["生成结束后的物理与集合指标评价"]
        RF --> E
    end
~~~

| 部件 | 提供什么 | 当前 C2 学习阶段是否更新 |
|---|---|---|
| Planner | 条件、原子数、组成及原始 prompt；确定生成请求 | 否 |
| DLM / B0 初始化 | 晶格与坐标 token 的 logits，以及同次前向的 hidden states | 只更新已有 DLM LoRA；骨干和训练过的输入输出词表保持固定 |
| C1 周期头 | 当前轴的周期边势、联合候选分布 | 参数固定；分布随 DLM 和已生成上下文改变 |
| 冻结连续 F | 晶格、坐标候选；可在部署时继续作为可选精修器 | 否 |
| CHGNet、弛豫与 hull 参考 | 对确切结构的能量、力、应力及稳定性代理测量 | 不参与本损失的反向传播 |
| C2 反馈机制 | 条件绑定、教师注册、前缀补全、核验、监督选择与采样权重 | 产生训练材料，学习效果保存在更新后的 DLM 中 |

这里的“老师”是一套**候选生成、物理核验与监督构造流程**，不是另外训练出来的大语言模型。F 提出连续几何，物理工具提供评价，注册和前缀约束把这些几何转成学生可学的目标；三者各有职责。也不能把 F 的一次输出自动称为好老师。

<a id="notation"></a>
## 2. 符号与晶体表示

| 符号 | 含义 |
|---|---|
| $c$ | 同一请求的原 Plan / prompt；原子数与元素槽位固定 |
| $S=(A,L,X)$ | 组成 / 元素槽位 $A$、以行向量存储的晶格矩阵 $L$、分数坐标 $X\in[0,1)^{N\times3}$ |
| $y=\operatorname{Tok}(S)$，$\operatorname{Dec}(y)$ | 离散晶体 token，以及从这些确切 token 解码的结构 |
| $\theta,\psi,\phi$ | DLM LoRA 可学习参数、C1 周期头参数、连续 F 参数；DLM 其余参数在本文中省略 |
| $s_t=(c,y_{\rm visible},M_t)$ | 某次真实构造前缀及可见 / 遮蔽状态 |
| $a\in\{X,Y,Z\}$，$z_i\in\mathbb Z_Q$ | 当前坐标轴及第 $i$ 个原子的坐标 bin；$Q=100$ |
| $\widetilde S,F_b(G;\xi)$ | 连续候选，以及从 raw $G$、随机流 $\xi$ 出发的 $b$ 步精修端点 |
| $u(S),\Delta_v,w_v$ | 原生质量代理、前缀补全相对原 raw 的增益、用于抽样的正权重 |

当前 token body 有 $7+4N$ 个槽位：原子数、6 个晶格参数，再加每个原子的元素和 XYZ。默认晶格长度分辨率为 0.1 Å，角度为 1°，分数坐标为 0.01。坐标 000 与 100 表示同一周期位置，概率处理必须合并这两个别名。

原 prompt 一直保留。固定组成不意味着 prompt 中所有软条件都已被严格满足；几何与物理质量仍需测量。

<a id="c1"></a>
## 3. C1：当前坐标轴上的周期联合候选

### 3.1 同次 DLM 前向提供一元项与上下文

在状态 $s$ 下，DLM 输出 logits $\ell_\theta$ 和 hidden states $h_\theta$。把坐标 token 别名先合并为语义 bin：

$$
\bar\ell_i(k;s)=
\log\sum_{v:\operatorname{bin}(v)=k}
\exp\ell_{\theta,i}(v;s),\qquad k\in\mathbb Z_Q.
$$

实现先应用调用方既有的类型 / 几何支持约束，再合并别名，最后施加温度；被排除的值是零概率支持。可见坐标作为条件证据固定，其一元常数可以置零。C1 使用同一次前向的 hidden states，不为周期头额外调用 DLM。

### 3.2 晶格条件的 Fourier 周期边势

固定树以原子槽位 0 为根。当前默认父节点为 $p(j)=\lfloor(j-1)/2\rfloor$，$j=1,\ldots,N-1$。每条父子边的上下文包含：

- 两端的归一化 hidden states 和元素 embedding；
- 晶格 Gram 矩阵 $LL^\top/\operatorname{tr}(LL^\top)$ 的 6 个独立分量、$\log|\det L|$ 和当前轴；
- 其他两轴上两端都已知时的周期差的 sin / cos，以及各端的可见性。

当前轴的数值不直接作为边势特征输入，已知值通过条件证据处理；DLM hidden 仍合法地包含整个可见上下文。小型网络输出 Fourier 系数，形成有限的**对数相容性势**：

$$
g_{\psi,pj}(d;s)=B\tanh\!\left[
\frac{1}{\sqrt H}\sum_{m=1}^{H}
\left(
\alpha_{pjm}(s)\cos\frac{2\pi md}{Q}
+\beta_{pjm}(s)\sin\frac{2\pi md}{Q}
\right)\right],
\qquad d=(z_p-z_j)\bmod Q.
$$

当前默认 $H=8$、$B=4$、网络宽度 64。这些势是学到的概率相容性分数，不能解释成 CHGNet 或 DFT 势能。

### 3.3 条件分布与精确树推断

令 $\mathcal E_s$ 表示可见坐标证据，支持约束并入 $\bar\ell$。C1 建立

$$
q_{\theta,\psi}^{(a)}(z\mid s)=
\frac{\mathbf 1[z\text{ 满足 }\mathcal E_s]}{Z(s)}
\exp\!\left\{\frac{1}{\tau}
\left[\sum_i\bar\ell_i(z_i;s)
+\sum_{j=1}^{N-1}g_{\psi,p(j)j}((z_{p(j)}-z_j)\bmod Q;s)\right]\right\}.
$$

温度 $\tau$ 同时缩放一元项和边势。固定本次前向得到的势后，树上的向上传递为

$$
m_{j\to p}(z_p)=
\log\sum_{z_j}
\exp\!\left[
\frac{\bar\ell_j(z_j)+g_{\psi,pj}(z_p-z_j)}{\tau}
+\sum_{k\in\operatorname{ch}(j)}m_{k\to j}(z_j)
\right].
$$

可见证据通过排除不相符的 $z_j$ 实现。由根节点求出 $Z(s)$，再计算条件边缘和祖先式联合采样，时间复杂度为 $O(NQ^2)$。这里“精确”指**固定势、指定树和支持下的概率推断**，实现仍受浮点运算与随机数精度限制。

周期势表达分数坐标差的周期性；固定槽位树本身不保证原子置换等变，也不保证全分布对任意坐标平移不变，因为一元项和 DLM 上下文仍依赖表示。

### 3.4 联合候选怎样进入真实构造

实际流程先固定组成，再生成晶格；随后按 X、Y、Z 和元素分组的既定 schedule 构造坐标。每步从当前轴联合分布采样候选，再按当前活动组的置信度选出要提交的位置；未提交的候选仍是辅助变量。提交后重跑 DLM，重建下一步的周期分布。

当前反馈实验保留 `legacy_unary` 提交置信度：在联合候选值处读取未加采样温度的一元 softmax 分数，再沿原规则选择提交。代码另有 `joint_marginal` 选项，它未作为本次配方的新增贡献。

因此，联合候选概率 $\log q(z\mid s)$ **不等于**置信度投影后单次实际提交的概率。若以 $\Phi_s(z)$ 表示候选到提交后状态的映射，则坐标步骤的真实转移形式是

$$
K_{\theta,\psi}(s'\mid s)=
\sum_z q_{\theta,\psi}(z\mid s)\,
\mathbf 1\{\Phi_s(z)=s'\}.
$$

实现没有把这个对辅助变量求和的 $K$ 当作精确轨迹似然计算。C1 也没有一次性精确求解“晶格 + 全部 XYZ”的全局联合分布。

### 3.5 C1 自身训练与后续 C2 学习分开

C1 自身训练时固定 DLM，使用原训练晶体的轴前缀视图，优化条件联合 NLL：

$$
\mathcal L_{\mathrm{C1}}(\psi)=
\mathbb E_{(c,y),a,\mathrm{cut}}
\left[-\log q_{\theta_0,\psi}^{(a)}
(z^{\rm target}\mid s^{\rm target}_{a,\mathrm{cut}})\right].
$$

该视图保留之前各轴和当前轴的可见前缀，遮住其余坐标。后续本文的 C2 配方固定已经选定的 $\psi$，只更新 DLM LoRA；这是两个不同训练阶段，不能把联合头再训练算入当前反馈收益。

<a id="teacher"></a>
## 4. 完整老师：连续几何经过离散与物理两道核验

### 4.1 候选来源和各自的角色

对同一条件 $c$，保留一个学习前 raw 作为表示锚点 $G_c$。候选可以来自冻结 F 的端点，也可以来自允许的 CHGNet 弛豫终态。当前 1306 请求研究复用了原 G/F、严格匹配到原 F 输入的弛豫终态与原始构造轨迹，没有导入旧 post-F 编辑 E 作为教师。

使用连续候选 $\widetilde S$ 时，实际交给 DLM 的是 $y=\operatorname{Tok}(\widetilde S)$。物理标签必须对应 $\operatorname{Dec}(y)$；原连续端点的好分数不能直接复制给它。原始记录 ID 也不能代替结构 key 的核验。

### 4.2 保留老师的物理几何，选择更合适的 token 表示

晶体可通过全局周期平移、同元素原子重排得到等价表示。完整教师只允许：

$$
X_i^{\rm reg}=
\left(X_{\pi(i)}^{\rm teacher}+\frac{\delta}{Q}\right)\bmod1,
\qquad
A_{\pi(i)}=A_i,\quad \delta\in\mathbb Z_Q^3,
\qquad L^{\rm reg}=L^{\rm teacher}.
$$

注册过程近似寻找使教师接近锚点表示的 $(\pi,\delta)$：

$$
\min_{\pi,\delta}
\frac1N\sum_i
d_{L^{\rm teacher}}^2
\left(X_i^{G_c},
\left(X_{\pi(i)}^{\rm teacher}+\delta/Q\right)\bmod1\right),
$$

其中 $d_L$ 是在教师晶格下的最短周期距离。实现使用多个平移起点、同元素 Hungarian 匹配和有限次修正，**不是全局最优保证**。整数 bin 平移和同元素置换保留老师的晶格及相对周期几何；得到的新 token body 再解码，并以自己的结构 key 做物理核验。

这里对齐的是等价表示，不是把老师所有坐标拉回坏的 raw，也不是把教师强行放进学生晶格。完整教师仍用自己的晶格与坐标作为相容目标。

### 4.3 原生质量代理及当前材料的选择规则

记 CHGNet 单点的力 RMS 为 $f_{\rm rms}$、最大力为 $f_{\max}$、最大应力为 $\sigma_{\max}$、相对参考 hull 的原生能量为 $e_{\rm hull}^{\rm raw}$。当前 `raw_utility` 为

$$
u(S)=-\min\left\{12,\,
0.35\log\left(1+\frac{\max(f_{\rm rms},0)}{0.1}\right)
+0.15\log\left(1+\frac{\max(f_{\max},0)}{0.2}\right)
+0.15\log\left(1+\frac{\max(\sigma_{\max},0)}{0.5}\right)
+0.35\log\left(1+\frac{\max(e_{\rm hull}^{\rm raw},0)}{0.1}\right)
\right\}.
$$

力用 eV/Å、应力用 GPa、hull 能量用 eV/atom；各比例为无量纲。已知有效测量的 $u\in[-12,0]$，越大越好；过度负的 hull 不额外加分。这里的 0.2 等是代理的尺度，**不改变稳定性判定阈值**。未知测量不产生可比较的 $u$；已知生成失败 / 无效 raw 在该函数中记为 −12，两者必须区分。

对 1306 汇报请求的这次材料构造，定义原生严格 / 亚稳定代理

$$
I_h(S)=\mathbf1\{
e_{\rm hull}^{\rm raw}\le h,\quad
f_{\max}\le0.1,\quad
\sigma_{\max}\le0.5
\},\qquad h\in\{0,0.1\}.
$$

在同条件、支持合法、测量已知的候选中，允许 $I_{0.1}=1$ **或** $u(S)>u(G_c)$ 的候选入池；再按 $(I_0,I_{0.1},u)$ 字典序选择一个完整教师，平分时按结构 key 固定排序。这个规则没有要求先达到 SUN/MSUN。它是本次材料的固定选择协议；公开训练入口消费审核后的材料，并不自行重做候选筛选。

最终 1244 个完整教师中，15 个达到原生严格代理，50 个达到含严格的原生亚稳定代理，其余 1194 个来自已知的相对改善。教师质量分布、未选候选及未知记录均保留。找到这些教师说明监督材料可用，尚不证明新学生已经改善。

<a id="prefix"></a>
## 5. 前缀老师：在学生真正到过的位置给监督

完整教师能教相容的晶格与坐标，但教师前缀可能不同于学生实际生成的前缀。第二类监督因而读取原始构造轨迹：原 prompt、学生已经生成的晶格、每个可见坐标，以及该步实际提交的位置集合 $A_v$。

当前 1306 请求研究在获得新补全物理标签前，按每个轴的实际 first / middle / last 事件选视图，每条轨迹最多 9 个。这些是缓存的真实学生轨迹，不是当前更新后策略的新 rollout；完整教师的训练遮蔽视图也不能冒称真实学生轨迹。

### 5.1 补全保持全部可见内容

令 $M_{ia}=1$ 表示坐标可见。前缀补全 $B_v$ 满足

$$
L^{B_v}=L^{\rm student},\qquad
X_{ia}^{B_v}=
\begin{cases}
X_{ia}^{\rm student},&M_{ia}=1,\\
(X_{\pi(i),a}^{\rm teacher}+\delta_a/Q)\bmod1,&M_{ia}=0.
\end{cases}
$$

元素槽位和原 prompt 同样固定。前缀注册用可见分量的、按学生晶格长度缩放的 wrapped 差作近似匹配代价：

$$
\sum_{i,a}M_{ia}\,l_a^2\,
\operatorname{wrap}\!\left(
X_{ia}^{\rm student}-X_{\pi(i),a}^{\rm teacher}-\delta_a/Q
\right)^2.
$$

该代价使用长度权重，**不是非正交晶胞完整 Gram 度量下的精确匹配**。它只提出缺失坐标；改变所处晶格、保留一部分学生坐标后，整个补全结构一般不再与完整教师等价。必须对 $B_v$ 的确切 token 解码结构重新做物理单点，才能知道它是否有用。

### 5.2 增益标签属于完整补全

参考对象是生成这条实际前缀的原始完整 raw $G_{\rm request(v)}$：

$$
\Delta_v=u(B_v)-u(G_{\rm request(v)}),\qquad
w_v=\frac{\Delta_v}{1+\Delta_v}\quad(\Delta_v>0).
$$

当前核心前缀监督要求同组成、同物理协议、两侧增益已知、$\Delta_v>0$，并且至少一个原提交目标发生改变。其他物理字段的改善与取舍另行保留；未知不是负例，没有正核心增益的记录也不会被包装成成功监督。

训练输入仍为真实前缀，训练目标只取 $B_v$ 在原提交位置 $A_v$ 上的 token。其余补全坐标用于完整结构的物理核验，不逐个计入这条前缀视图的损失。当前得到 3747 个有效视图，覆盖 945 个条件、1937 个 source-axis 组合。

$\Delta_v$ 衡量的是**整个补全相对原 raw 的质量差**。本过程没有控制其余未来坐标不变，也没有估计执行单个动作后的策略价值，故不能把它称为孤立提交动作的因果 advantage、精确 Q 值或精确 trajectory DPO 标签。

<a id="learning"></a>
## 6. C2 的实际训练目标

### 6.1 两类视图共用有类型约束的 token 重构

对视图 $v=(c,y_{\rm input},A_v,y^+)$，在对应字段合法 token 家族内归一化。坐标别名仍在温度前合并：

$$
\widetilde p_\theta(k\mid v,j)=
\frac{\exp(\bar\ell_{\theta,j}(k;v)/T_{\rm train})}
{\sum_{k'\in\mathcal V_j}\exp(\bar\ell_{\theta,j}(k';v)/T_{\rm train})},
\qquad
\ell(v;\theta)=-\frac1{|A_v|}
\sum_{j\in A_v}\log\widetilde p_\theta(y_j^+\mid v,j).
$$

当前 $T_{\rm train}=0.7$。这是训练中的 typed-unary 重构分布，省略 C1 边势、动态几何 mask 和置信度提交投影；它与生成时的完整 $q$ 或真实转移 $K$ 不同。

完整教师视图分为晶格、X、Y、Z 四个 phase。当前 phase 待预测字段和未来 phase 被遮住；晶格 phase 的未来坐标始终不可见。训练交替使用整个当前 phase 与部分当前 phase 的遮蔽，原子数与组成不变。先前 phase 的可见值来自同一个完整教师，因此晶格与坐标目标相容。

真实前缀视图则直接保留记录中的输入，只监督实际提交位置上的已核验纠正目标。

### 6.2 75/25 混合与权重的确切含义

整个固定 microbatch 周期的目标可写成

$$
\mathcal L_{\rm C2}(\theta)=
0.75\,\mathbb E_{v\sim\mathcal D_T^{\rm balanced}}\ell(v;\theta)
+0.25\,\mathbb E_{v\sim\mathcal D_P^{\rm balanced,w}}\ell(v;\theta).
$$

完整教师按 source 和四个 phase 遍历；前缀先平衡 source / axis，再在对应池内按 $w_v$ 抽样：

$$
\Pr(v\mid \text{source},a)=
\frac{w_v}{\sum_{v'\in\mathcal P_{\text{source},a}}w_{v'}}.
$$

**$w_v$ 用于抽样，不是抽到后再额外乘一次损失。** 每个视图先对监督字段取平均，避免长晶体仅因 token 多而天然获得更大权重。固定 microbatch 顺序为 teacher / feedback / teacher / teacher，75/25 是完整周期的比例，未要求每个 optimizer update 都恰好 75/25。

优化器为 AdamW，LoRA LR $10^{-5}$，weight decay 0，梯度范数裁剪 1，microbatch 4、effective batch 8。当前配方没有 DPO、reference-KL 项或额外 TRAIN replay，也没有对 F / CHGNet 反向传播。保留的旧实验接口不能被算入这条损失。

### 6.3 固定 C1 为什么仍能得到不同的生成分布

训练只改变 $\theta$。下一轮生成的因果路径是

$$
\mathcal D_T,\mathcal D_P
\ \longrightarrow\ \theta^+
\ \longrightarrow\ (\ell_{\theta^+}(s),h_{\theta^+}(s))
\ \longrightarrow\ q_{\theta^+,\psi}^{(a)}(\,\cdot\mid s)
\ \longrightarrow\ \text{候选、实际提交与后续新前缀}.
$$

即使固定 $\psi$，新 hidden states 也会改变周期头预测的边势；新 logits 改变一元项；新生成的晶格与可见坐标又改变后续上下文。C2 因而通过学回 DLM 影响 C1 的后续位置选择。当前反馈损失并未直接优化 C1 联合似然或让梯度穿过提交采样。

部署时只加载冻结的模型资产和协议，用原 prompt 与已生成内容构造 raw；没有在线教师检索、CHGNet 打分选候选或未来 F 标签。若另外研究 learned 在线 critic，需要独立训练和对照证据。

### 6.4 小配方与扩大训练

| 设置 | 小规模已见 TRAIN 机制 | 1306 请求的扩大训练 |
|---|---:|---:|
| 更新步数 | 512 | 6656 |
| teacher / feedback 比例 | 75% / 25% | 75% / 25% |
| C1 参数 | 固定 | 固定 |
| 训练 / 生成温度 | 0.7 / 0.2 | 0.7 / 0.2 |
| 扩大训练实际视图数 | — | teacher 39936；feedback 13312 |
| 扩大训练覆盖 | — | 4976 个 teacher source-phase，每个至少 8 次训练视图曝光 |

扩大训练的步数由材料与 phase 覆盖设定，没有机械复用 16 条小样本的 512 步。公开 [参考 recipe](../configs/registered_feedback.json) 保留小配方；6656 步是扩大研究单独固定的 profile，不是该文件的新默认值。

checkpoint 选择只看预先固定的教师 / 前缀探针：

$$
J_{\rm fit}(\theta)=
0.75\,\mathrm{NLL}_{\rm teacher}(\theta)
+0.25\,\mathrm{NLL}_{\rm fixed\ prefix}(\theta).
$$

教师评分覆盖全部完整教师的四个 phase；固定前缀探针对每个 source-axis 选一个视图。它们衡量拟合，不能代替自由生成与物理评价。

<a id="evidence"></a>
## 7. 如何检验整个故事

### 7.1 三层证据分别报告

| 问题 | 所需证据 |
|---|---|
| 外部工具是否提供了可学的改善？ | 教师 / 原 raw 的物理比较、等价注册、精确 token 重验与原 prompt 绑定 |
| 学生是否把改善学到了参数中？ | 拟合诊断之后，用冻结 checkpoint 自由生成；报告同 Plan 的成对物理结果 |
| 是否得到泛化或精修效率收益？ | 独立 DEV、冻结最终测试；或同 raw 的 F400/F800 质量与实际成本对照 |

截至 **2026-09-20 本次方法文档更新时**：16 个已见 TRAIN 条件 × 4 个新流的对照已完成；扩大训练和 1306 请求的完整 raw 配对也已完成。扩大 raw 没有复现原生稳定改善，F800 / F400 成对评价尚未完成；当前评价与后续研究在本次文档提交后暂停讨论。

小规模同温度对照中，raw 原生严格稳定代理 7/64→35/64，28 增、0 损，按 16 条来源分组的 bootstrap 增量 95% 区间为 23.44–65.63 个百分点。raw 联合 SUN 6→13、MSUN 11→19；F800 联合 SUN 10→9、MSUN 17→16。raw 精确结构 key 56→35，存在集中与重复增加。详见[完整结果和取舍](results/registered-feedback-train.md)，这些数字不等于未见 MP-20 泛化或 35 种新材料。

扩大训练最佳 checkpoint 为 step 6656。完整教师 macro NLL 4.81624→3.70280，固定前缀 NLL 5.53935→3.51846；坐标 top-1 仅 11.6%、±1 bin 13.9%。这说明拟合进步，仍需真实生成检验。其 before 是小规模学习的 model25 起点，不是论文原始 B0。

实际扩大 raw 的原生严格代理两侧均为 1/1306、各 39 个未知；常规联合 SUN 33→34、MSUN 178→159，SUN 未知为 152→146，共同已知 SUN 为 15 增、15 损。平均原生能量和力 RMS 变差，最短异位点距离 <1 Å 的请求也增多。这个结果显示拟合与自由生成质量脱节，尚未确定具体原因；不能把此前的小 TRAIN 结果外推为扩大学习成功。见[扩大 raw 完整结果及只读诊断](results/reporting-1306-raw.md)。

### 7.2 原生代理、常规 SUN 与集合指标

原生单点代理在确切生成端点上测量能量、力和应力。常规 raw SUN/MSUN 中的稳定性标签来自既定 CHGNet 弛豫终态，U/N 使用保存的生成结构；F 报告中的 raw 字段指该 F 端点。二者不能混作同一种“raw 稳定性”。

常规 Stable 使用 hull $\le0$，MetaStable 使用 $\le0.1$ eV/atom，MetaStable 包含 Stable；原 force / stress 判据保持不变。SUN / MSUN 还需要相应的有效性、Unique 和 Novelty，不能用单个样本的能量标签直接代替。完整定义见[评价协议](evaluation.md)。

1306 请求包含 1298 个不同原 prompt，8 个重复请求；原失败 Plan 也保留分母。教师和前缀覆盖不足的条件不从评价面板中删除。联合 1306 与 H1A2_1050、R03_256 子面板分别重算 U/N 和完整 Direct；共享结构 key 的物理标签可以复用，Unique 数不能简单相加。bootstrap 按原 prompt 来源归组。

这些旧汇报请求已经转作 TRAIN。DEV 的 Direct 参考用 VAL，最终固定测试才用 TEST；Novelty 使用完整 benchmark TRAIN 参考。独立 DEV 证据成立前，不把这批 seen-TRAIN 结果当作最终主面板放行依据。

### 7.3 F400 是预先固定的质量与预算对照

对每个模型 $m\in\{\mathrm{before},\mathrm{student}\}$ 只生成一次

$$
G_c^m=\operatorname{Generate}_{\theta_m,\psi}(c;\xi_{\rm body}),
\qquad
S_{c,b}^m=F_b(G_c^m;\xi_F),\quad b\in\{800,400\}.
$$

比较相同端点的 student / before、各模型内部 F400 / F800，以及 student-F400 / before-F800。两个模型保持原 Plan 顺序、prompt、body / F seed、body 温度 0.2、C1 规则与物理协议。F400 不额外生成一份 raw。

源码设置 `time_start = diff_steps`；F400 同时改变扩散起始时间和步数，不能称为同一条 F800 轨迹无损砍半。两者使用 `ordered_csr_v1`，独立输出与缓存身份。报告精修实际耗时、生成成本、物理评价耗时和缓存复用，不能把物理评价总时长当成部署精修成本。只有完整质量、未知和成本数据出来后，才能判断是否存在效率收益。

<a id="implementation"></a>
## 8. 从数学到代码

| 定义 / 操作 | 当前实现 |
|---|---|
| 周期别名合并、树分布、Fourier 势、分区函数和采样 | [c1/distribution.py](../src/dlm_iclr/c1/distribution.py) |
| 同次 logits / hidden、几何支持接入、候选与提交记录 | [c1/sampling.py](../src/dlm_iclr/c1/sampling.py)、[c1/generation.py](../src/dlm_iclr/c1/generation.py) |
| 轴前缀条件联合 NLL，固定 DLM 训练 C1 | [c1/objectives.py](../src/dlm_iclr/c1/objectives.py) |
| 完整教师的同元素置换与整数 bin 平移 | [teacher_registration.py](../src/dlm_iclr/draft_loop/teacher_registration.py) |
| 真实提交轨迹提取、保持学生前缀的补全提案 | [state_replay.py](../src/dlm_iclr/draft_loop/state_replay.py)、[prefix_teacher.py](../src/dlm_iclr/draft_loop/prefix_teacher.py) |
| 原生质量代理 $u$ | [quality.py 的 raw_utility](../src/dlm_iclr/draft_loop/quality.py)；该文件其他旧 gate 不等于本文材料筛选规则 |
| typed-unary 别名与概率、双类视图、权重抽样、75/25 学习 | [learning.py 的 TypedVocabulary](../src/dlm_iclr/draft_loop/learning.py)、[teacher_fit.py](../src/dlm_iclr/draft_loop/teacher_fit.py) |
| 材料 digest、token / 物理绑定与公开 train / evaluate 入口 | [feedback.py](../src/dlm_iclr/feedback.py) |
| 连续精修与生成后评价 | [refinement.py](../src/dlm_iclr/diffusion/refinement.py)、[draft_loop/evaluation.py](../src/dlm_iclr/draft_loop/evaluation.py) |

公开接口提供材料构造原语及审核材料的训练 / 评价入口。此次旧汇报数据的来源整合与队列编排属于研究运行材料；上文说明其选择协议，不把私人编排脚本假称为公开 CLI 的自动功能。私有资产路径、密钥与大权重不属于本文档。

这次更新只完善说明。执行方法仍对应科学实现 `45acf51` 加公开入口身份校验修复 `8aa8268`，未借文档更新改变当前评价中的采样、教师、损失或物理判据。
