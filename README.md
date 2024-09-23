<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#stereo>stereo</a></li>
  </ol>
</details>

<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#architecture>architecture</a></li>
  </ol>
</details>

<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#gs_nerf>gs_nerf</a></li>
  </ol>
</details>

<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#data_synthesis>Data_Synthesis</a></li>
  </ol>
</details>

<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#diffusion>diffusion</a></li>
  </ol>
</details>

<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#vfm>VFM</a></li>
  </ol>
</details>

<details>
            <summary>Table of Contents</summary>
            <ol>
                <li><a href=#vlms>vlms</a></li>
  </ol>
</details>


### vlms

#### 1. [AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One](https://arxiv.org/abs/2312.06709) 发表时间: 2023-12-10

![Key Image](images/2312.06709_page_1_img_1.png)

![Key Image](images/2312.06709_page_18_img_7.png)

最近出现了一些视觉基础模型 (VFM)，作为众多下游任务的支柱。像 CLIP、DINOv2、SAM 这样的 VFM 使用不同的目标进行训练，展现出针对各种下游任务的独特特征。我们发现，尽管它们在概念上存在差异，但这些模型可以通过多教师蒸馏有效地合并成一个统一模型。我们将这种方法命名为 AM-RADIO（聚合模型 - 将所有领域简化为一）。这种集成方法不仅超越了单个教师模型的性能，而且还融合了它们的独特功能，例如零样本视觉语言理解、详细的像素级理解和开放词汇分割能力。为了追求最具硬件效率的骨干网络，我们使用相同的训练方案在多教师蒸馏流程中评估了许多架构。这导致了一种新架构 (E-RADIO) 的开发，该架构超越了其前辈的性能，并且至少比教师模型快 7 倍。我们全面的基准测试过程涵盖了下游任务，包括 ImageNet 分类、ADE20k 语义分割、COCO 对象检测和 LLaVa-1.5 框架。代码：https://github.com/NVlabs/RADIO 


---


### VFM

#### 1. [AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One](https://arxiv.org/abs/2312.06709) 发表时间: 2023-12-10

![Key Image](images/2312.06709_page_1_img_1.png)

![Key Image](images/2312.06709_page_18_img_7.png)

近来，少数视觉基础模型 (VFM) 已经成为众多下游任务的支柱。像 CLIP、DINOv2、SAM 这样的 VFM 使用不同的目标进行训练，展现出针对各种下游任务的独特特征。我们发现，尽管它们在概念上存在差异，但这些模型可以通过多教师蒸馏有效地合并成一个统一模型。我们将这种方法命名为 AM-RADIO（聚合模型——将所有领域融合为一）。这种集成方法不仅超越了单个教师模型的性能，还融合了它们各自的特征，例如零样本视觉语言理解、详细的像素级理解和开放词汇分割能力。为了追求最具硬件效率的骨干网络，我们使用相同的训练方案在多教师蒸馏流程中评估了众多架构。最终开发出一种名为 E-RADIO 的新型架构，其性能超越了之前的模型，并且速度至少比教师模型快 7 倍。我们全面的基准测试过程涵盖了下游任务，包括 ImageNet 分类、ADE20k 语义分割、COCO 对象检测和 LLaVa-1.5 框架。代码：https://github.com/NVlabs/RADIO 


---


### diffusion

#### 1. [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/abs/2409.11355) 发表时间: 2024-09-17

![Key Image](images/2409.11355_page_3_img_3.png)

最近的研究表明，大型扩散模型可以通过将深度估计视为图像条件图像生成任务，重新用作高精度单目深度估计器。虽然所提出的模型取得了最先进的结果，但由于多步推理导致的高计算需求限制了它在许多场景中的使用。在本文中，我们发现，感知到的低效率是由推理流程中的一个缺陷造成的，这个缺陷迄今为止还没有被注意到。通过简单的修改，我们能够将推理减少到单步，而性能没有明显下降：固定模型的性能与之前报道的最佳配置相当，但速度提高了200多倍。为了优化下游任务性能，我们在单步模型的基础上使用特定任务的损失进行端到端微调，并获得了一个确定性模型，该模型在常见的零样本基准测试中优于所有其他基于扩散的深度和法线估计模型。我们惊奇地发现，这种微调协议也直接适用于Stable Diffusion，并实现了与当前最先进的基于扩散的深度和法线估计模型相当的性能，这对先前工作中得出的一些结论提出了质疑。 


---

#### 2. [Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695) 发表时间: 2024-09-16

![Key Image](images/2409.10695_page_6_img_1.png)

我们隆重推出最新的文本到图像模型 Playground v3 (PGv3)，该模型在多个测试基准上实现了最先进的 (SoTA) 性能，在图形设计能力方面表现出色，并引入了新的功能。与依赖于 T5 或 CLIP 文本编码器等预训练语言模型的传统文本到图像生成模型不同，我们的方法将大型语言模型 (LLM) 与一种新颖的结构完全集成，该结构仅利用来自仅解码器 LLM 的文本条件。此外，为了提高图像描述质量，我们开发了一种内部描述器，能够生成具有不同详细程度的描述，从而丰富文本结构的多样性。我们还引入了一个新的基准测试 CapsBench 来评估详细的图像描述性能。实验结果表明，PGv3 在文本提示 adherence、复杂推理和准确的文本渲染方面表现出色。用户偏好研究表明，我们的模型在常见的的设计应用（例如贴纸、海报和徽标设计）中，具有超强的图形设计能力。此外，PGv3 还引入了新的功能，包括精确的 RGB 颜色控制和强大的多语言理解能力。 


---

#### 3. [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103) 发表时间: 2024-03-26

近年来，生成式工具的惊人发展促进了许多令人兴奋的应用，例如文本到图像生成和文本到视频生成。这些生成式工具背后的基本原理是扩散的概念，这是一种特殊的采样机制，它克服了先前方法中被认为难以克服的一些缺点。本教程的目标是讨论扩散模型的基本思想。本教程的目标受众包括有兴趣研究扩散模型或应用这些模型解决其他问题的本科生和研究生。 


---


### Data_Synthesis

#### 1. [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) 发表时间: 2024-06-12

高质量的指令数据对于对齐大型语言模型 (LLM) 至关重要。尽管一些模型（如 Llama-3-Instruct）具有开放权重，但它们的校准数据仍然是私有的，这阻碍了人工智能的民主化。高昂的人工成本和有限的、预定义的提示范围阻碍了现有开源数据创建方法的有效扩展，这可能会限制公共校准数据集的多样性和质量。是否可以通过直接从校准的 LLM 中提取高质量的指令数据来大规模合成？我们提出了一种用于生成大规模校准数据的自合成方法，名为 Magpie。我们的主要观察结果是，像 Llama-3-Instruct 这样经过校准的 LLM 可以在我们仅输入用户消息保留位置之前的左侧模板时生成用户查询，这要归功于它们的自回归性质。我们使用这种方法来提示 Llama-3-Instruct 并生成 400 万条指令及其相应的响应。我们对提取的数据进行了全面分析，并选择了 30 万个高质量实例。为了将 Magpie 数据与其他公共指令数据集进行比较，我们使用每个数据集微调 Llama-3-8B-Base，并评估微调模型的性能。我们的结果表明，在某些任务中，使用 Magpie 微调的模型的性能与官方 Llama-3-8B-Instruct 相当，尽管后者通过监督微调 (SFT) 和后续反馈学习增强了 1000 万个数据点。我们还表明，单独使用 Magpie 进行 SFT 可以超越之前用于 SFT 和偏好优化的公共数据集的性能，例如使用 UltraFeedback 进行直接偏好优化。这种优势在 AlpacaEval、ArenaHard 和 WildBench 等校准基准测试中很明显。 


---

#### 2. [UWStereo: A Large Synthetic Dataset for Underwater Stereo Matching](https://arxiv.org/abs/2409.01782) 发表时间: 2024-09-03

![Key Image](images/2409.01782_page_5_img_3.png)

尽管立体匹配技术近期取得了进展，但其扩展到复杂的水下环境仍然 unexplored，这主要是因为：1) 水下图像存在能见度降低、对比度低和其他不利影响；2) 难以获得用于训练深度学习模型的 ground truth 数据，即在水下环境中同时捕捉图像并估计其相应的像素级深度信息。为了进一步推进水下立体匹配技术，我们引入了一个名为 UWStereo 的大型合成数据集。我们的数据集包含 29,568 对合成立体图像对，以及左视图的密集且准确的视差标注。我们设计了四个不同的水下场景，其中充满了珊瑚、船舶和机器人等各种物体。我们还引入了相机模型、照明和环境效果方面的额外变化。与现有的水下数据集相比，UWStereo 在规模、变化性、标注和照片级图像质量方面都具有优势。为了证实 UWStereo 数据集的有效性，我们与 9 种最先进的算法进行了全面比较评估。结果表明，当前的模型仍然难以泛化到新领域。因此，我们设计了一种新策略，该策略学习在立体匹配训练之前重建跨域遮罩图像，并集成跨视图注意力增强模块，该模块聚合远程内容信息以增强泛化能力。

---


### gs_nerf

#### 1. [GLC-SLAM: Gaussian Splatting SLAM with Efficient Loop Closure](https://arxiv.org/abs/2409.10982) 发表时间: 2024-09-17

三维高斯 splatting (3DGS) 因其在密集同步定位和建图 (SLAM) 中的应用而备受关注，它支持实时渲染和高保真建图。然而，现有的基于 3DGS 的 SLAM 方法经常遭受累积跟踪误差和地图漂移的困扰，尤其是在大规模环境中。为了解决这些问题，我们引入了 GLC-SLAM，这是一种集成了相机位姿和场景模型全局优化的高斯 splatting SLAM 系统。我们的方法采用帧到模型跟踪，并使用全局到局部策略触发分层闭环，以最大程度地减少漂移累积。通过将场景划分为 3D 高斯子图，我们可以在大场景中轻松高效地进行闭环校正后的地图更新。此外，我们不确定性最小化的关键帧选择策略优先考虑观察到更有价值的 3D 高斯的关键帧，以增强子图优化。在各种数据集上的实验结果表明，与最先进的密集 RGB-D SLAM 系统相比，GLC-SLAM 实现了更好或相当的跟踪和建图性能。 


---

#### 2. [Improving 2D Feature Representations by 3D-Aware Fine-Tuning](https://arxiv.org/abs/2407.20229) 发表时间: 2024-07-29

![Key Image](images/2407.20229_page_5_img_7.png)

![Key Image](images/2407.20229_page_5_img_11.png)

![Key Image](images/2407.20229_page_5_img_12.png)

![Key Image](images/2407.20229_page_5_img_17.png)

![Key Image](images/2407.20229_page_10_img_20.png)

![Key Image](images/2407.20229_page_10_img_21.png)

![Key Image](images/2407.20229_page_11_img_13.png)

![Key Image](images/2407.20229_page_11_img_14.png)

![Key Image](images/2407.20229_page_13_img_17.png)

![Key Image](images/2407.20229_page_23_img_50.png)

![Key Image](images/2407.20229_page_24_img_34.png)

当前的视觉基础模型仅仅在非结构化的二维数据上进行训练，这限制了它们对物体和场景三维结构的理解。在这项工作中，我们展示了在具有三维感知能力的数据上进行微调可以提高新兴语义特征的质量。我们设计了一种将二维语义特征提升为高效三维高斯表示的方法，该方法允许我们为任意视图重新渲染这些特征。利用渲染后的具有三维感知能力的特征，我们设计了一种微调策略，将这种三维感知能力迁移到二维基础模型中。我们证明，以这种方式微调的模型产生的特征可以通过简单的线性探测 readily 提高语义分割和深度估计的下游任务性能。值得注意的是，尽管是在单个室内数据集上进行微调，但这种改进可以迁移到各种室内数据集和域外数据集。我们希望我们的研究能够鼓励社区在训练二维基础模型时考虑注入三维感知能力。项目页面：https://ywyue.github.io/FiT3D。 


---

#### 3. [GS-Net: Generalizable Plug-and-Play 3D Gaussian Splatting Module](https://arxiv.org/abs/2409.11307) 发表时间: 2024-09-17

![Key Image](images/2409.11307_page_3_img_1.png)

三维高斯 splatting (3DGS) 结合了基于图元的表示方法和体积渲染技术的优势，实现了实时、高质量的渲染。然而，3DGS 模型通常过度拟合单场景训练，并且对高斯椭球的初始化非常敏感，这些椭球是根据运动恢复结构 (SfM) 点云启发式地得到的，这限制了泛化能力和实用性。为了解决这些限制，我们提出了 GS-Net，这是一个可泛化的即插即用 3DGS 模块，可以从稀疏的 SfM 点云中生成密集的高斯椭球，从而增强几何结构表示。据我们所知，GS-Net 是第一个具有跨场景泛化能力的即插即用 3DGS 模块。此外，我们还介绍了 CARLA-NVS 数据集，该数据集包含额外的相机视点，可以全面评估重建和渲染质量。大量实验表明，将 GS-Net 应用于 3DGS 可以使传统视点的 PSNR 提高 2.08 dB，新视点的 PSNR 提高 1.86 dB，证实了该方法的有效性和鲁棒性。 


---


### architecture

#### 1. [Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2409.10594) 发表时间: 2024-09-16

Transformer是现代深度学习的基石。传统上，这些模型依赖于多层感知器 (MLP) 层来混合通道之间的信息。在本文中，我们介绍了 Kolmogorov-Arnold Transformer (KAT)，这是一种新颖的架构，它用 Kolmogorov-Arnold 网络 (KAN) 层取代 MLP 层，以增强模型的表达能力和性能。然而，将 KAN 集成到 Transformer 中并非易事，尤其是在扩展时。具体来说，我们确定了三个关键挑战：(C1) 基函数。KAN 中使用的标准 B 样条函数并未针对现代硬件上的并行计算进行优化，导致推理速度较慢。(C2) 参数和计算效率低下。KAN 需要为每个输入-输出对使用唯一的函数，这使得计算量非常大。(C3) 权重初始化。由于 KAN 的可学习激活函数对于深度神经网络的收敛至关重要，因此其权重初始化特别具有挑战性。为了克服上述挑战，我们提出了三个关键解决方案：(S1) 有理基。我们用有理函数替换 B 样条函数，以提高与现代 GPU 的兼容性。通过在 CUDA 中实现这一点，我们实现了更快的计算速度。(S2) 组 KAN。我们通过一组神经元共享激活权重，以在不牺牲性能的情况下减少计算负载。(S3) 方差保持初始化。我们仔细初始化激活权重，以确保跨层保持激活方差。通过这些设计，KAT 可以有效地扩展并且轻松优于传统的基于 MLP 的 Transformer。 


---


### stereo

#### 1. [MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry](https://arxiv.org/abs/2409.09479v1) 发表时间: 2024-09-14

我们提出了MAC-VO，一种新颖的基于学习的立体视觉里程计，它利用学习的度量感知匹配不确定性来实现双重目的：选择关键点和在位姿图优化中加权残差。与传统的优先考虑富含纹理特征（如边缘）的几何方法相比，我们的关键点选择器采用学习的不确定性，根据全局不一致性过滤掉低质量特征。与为协方差建模尺度无关的对角权重矩阵的基于学习的算法相比，我们设计了一种度量感知协方差模型来捕获关键点配准过程中的空间误差以及不同轴之间的相关性。将此协方差模型集成到位姿图优化中，增强了位姿估计的鲁棒性和可靠性，特别是在具有变化的光照、特征密度和运动模式的挑战性环境中。在公共基准数据集上，MAC-VO 的性能优于现有的 VO 算法，甚至在具有挑战性的环境中优于某些 SLAM 算法。协方差图还提供了有关估计位姿可靠性的宝贵信息，这有利于自主系统的决策。

---

