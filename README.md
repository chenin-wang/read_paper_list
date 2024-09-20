## [AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One](https://arxiv.org/abs/2312.06709) 发表时间: 2023-12-10


近来，少数视觉基础模型 (VFM) 已经成为众多下游任务的支柱。像 CLIP、DINOv2、SAM 这样的 VFM 使用不同的目标进行训练，展现出针对各种下游任务的独特特征。我们发现，尽管它们在概念上存在差异，但这些模型可以通过多教师蒸馏有效地合并成一个统一的模型。我们将这种方法命名为 AM-RADIO（聚合模型——将所有领域简化为一个）。这种集成方法不仅超越了单个教师模型的性能，而且还融合了它们各自的独特功能，例如零样本视觉语言理解、详细的像素级理解和开放词汇分割能力。为了追求最高效的硬件骨干网络，我们使用相同的训练方法在多教师蒸馏流程中评估了许多架构。这导致了一种新架构 (E-RADIO) 的开发，该架构超越了其前身的性能，并且至少比教师模型快 7 倍。我们全面的基准测试过程涵盖了下游任务，包括 ImageNet 分类、ADE20k 语义分割、COCO 对象检测和 LLaVa-1.5 框架。代码：https://github.com/NVlabs/RADIO 


---

## [Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/abs/2409.11355) 发表时间: 2024-09-17


最近的研究表明，大型扩散模型可以通过将深度估计视为图像条件图像生成任务，将其重新用作高精度单目深度估计器。虽然所提出的模型取得了最先进的结果，但由于多步推理导致的高计算需求限制了其在许多场景中的应用。在本文中，我们发现，感知到的低效率是由推理流程中的一个缺陷造成的，这个缺陷迄今为止一直未被注意到。通过简单的修改，即只使用扩散模型的第一步并固定模型的其余部分，我们获得了确定性深度估计器。固定的模型性能与之前报道的最佳配置相当，但速度提高了200多倍。为了优化下游任务性能，我们在单步模型的基础上使用特定于任务的损失进行了端到端的微调，并得到了一个确定性模型，该模型在常见的零样本基准测试中优于所有其他基于扩散的深度和法线估计模型。我们惊奇地发现，这种微调协议也直接适用于稳定扩散，并且达到了与当前最先进的基于扩散的深度和法线估计模型相当的性能，这对先前工作中得出的一些结论提出了质疑。 


---

## [GLC-SLAM: Gaussian Splatting SLAM with Efficient Loop Closure](https://arxiv.org/abs/2409.10982) 发表时间: 2024-09-17

三维高斯 splatting (3DGS) 因其在密集同步定位和建图 (SLAM) 中的应用而备受关注，它实现了实时渲染和高保真建图。然而，现有的基于 3DGS 的 SLAM 方法经常会遇到累积跟踪误差和地图漂移的问题，尤其是在大规模环境中。为了解决这些问题，我们引入了 GLC-SLAM，这是一个集成了相机姿态和场景模型全局优化的高斯 splatting SLAM 系统。我们的方法采用帧到模型跟踪，并使用全局到局部策略触发分层闭环，以最大程度地减少漂移累积。通过将场景划分为 3D 高斯子地图，我们可以在大型场景中轻松进行循环校正后的地图更新。此外，我们基于最小化不确定性的关键帧选择策略优先考虑观察到更有价值的 3D 高斯的关键帧，以增强子地图优化。在各种数据集上的实验结果表明，与最先进的密集 RGB-D SLAM 系统相比，GLC-SLAM 实现了更好或更具竞争力的跟踪和建图性能。 


---

## [Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2409.10594) 发表时间: 2024-09-16

Transformer 是现代深度学习的基石。传统上，这些模型依赖于多层感知机 (MLP) 层来混合通道之间的信息。在本文中，我们介绍了 Kolmogorov-Arnold Transformer (KAT)，这是一种新颖的架构，它用 Kolmogorov-Arnold Network (KAN) 层取代了 MLP 层，以增强模型的表达能力和性能。然而，将 KAN 集成到 Transformer 中并非易事，尤其是在扩展时。具体来说，我们确定了三个关键挑战：(C1) 基函数。KAN 中使用的标准 B 样条函数并未针对现代硬件上的并行计算进行优化，导致推理速度较慢。(C2) 参数和计算效率低下。KAN 需要为每个输入-输出对使用唯一的函数，这使得计算量非常大。(C3) 权重初始化。由于 KAN 的可学习激活函数对于在深度神经网络中实现收敛至关重要，因此其权重初始化特别具有挑战性。为了克服上述挑战，我们提出了三个关键解决方案：(S1) 有理基。我们用有理函数替换 B 样条函数，以提高与现代 GPU 的兼容性。通过在 CUDA 中实现这一点，我们实现了更快的计算。(S2) 组 KAN。我们通过一组神经元共享激活权重，以在不牺牲性能的情况下减少计算负载。(S3) 方差保持初始化。我们仔细初始化激活权重，以确保跨层保持激活方差。通过这些设计，KAT 可以有效地扩展，并且很容易优于传统的基于 MLP 的 Transformer。 


---

## [Improving 2D Feature Representations by 3D-Aware Fine-Tuning](https://arxiv.org/abs/2407.20229) 发表时间: 2024-07-29

目前的视觉基础模型仅仅在非结构化的二维数据上进行训练，这限制了它们对物体和场景三维结构的理解。在这项工作中，我们证明了在具有三维感知能力的数据上进行微调可以提高新兴语义特征的质量。我们设计了一种将二维语义特征提升为高效三维高斯表示的方法，该方法允许我们针对任意视角重新渲染这些特征。利用渲染后的具有三维感知能力的特征，我们设计了一种微调策略，将这种三维感知能力迁移到二维基础模型中。我们证明了以这种方式微调的模型所产生的特征可以通过简单的线性探测 readily 提高语义分割和深度估计等下游任务的性能。值得注意的是，尽管只在一个室内数据集上进行了微调，但这种改进可以迁移到各种室内数据集和域外数据集。我们希望我们的研究能够鼓励社区在训练二维基础模型时考虑注入三维感知能力。项目页面：https://ywyue.github.io/FiT3D. 


---

## [GS-Net: Generalizable Plug-and-Play 3D Gaussian Splatting Module](https://arxiv.org/abs/2409.11307) 发表时间: 2024-09-17

![Key Image](images/2409.11307_page_3_img_1.png)

3D高斯 splatting (3DGS)结合了基于图元的表示和体绘制技术的优势，实现了实时、高质量的渲染。然而，3DGS模型通常过度拟合单场景训练，并且对高斯椭球的初始化非常敏感，这些椭球是根据运动恢复结构(SfM)点云启发式地得到的，这限制了泛化性和实用性。为了解决这些限制，我们提出了GS-Net，一个可泛化的、即插即用的3DGS模块，它可以从稀疏的SfM点云中密集化高斯椭球，增强几何结构表示。据我们所知，GS-Net是第一个具有跨场景泛化能力的即插即用3DGS模块。此外，我们还介绍了CARLA-NVS数据集，该数据集包含额外的相机视角，可以全面评估重建和渲染质量。大量实验表明，将GS-Net应用于3DGS可以使传统视角的PSNR提高2.08 dB，新视角的PSNR提高1.86 dB，证实了该方法的有效性和鲁棒性。 


---


## [Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models](https://arxiv.org/abs/2409.10695) 发表时间: 2024-09-16

我们推出了最新的文本到图像模型 Playground v3 (PGv3)，该模型在多个测试基准上实现了最先进的 (SoTA) 性能，在图形设计能力方面表现出色，并引入了新的功能。与依赖于 T5 或 CLIP 文本编码器等预训练语言模型的传统文本到图像生成模型不同，我们的方法将大型语言模型 (LLM) 与一种新颖的结构完全集成，该结构仅利用来自仅解码器 LLM 的文本条件。此外，为了提高图像描述质量，我们开发了一种内部描述器，能够生成不同详细程度的描述，丰富了文本结构的多样性。我们还引入了一个新的基准测试 CapsBench 来评估详细的图像描述性能。实验结果表明，PGv3 在文本提示 adherence、复杂推理和准确的文本渲染方面表现出色。用户偏好研究表明，我们的模型在常见设计应用（如贴纸、海报和徽标设计）方面具有超人的图形设计能力。此外，PGv3 还引入了新功能，包括精确的 RGB 颜色控制和强大的多语言理解能力。 


---

## [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) 发表时间: 2024-06-12

高质量的指令数据对于对齐大型语言模型 (LLM) 至关重要。尽管某些模型（如 Llama-3-Instruct）具有开放权重，但它们的校准数据仍然是私有的，这阻碍了人工智能的民主化。高昂的人工成本以及有限的、预先定义的提示范围，阻碍了现有开源数据创建方法的有效扩展，这可能会限制公共校准数据集的多样性和质量。是否可以通过直接从校准的 LLM 中提取高质量的指令数据来大规模合成？我们提出了一种名为 Magpie 的自合成方法，用于生成大规模校准数据。我们的主要观察结果是，像 Llama-3-Instruct 这样的校准 LLM 可以生成用户查询，当我们仅输入左侧模板直到为用户消息保留的位置时，这要归功于它们的自回归性质。我们使用这种方法来提示 Llama-3-Instruct 并生成 400 万条指令及其相应的响应。我们对提取的数据进行了全面分析，并选择了 30 万个高质量的实例。为了将 Magpie 数据与其他公共指令数据集进行比较，我们使用每个数据集微调 Llama-3-8B-Base，并评估微调模型的性能。我们的结果表明，在某些任务中，使用 Magpie 微调的模型的性能与官方 Llama-3-8B-Instruct 相当，尽管后者通过监督微调 (SFT) 和后续反馈学习使用 1000 万个数据点进行了增强。我们还表明，仅将 Magpie 用于 SFT 可以超越之前用于 SFT 和偏好优化的公共数据集的性能，例如使用 UltraFeedback 进行直接偏好优化。这种优势在校准基准测试（如 AlpacaEval、ArenaHard 和 WildBench）中很明显。 


---

## [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103) 发表时间: 2024-03-26

近年来，生成工具的惊人发展为文本到图像生成和文本到视频生成等许多令人兴奋的应用提供了支持。这些生成工具背后的基本原理是扩散的概念，这是一种特殊的采样机制，它克服了以前方法中被认为难以克服的一些缺点。本教程的目标是讨论扩散模型的基本思想。本教程的目标受众包括有兴趣研究扩散模型或应用这些模型解决其他问题的本科生和研究生。 


---

## [UWStereo: A Large Synthetic Dataset for Underwater Stereo Matching](https://arxiv.org/abs/2409.01782) 发表时间: 2024-09-03

尽管立体匹配技术近期取得了进展，但其扩展到复杂的水下环境仍然 unexplored，这主要是由于：1) 水下图像能见度低、对比度低和其他不利影响；2) 难以获得用于训练深度学习模型的 ground truth 数据，即在水下环境中同时捕获图像并估计其相应的像素级深度信息。为了进一步推动水下立体匹配的研究，我们引入了一个名为 UWStereo 的大型合成数据集。我们的数据集包含 29,568 对合成立体图像对，以及密集且准确的左视图视差标注。我们设计了四个不同的水下场景，场景中充满了珊瑚、船舶和机器人等多样化的物体。我们还引入了相机模型、照明和环境效果方面的额外变化。与现有的水下数据集相比，UWStereo 在规模、变化、标注和照片级图像质量方面都更胜一筹。为了证实 UWStereo 数据集的有效性，我们与九种最先进的算法作为基准进行了全面评估。结果表明，当前的模型仍然难以泛化到新领域。因此，我们设计了一种新策略，在立体匹配训练之前学习重建跨域掩蔽图像，并集成跨视图注意力增强模块，该模块聚合远程上下文信息以增强泛化能力。 


---

## [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) 发表时间: 2021-03-25

本文提出了一种新的视觉Transformer，称为Swin Transformer，它可以作为计算机视觉的通用骨干网络。将Transformer从语言适应到视觉的挑战源于这两个领域之间的差异，例如视觉实体的规模变化很大，以及图像中像素的分辨率远远高于文本中的单词。为了解决这些差异，我们提出了一种层次化的Transformer，其表示是使用**移位窗口**计算的。移位窗口方案通过将自注意力计算限制在非重叠的局部窗口内来提高效率，同时也允许跨窗口连接。这种层次结构具有在各种尺度上建模的灵活性，并且相对于图像大小具有线性计算复杂度。Swin Transformer的这些特性使其能够兼容广泛的视觉任务，包括图像分类（ImageNet-1K上top-1准确率为87.3%）和密集预测任务，如目标检测（COCO test-dev上box AP为58.7，mask AP为51.1）和语义分割（ADE20K val上mIoU为53.5）。它的性能大大超过了之前的最佳水平，在COCO数据集上box AP和mask AP分别提升了+2.7和+2.6，在ADE20K数据集上mIoU提升了+3.2，证明了基于Transformer的模型作为视觉骨干网络的潜力。分层设计和移位窗口方法也被证明有利于所有MLP架构。代码和模型已公开发布在~\url{https://github.com/microsoft/Swin-Transformer}。 


---

## [MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry](https://arxiv.org/abs/2409.09479v1) 发表时间: 2024-09-14

我们提出了MAC-VO，一种新颖的基于学习的立体视觉里程计，它利用学习的度量感知匹配不确定性来实现双重目的：选择关键点和在位姿图优化中加权残差。与传统的几何方法优先考虑富含纹理的特征（如边缘）不同，我们的关键点选择器采用学习的不确定性，根据全局不一致性过滤掉低质量的特征。与为协方差矩阵建模尺度无关的对角线权重矩阵的基于学习的算法相比，我们设计了一种度量感知协方差模型来捕获关键点配准过程中的空间误差以及不同轴之间的相关性。将此协方差模型集成到位姿图优化中，增强了位姿估计的鲁棒性和可靠性，特别是在具有变化的光照、特征密度和运动模式的挑战性环境中。在公共基准数据集上，MAC-VO优于现有的VO算法，甚至在具有挑战性的环境中优于某些SLAM算法。协方差图还提供了有关估计位姿的可靠性的宝贵信息，这有利于自主系统的决策。 


---

