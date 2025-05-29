# SeqAdvGAN: 基于序列生成对抗网络的塔防AI注意力机制对抗研究

## ✨ 项目简介

本项目 **SeqAdvGAN** （**Seq**uence **Adv**ersarial **G**enerative **A**dversarial **N**etwork）专注于利用 **序列生成对抗网络** 对基于 **Transformer 架构** 的塔防游戏 **AI 决策模型 (Attentive-Trigger Network, ATN)** 的 **注意力机制** 进行 **对抗性攻击** 研究。

我们已完成了第一阶段（特征层攻击）的基础工作，目前项目正聚焦于 **第二阶段（像素级攻击）** 的训练和调试。我们核心目标是训练一个 **生成器 CNN** 来生成针对五维 (Batch, Channels, Sequence, Height, Width) 图片序列输入的微小像素扰动，旨在误导预训练且冻结的 ATN 的决策。整个攻击过程主要分为两个阶段：

* **阶段一 (Phase 1)：特征层攻击 (已完成)**
    * 在这一阶段，我们着重于攻击 ATN 的**特征提取层**。通过生成精心设计的微小扰动，我们致力于最大化对抗样本在 ATN 特征提取器中产生的特征表示与原始样本特征表示之间的差异，从而使 ATN 无法正确识别原始的特征模式。
* **阶段二 (Phase 2)：像素级攻击及决策图、注意力机制攻击 (当前重点)**
    * 在成功完成第一阶段的基础上，我们将进一步探索如何直接操纵 ATN 的**最终决策图和显式序列注意力权重**。**本项目当前的工作重点集中在通过生成像素级的对抗扰动来实现对 ATN 决策和注意力机制的攻击。** 这一阶段将引入针对这些特定输出的攻击损失，旨在更精准地控制 AI 的决策逻辑和其注意力聚焦区域。

与传统的图像对抗攻击不同，本项目聚焦于 **序列图像数据** 和具有 **注意力机制** 的复杂 AI 模型。我们不仅尝试改变最终决策，更深入探索如何 **操纵 ATN 的内部注意力聚焦区域**，这对于理解、评估和增强基于注意力的模型在对抗环境下的鲁棒性具有重要意义。

**关键词:** 对抗攻击, 对抗防御, 注意力机制, Transformer, 序列图像, GAN, AdvGAN, 3DCNN, 塔防AI, 注意力操纵

## 🚀 项目特性 (Features)

本项目具备以下关键特性：

* **✨ 精细化序列对抗扰动生成:**
    * 基于 **3D 卷积** 和 **3D Bottleneck** 构建生成器 CNN，能够针对五维图片序列 (Batch, Channels, Sequence, Height, Width) 生成高质量、隐蔽的对抗扰动。
* **🛡️ 灵活的 GAN 训练框架:**
    * 采用类似 AdvGAN 的生成对抗网络框架进行训练，支持 **多种 GAN 损失类型 (如 BCE、LSGAN)**。
    * 通过 **可插拔的判别器模块 (CNN Discriminator, PatchGAN Discriminator)** 设计，增强框架灵活性，便于探索不同判别器结构对生成器训练的影响。
* **🎯 创新的分阶段攻击策略:**
    * **阶段一 (Phase 1): 特征层攻击:** 通过在 ATN 特征提取层最大化对抗样本与原始样本的特征输出差异来误导模型，旨在使 ATN 无法识别原始的特征模式。
    * **阶段二 (Phase 2): 决策图和注意力机制攻击:** 在第一阶段成功的基础上，进一步引入针对 ATN 最终决策图和序列注意力矩阵的攻击损失，以直接操纵 AI 的决策和其内部注意力聚焦。
* **⚖️ 全面的正则化损失:**
    * 集成 **L2 范数惩罚** 和 **Total Variation (TV) Loss**，有效约束生成扰动的大小和空间平滑性，从而提高对抗样本的隐蔽性和自然度。
* **📊 详细的评估体系:**
    * 实现多种攻击成功标准，包括：
        * **MSE 差异阈值:** 对抗样本与原始样本在目标输出（特征、决策或注意力）上的均方误差差异是否达到预设阈值。
        * **平均值变化:** 目标输出的平均值变化是否达到预设阈值。
        * **Top-K 值下降:** 原始 Top-K 位置上的值在对抗样本中是否显著下降。
        * **Top-K 位置变化:** 原始 Top-K 关注区域的索引位置集合与对抗样本的 Top-K 索引位置集合是否发生变化。
    * 支持 **感知评估指标 (PSNR, SSIM, LPIPS 框架)**，从视觉感知角度衡量对抗样本的质量和隐蔽性。在第一阶段，这些指标基于将特征层通过特定函数转换得到的"伪图像"计算，其解读需结合转换逻辑。
* **📈 丰富的可视化工具:**
    * 提供强大的可视化功能，通过 TensorBoard 直观展示原始序列、对抗样本、生成的扰动、ATN 决策图（含差异可视化）和 **ATN 注意力热力图（原始、对抗、差异对比）**，帮助深入理解攻击效果。
* **📦 模块化设计:**
    * 代码结构清晰，各模块（模型、损失、工具、运行脚本）职责分明，降低耦合度，易于理解、扩展和修改，便于团队协作和项目迭代。

## 📂 项目结构 (Project Structure)

以下是 SeqAdvGAN 项目的主要目录和文件结构：

```
SeqAdvGAN/
├── configs/                  # 训练和模型配置目录
│   ├── stage1_config.yaml    # 第一阶段训练配置
│   └── stage2_config.yaml    # 第二阶段训练配置 (当前训练使用)
├── data/                     # 数据文件存放目录 (例如视频文件)
├── doc/                      # 文档目录 (例如集成与调试总结)
├── IrisArknights/            # 子模块：塔防游戏相关工具库 (例如 Level 解析, 图像变换)
├── IrisBabel/                # 子模块：预训练的 ATN 模型库及其组件
├── losses/                   # 损失函数定义目录
│   ├── attack_losses.py      # 攻击损失函数
│   ├── gan_losses.py         # GAN 损失函数
│   └── regularization_losses.py # 正则化损失函数
├── models/                   # 模型定义目录
│   ├── generator_cnn.py      # 生成器模型
│   ├── discriminator_cnn.py  # CNN 判别器模型
│   ├── discriminator_patchgan.py # PatchGAN 判别器模型
│   ├── bottleneck3d.py       # 3D Bottleneck 模块定义
│   └── layers.py             # 其他自定义层 (例如 Lambda)
├── resources/                # 资源文件目录 (例如关卡 JSON 文件)
├── runs/                     # 训练运行输出目录 (TensorBoard 日志, 检查点等)
├── scripts/                  # 运行脚本目录
│   ├── train_generator_stage2.py # 第二阶段主训练脚本 (当前使用)
│   ├── train_generator.py    # (旧) 主训练脚本
│   ├── evaluate_attack.py    # 攻击效果评估脚本
│   └── visualize_results.py  # 结果可视化脚本
├── utils/                    # 辅助工具函数目录
│   ├── atn_utils.py          # ATN 模型加载和输出获取工具
│   ├── config_utils.py       # 配置加载工具
│   ├── data_utils.py         # 数据加载工具 (GameVideoDataset)
│   ├── eval_utils.py         # 评估指标计算工具
│   └── vis_utils.py          # TensorBoard 可视化工具
├── .gitattributes            # Git 属性配置
├── .gitignore                # Git 忽略文件配置
├── requirements.txt          # Python 依赖列表
└── README.md                 # 项目说明文档

```

## 🚀 快速开始 (Quick Start)

按照以下步骤，快速设置环境并开始您的 SeqAdvGAN 训练。

### 1. 环境准备

1.  **安装 WSL2 和 NVIDIA 显卡驱动/WSL2 CUDA Toolkit:**
    * 如果使用 WSL2 环境，请参考 [NVIDIA 官方文档](https://developer.nvidia.com/cuda-downloads) 为您的 WSL2 环境安装与您 GPU 和驱动兼容的 CUDA Toolkit。
2.  **安装 Miniconda 或 Anaconda:**
    * 访问 [Miniconda 官网](https://www.anaconda.com/docs/getting-started/miniconda/install) 下载并安装适合您操作系统的 Miniconda。
3.  **创建并激活 conda 环境:**
    ```bash
    conda create -n seqadvgan_env python=3.9 # 示例使用 Python 3.9
    conda activate seqadvgan_env
    ```
4.  **安装 PyTorch 及其他依赖:**
    * 在已激活的 `seqadvgan_env` 环境中，**务必**访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)，根据您的 CUDA 版本、操作系统等选择正确的配置，复制生成的命令并执行。
        * **示例命令 (请以官网为准，确保 CUDA 版本匹配):**
            ```bash
            conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
            ```
    * 安装其他必要的 Python 依赖：
        ```bash
        pip install matplotlib opencv-python tensorboard piq numpy tqdm pyyaml easydict
        ```
        *(注: `piq` 用于计算感知指标 LPIPS/SSIM, `tqdm` 用于显示进度条)*
        *（提示: `piq` 库可能存在版本兼容性问题，建议查阅其文档或尝试已测试版本，例如 0.8.0 或更高版本，以确保评估指标能正常计算。）*
    **重要提示：** 请确保以上所有依赖库都安装在您将用于运行训练和 TensorBoard 的同一个 **conda/虚拟环境** 中。激活环境后，通常可以直接使用 `python` 或已安装的可执行文件（如 `tensorboard`）。如果遇到 `ModuleNotFoundError` 或找不到命令的问题，这通常意味着您当前使用的 Python 解释器或命令路径不正确。此时，尝试使用虚拟环境中 Python 解释器的完整路径来运行脚本或模块（例如，`/path/to/your/conda/envs/your_env/bin/python your_script.py` 或 `/path/to/your/conda/envs/your_env/bin/tensorboard --logdir ...`）。
5.  **克隆项目仓库:**
    ```bash
    git clone https://github.com/mirror-ptr/SeqAdvGAN.git
    cd SeqAdvGAN
    ```

### 2. 数据准备

1.  将您的塔防游戏视频文件放置在 `data/videos/` 目录下（例如 `data/videos/1392e8264d9904bfffe142421b784cbd.mp4`）。
2.  将对应的关卡 JSON 文件放置在 `resources/level/` 目录下（例如 `resources/level/hard_15-01-obt-hard-level_hard_15-01.json`）。
3.  您可以通过修改 `configs/stage1_config.yaml` 中的 `data.video_path` 和 `data.level_json_path` 来指定数据路径。
4.  **（可选）使用模拟数据：** 如果您希望在没有真实数据的情况下进行测试，可以在 `configs/stage1_config.yaml` 中将 `data.use_mock_data` 设置为 `True`，并调整 `data.mock_num_samples`。

### 3. 开始训练 (第一阶段：特征层攻击)

第一阶段的基础集成和部分训练已完成。如需运行，请使用以下命令：

```bash
python scripts/train_generator.py --config configs/stage1_config.yaml
```

训练监控: 训练过程中的日志、模型检查点和 TensorBoard 可视化数据将保存到 runs/ 和 checkpoints/ 目录下（可在配置文件中调整）。

实时查看训练进度: 您可以使用 TensorBoard 实时监控训练进度和可视化结果。在另一个终端中，导航到您的项目根目录，然后运行：
 
```bash
tensorboard --logdir runs/stage1_train # 或在 config 中设置的 logging.log_dir
```

**重要提示：** 如果直接运行 `tensorboard` 命令遇到问题（例如 `ModuleNotFoundError`），请尝试使用虚拟环境中可执行文件的完整路径，例如：
```bash
/home/xianke/miniconda3/envs/seqadvgan_env/bin/tensorboard --logdir runs/stage1_train # 将路径替换为您实际的环境路径
```

### 开始训练 (第二阶段：像素级攻击)

目前项目的重点是进行第二阶段的训练。运行以下命令开始 SeqAdvGAN 的第二阶段训练。请确保 `configs/stage2_config.yaml` 配置正确，特别是 ATN 模型路径指向完整的 ATN 模型权重。

```bash
python scripts/train_generator_stage2.py --config configs/stage2_config.yaml
```
第二阶段的训练目标是攻击 ATN 的决策图和注意力机制，相关的损失函数和评估指标权重需要在 `configs/stage2_config.yaml` 中配置（例如 `losses.decision_loss_weight` 和 `losses.attention_loss_weight` ）。请注意，训练脚本为 `train_generator_stage2.py`。

**实时查看第二阶段训练进度:**
```bash
tensorboard --logdir runs/stage2_train # 或在 config 中设置的 logging.log_dir for stage2
```

### 4. 训练参数配置

训练参数可以在 `configs/stage2_config.yaml` 或 `configs/stage1_config.yaml` 中进行详细配置，具体取决于您运行的训练阶段。常用的可配置项包括：

*   `training.num_epochs`: 训练的总 epoch 数量。
*   `training.batch_size`: 批处理大小 (根据您的 GPU 显存调整，过大可能导致 OOM 错误)。
*   `training.lr_g`, `training.lr_d`: 生成器和判别器的学习率。
*   `model.generator.epsilon`: L-inf 范数约束的最大扰动值，控制扰动强度（主要用于第二阶段）。
*   `losses.decision_loss_weight`: 攻击损失的权重。在第一阶段用于特征层差异，在第二阶段用于 TriggerNet 输出差异。
*   `losses.attention_loss_weight`: 注意力攻击损失的权重（主要用于第二阶段，如果实现）。
*   `regularization.lambda_l_inf`, `lambda_l2`, `lambda_tv`, `lambda_l2_penalty`: 各种正则化项的权重，用于约束扰动质量。

**重要提示：** 请确保您使用的配置文件（例如 `configs/stage2_config.yaml`）包含了脚本所需的所有必要参数，特别是：
- ATN 模型加载相关的参数（在 `model.atn` 部分）。
- 数据加载相关的参数（在 `data` 部分）。
- 评估相关的参数（在 `evaluation` 部分）。
缺失或错误的参数可能导致 `AttributeError` 或其他运行时错误。

更多详细配置请参考相应的配置文件。

**注意:** 如果训练过程中遇到 `Killed` 错误，通常是显存或内存不足，请 **显著减小 `--batch_size`**。

### 📈 可视化结果 (`scripts/visualize_results.py`)

此脚本用于加载训练好的生成器权重，生成对抗样本，并使用 TensorBoard 将原始/对抗样本、扰动、决策图和注意力热力图进行可视化，便于直观分析攻击效果。

常用命令行参数示例:

```bash
python scripts/visualize_results.py \
    --generator_weights_path checkpoints/stage1/epoch_50_G.pth \
    --model.atn.model_path models/feature_heads/iris_feature_13_ce_4bn_20f_best.pth \
    --training.train_stage 1 \
    --visualization.num_vis_samples 8 \
    --training.batch_size 8 \
    --visualization.sequence_step_to_vis 0 \
    --visualization.num_vis_heads 1 \
    --data.use_mock_data False \
    --data.video_path data/videos/test_video.mp4 \
    --logging.log_dir runs/visualization_output
    # ... 更多参数请参考脚本内部 argparse 定义或配置文件
```

## 可视化示例 (Placeholder)

(请在此处插入您在 TensorBoard 中生成的实际可视化截图。例如，展示原始视频帧、生成的扰动、施加扰动后的对抗帧，以及它们通过 ATN 后的特征图、决策图对比。在第一阶段，主要关注特征图的差异。如果您 ATN 模型也能输出注意力图，即使不是主要攻击目标，也可以展示其变化。)

## 📊 评估指标 (Evaluation Metrics)

`utils/eval_utils.py` 中实现了用于衡量攻击效果的多种评估指标：

*   **攻击成功率:**
    *   **阶段一 (特征层攻击):** 我们主要关注 `Attack_Success_Rate_Feature_Stage1`，它衡量对抗样本通过 ATN 特征提取器后，其特征表示与原始特征表示之间的差异是否达到预设阈值。
    *   **通用成功标准 (可配置):**
        *   `mse_diff_threshold`: 对抗样本与原始样本决策图（或此处为特征图）的 **MSE 差异** 大于指定阈值即视为攻击成功。
        *   `mean_change_threshold`: 决策图（或特征图）的 **平均值变化** 大于指定阈值（考虑原始决策图是越高越好还是越低越好）。
        *   `topk_value_drop`: 原始决策图 **Top-K 位置的值** 在对抗样本中 **平均下降** 超过指定阈值即视为攻击成功（主要针对阶段二）。
        *   `topk_position_change`: 原始决策图的 **Top-K 索引位置集合** 与对抗样本的 **Top-K 索引位置集合** 不同即视为攻击成功（主要针对阶段二）。
*   **扰动大小:**
    *   **L-infinity 范数 (`Linf_Norm_Avg`)** 和 **L2 范数 (`L2_Norm_Avg`)**，用于量化生成扰动的隐蔽性和强度，确保其不易被人眼察觉。
*   **判别器分数:**
    *   `Discriminator_Real_Score_Avg` 和 `Discriminator_Fake_Score_Avg`，反映判别器的训练情况和生成对抗样本的真实性。
*   **感知质量指标 (可选):**
    *   `PSNR` (`PSNR_Avg`), `SSIM` (`SSIM_Avg`) 和 `LPIPS` (`LPIPS_Avg`) （框架已搭建，依赖 `piq` 库），用于衡量对抗样本与原始样本在视觉感知上的差异，进一步确保扰动的隐蔽性。

**请注意，在第一阶段的训练和评估中，主要关注的是对 ATN 特征层的攻击效果，因此 `Attack_Success_Rate_Feature_Stage1` 是核心评估指标。**

**在第二阶段的训练和评估中，主要关注的是对 ATN TriggerNet 输出（决策）的像素级攻击效果，核心评估指标包括 `Attack_Success_Rate_TriggerNet` 以及衡量扰动隐蔽性的指标。**

## 💡 未来工作 (Future Work)

我们对 SeqAdvGAN 项目的未来发展抱有以下设想和计划：

*   [ ] 实现更高级的区域性攻击策略： 探索针对特定区域或对象的扰动生成方法。
*   [ ] 实现基于优化的攻击方法 (如 PGD) 进行对比实验： 引入非 GAN 类的攻击方法作为基线，全面评估 SeqAdvGAN 的效果。
*   [ ] 完善真实数据集的加载和预处理： 增强数据处理的通用性和鲁棒性，以适应更复杂的视频数据。
*   [ ] 实施和评估不同的对抗性训练防御策略： 研究并集成防御机制，提升 ATN 模型在对抗环境下的鲁棒性。
*   [ ] 在真实塔防游戏环境中测试攻击和防御效果： 将 SeqAdvGAN 集成到实际游戏流程中，验证其在实际应用场景下的有效性。
*   [ ] 探索其他针对 Transformer 注意力机制的攻击方法： 深入研究 Transformer 特性，开发更具针对性的攻击策略。

## 🤝 贡献 (Contributing)

我们非常欢迎对本项目做出贡献！如果您有任何想法、建议或遇到了问题，请随时通过以下方式联系我们或参与：

*   提交 [Issue](link_to_your_issues) 报告 Bug 或提出功能请求。
*   提交 [Pull Request](link_to_your_pull_requests) 贡献您的代码或改进。

## ❤️ 致谢

衷心感谢所有为本项目提供帮助、指导和支持的朋友、同事以及开源社区的成员。
