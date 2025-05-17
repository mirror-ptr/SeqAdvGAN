# SeqAdvGAN: 基于序列生成对抗网络的塔防AI注意力机制对抗研究

## ✨ 项目简介

本项目 **SeqAdvGAN** 专注于利用 **序列生成对抗网络 (SeqAdvGAN)** 对基于 **Transformer 架构** 的塔防游戏 **AI 决策模型 (Attentive-Trigger Network, ATN)** 的 **注意力机制** 进行 **对抗性攻击** 研究。我们训练一个 **生成器 CNN** 来生成针对五维图片序列输入的微小扰动，旨在误导 ATN 的决策，并深入分析和攻击其显式输出的序列注意力权重。

与传统图像对抗攻击不同，本项目聚焦于 **序列图像数据** 和具有 **注意力机制** 的复杂 AI 模型。我们不仅尝试改变最终决策，更探索如何 **操纵 ATN 的内部注意力聚焦区域**，这对于理解和攻击基于注意力的模型具有重要意义。

**关键词:** 对抗攻击, 对抗防御, 注意力机制, Transformer, 序列图像, GAN, AdvGAN, 3DCNN, 塔防AI, 注意力操纵

## 🚀 项目特性 (Features)

- ✨ **精细化序列对抗扰动生成:** 基于 **3D 卷积** 和 **3D Bottleneck** 构建生成器 CNN，针对五维 (Batch, Channels, Sequence, Height, Width) 图片序列生成高质量、隐蔽的对抗扰动。
- 🛡️ **灵活的 GAN 训练框架:** 采用类似 AdvGAN 的 GAN 框架训练，支持 **多种 GAN 损失类型 (BCE, LSGAN)**，并通过 **可插拔的判别器模块 (CNN Discriminator, PatchGAN Discriminator)** 增强框架灵活性，便于探索不同判别器结构对生成器训练的影响。
- 🎯 **创新的注意力感知攻击损失:** 设计并实现多种攻击损失函数，直接针对 ATN 的输出：
  - **决策损失:** 支持 **MSE** 和 **L1** 距离，以及 **KL/JS 散度** (概念框架)，旨在最大化对抗样本与原始样本在 ATN 最终决策图上的差异，支持 **区域性攻击**。
  - **注意力损失:** 支持 **MSE, L1, L2/L-inf 差异** 和 **余弦相似度** 等度量。特别引入 **Top-K 注意力损失**，直接攻击原始样本中 Top-K 重要的注意力连接，强制 ATN 将注意力转移到不重要的区域，或降低关键区域的注意力权重。
- ⚖️ **全面的正则化损失:** 集成 **L2 范数惩罚** 和 **Total Variation (TV) Loss**，约束扰动的大小和空间平滑性，提高对抗样本的隐蔽性。
- 📊 **详细的评估体系:** 实现多种攻击成功标准 (**MSE 差异阈值, 平均值变化, Top-K 值下降, Top-K 位置变化**)，以及 **感知评估指标 (PSNR, SSIM, LPIPS 框架)**，从多个角度衡量攻击效果和扰动质量。
- 📈 **丰富的可视化工具:** 提供可视化原始序列、对抗样本、扰动、ATN 决策图（含差异可视化）和 **ATN 注意力热力图（原始 vs. 对抗 vs. 差异）** 的功能，帮助直观理解攻击效果。
- 📦 **模块化设计:** 代码结构清晰，各模块（模型、损失、工具、脚本）职责分明，易于扩展和修改。

## 🏗️ 项目结构 (Project Structure)

```
. 项目根目录
├── data/                             # 数据集（模拟数据或真实数据）存放及加载脚本
├── models/                           # 模型定义
│   ├── generator_cnn.py              # 序列生成器 CNN
│   ├── discriminator_cnn.py          # CNN 判别器
│   ├── discriminator_patchgan.py     # PatchGAN 判别器
│   ├── bottleneck3d.py               # 3D Bottleneck 模块
│   └── atn_model/                    # 存放 ATN 模型文件 (需自行提供)
├── losses/                           # 损失函数定义
│   ├── gan_losses.py                 # GAN 对抗损失
│   ├── attack_losses.py              # 攻击损失 (决策, 注意力)
│   └── regularization_losses.py      # 正则化损失 (L-inf, L2, TV)
├── utils/                            # 辅助工具
│   ├── atn_utils.py                  # ATN 模型加载和输出获取
│   ├── data_utils.py                 # 数据加载和预处理 (包含模拟和真实数据加载框架)
│   ├── vis_utils.py                  # 可视化工具
│   └── eval_utils.py                 # 评估指标计算
├── scripts/                          # 运行脚本
│   ├── train_generator.py            # 训练生成器和判别器主脚本
│   ├── evaluate_attack.py            # 独立评估攻击效果脚本
│   ├── train_defense.py              # [TODO] 对抗训练防御脚本
├── README.md                         # 项目说明文件 (当前文件)
├── requirements.txt                  # 项目依赖库列表
└── ...                               # 其他项目文件
```

## 🛠️ 环境配置 (Environment Setup)

本项目建议在支持 GPU 加速的 Linux 环境（推荐 WSL2 + CUDA）下运行。请按照以下步骤配置环境。

1.  **安装 WSL2 和 NVIDIA 显卡驱动/WSL2 CUDA Toolkit:**
    参考 [NVIDIA 官方文档](https://developer.nvidia.com/cuda-downloads) 为您的 WSL2 环境安装与您 GPU 和驱动兼容的 CUDA Toolkit。

2.  **安装 Miniconda 或 Anaconda:**
    访问 [Miniconda 官网](https://www.anaconda.com/docs/getting-started/miniconda/install) 下载并安装适合您系统的 Miniconda。

3.  **创建并激活 conda 环境:**
    ```bash
    conda create -n seqadvgan_env python=3.9 # 示例使用 Python 3.9
    conda activate seqadvgan_env
    ```

4.  **安装 PyTorch 及其他依赖:**
    在已激活的 `seqadvgan_env` 环境中，**务必**访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)，根据您的 CUDA 版本、操作系统等选择正确的配置，复制生成的命令并执行。
    示例命令 (请以官网为准): `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
    安装其他依赖：
    ```bash
    pip install matplotlib opencv-python tensorboard piq numpy tqdm
    ```
    *(注: `piq` 用于计算感知指标 LPIPS/SSIM, `tqdm` 用于显示进度条)*

5.  **克隆项目仓库:**
    ```bash
    git clone https://github.com/mirror-ptr/SeqAdvGAN.git
    cd SeqAdvGAN
    ```

6.  **获取 ATN 模型:**
    请将您拥有的 ATN 模型文件 (权重 `.pth`, 架构定义等) 放置到 `models/atn_model/` 目录下，并确保 `utils/atn_utils.py` 中的 `load_atn_model` 函数能正确加载您的模型。

7.  **生成 requirements.txt (可选):**
    ```bash
    pip freeze > requirements.txt
    ```

## 🏃 使用 (Usage)

激活您的 conda 环境 (`conda activate seqadvgan_env`) 后，您可以使用以下脚本：

### 🏋️‍ 训练序列生成器和判别器 (`scripts/train_generator.py`)

此脚本用于训练 SeqAdvGAN 的生成器和判别器。您可以通过命令行参数配置训练过程、损失函数类型和权重、模型结构等。

主要参数示例：

```bash
python scripts/train_generator.py \
    --epochs 100 \              # 训练轮数
    --batch_size 4 \            # 训练批次大小 (根据显存调整，常见 Killed 错误原因)
    --lr_g 0.0002 \             # 生成器学习率
    --lr_d 0.0002 \             # 判别器学习率
    --epsilon 0.03 \            # L-infinity 扰动上限
    --attack_loss_weight 10.0 \ # 攻击损失权重
    --reg_loss_weight 1.0 \     # 正则化损失权重 (如 L2)
    --tv_loss_weight 0.1 \      # Total Variation Loss 权重 (用于平滑扰动)
    --attention_loss_weight 1.0 \ # 攻击损失中注意力损失的权重
    --decision_loss_weight 1.0 \  # 攻击损失中决策损失的权重
    --decision_loss_type mse \    # 决策损失类型 (mse, l1, kl, js)
    --attention_loss_type topk \  # 注意力损失类型 (mse, l1, kl, js, topk)
    --topk_k 5 \                # Top-K 注意力损失中 K 的值
    --discriminator_type patchgan \ # 判别器类型 (cnn, patchgan)
    --gan_loss_type lsgan \       # GAN Loss 类型 (bce, lsgan)
    --log_dir runs/my_experiment \ # TensorBoard 日志保存目录
    --checkpoint_dir checkpoints \ # 模型检查点保存目录
    --save_interval 10 \        # 每隔多少 epoch 保存检查点
    --eval_interval 5 \         # 每隔多少 epoch 进行一次评估
    --num_mock_samples 1000 \   # 使用模拟数据时的样本数量 (未来切换为真实数据参数)
    --atn_model_path /path/to/your/atn_model.pth \ # ATN 模型文件路径
    # ... 更多参数请参考脚本内部 argparse 定义
```

**注意:** 如果遇到 `Killed` 错误，通常是显存或内存不足，请 **显著减小 `--batch_size`**。


### 📊 独立评估攻击效果 (`scripts/evaluate_attack.py`)

此脚本用于加载训练好的生成器模型，并在指定数据集上评估攻击效果。

主要参数示例：

```bash
python scripts/evaluate_attack.py \
    --generator_weights_path checkpoints/checkpoint_epoch_50.pth \ # 训练好的生成器权重路径
    --atn_model_path /path/to/your/atn_model.pth \ # ATN 模型文件路径
    --batch_size 8 \             # 评估批次大小
    --eval_success_criterion topk_position_change \ # 评估时攻击成功标准
    --eval_success_threshold 0.1 \ # 评估成功阈值 (具体含义取决于 success_criterion)
    --topk_k 10 \                # 评估时 Top-K 相关的 K 值
    # 未来添加真实数据加载参数: --data_path /path/to/real_data.npy
    # ... 更多参数请参考脚本内部 argparse 定义
```


### 📈 可视化结果 (`scripts/visualize_results.py`)

此脚本用于加载训练好的生成器权重，生成对抗样本，并使用 TensorBoard 可视化原始/对抗样本、扰动、决策图和注意力热力图。

主要参数示例：

```bash
python scripts/visualize_results.py \
    --generator_weights_path checkpoints/checkpoint_epoch_50.pth \ # 训练好的生成器权重路径
    --atn_model_path /path/to/your/atn_model.pth \ # ATN 模型文件路径
    --num_vis_samples 8 \        # 可视化样本数量
    --batch_size 8 \             # 可视化批次大小
    --sequence_step_to_vis 0 \   # 可视化哪个序列步骤的特征/扰动
    --num_vis_heads 1 \          # 可视化多少个注意力头
    # 未来添加真实数据加载参数: --data_path /path/to/real_data.npy
    # ... 更多参数请参考脚本内部 argparse 定义
```

**可视化示例 (Placeholder):**

*(请在此处插入您生成的 TensorBoard 截图，例如原始/对抗特征、扰动、决策图对比、注意力热力图对比等)*

![Visualization Example 1]()
![Visualization Example 2]()

## 📊 评估指标 (Evaluation Metrics)

`utils/eval_utils.py` 中实现了用于衡量攻击效果的多种指标：

- **攻击成功率:**
  - `mse_diff_threshold`: 对抗样本与原始样本决策图的 **MSE 差异** 大于指定阈值即视为攻击成功。
  - `mean_change_threshold`: 决策图的 **平均值变化** 大于指定阈值（考虑原始决策图是越高越好还是越低越好）。
  - `topk_value_drop`: 原始决策图 **Top-K 位置的值** 在对抗样本中 **平均下降** 超过指定阈值即视为攻击成功。
  - `topk_position_change`: 原始决策图的 **Top-K 索引位置集合** 与对抗样本的 **Top-K 索引位置集合** 不同即视为攻击成功。
- **扰动大小:** **L-infinity 范数** 和 **L2 范数**，衡量扰动的可见度和强度。
- **感知质量:** **PSNR**, **SSIM** 和 **LPIPS** (框架已搭建，依赖 `piq` 库)，用于衡量对抗样本与原始样本在视觉感知上的差异，确保扰动的隐蔽性。


## 💡 未来工作 (Future Work)

- [ ] 实现更高级的区域性攻击策略。
- [ ] 实现基于优化的攻击方法 (如 PGD) 进行对比实验。
- [ ] 完善真实数据集的加载和预处理。
- [ ] 实施和评估不同的对抗性训练防御策略。
- [ ] 在真实塔防游戏环境中测试攻击和防御效果。
- [ ] 探索其他针对 Transformer 注意力机制的攻击方法。

## 🤝 贡献 (Contributing)

欢迎对本项目做出贡献！如果您有任何想法、建议或遇到了问题，请提交 issue 或 pull request。

## ❤️ 致谢

感谢所有为本项目提供帮助的朋友和社区。
