# SeqAdvGAN: 基于序列生成对抗网络的塔防AI注意力机制对抗研究

## 项目简介

本项目深入研究针对Transformer架构的塔防游戏AI决策模型（Attentive-Trigger Network, ATN）的对抗性攻击与防御方法。我们利用序列生成对抗网络（SeqAdvGAN）训练一个生成器CNN，该网络能够接收塔防游戏画面的五维图片序列输入 (Batch Size, Sequence Length, Width, Height, Channels)，并生成微小的、人眼难以察觉的同等形状扰动。将扰动添加到原始图片序列上形成对抗样本，旨在误导ATN模型做出错误的决策，并特别关注攻击如何影响ATN模型显式输出的序列注意力机制。项目还将研究并实现对抗性训练等防御策略，提升ATN模型对该攻击的鲁棒性。

本项目的研究内容属于“图对抗研究”范畴，专注于对处理序列图像数据和包含注意力机制的复杂决策AI进行对抗性分析。

**关键词:** 对抗攻击, 对抗防御, 注意力机制, Transformer, 序列图像, GAN, AdvGAN, 3DCNN, 塔防AI

## 项目特性 (Features)

* **序列对抗样本生成:** 实现基于3D卷积和3D Bottleneck的序列生成器CNN，生成针对五维图片序列的对抗扰动。
* **基于GAN的训练框架:** 采用类似AdvGAN的生成对抗网络框架训练生成器，结合判别器确保扰动的隐蔽性。
* **复合损失函数:** 设计并实现包含基于ATN序列注意力机制的损失、基于ATN决策输出的损失和L-infinity范数扰动约束的复合损失函数。
* **对抗性训练防御:** 研究并实现基于PGD等方法的对抗性训练，提升ATN模型鲁棒性。
* **可视化分析:** 开发工具可视化原始序列、对抗样本、扰动以及ATN在序列上的注意力权重分布。
* **在特定塔防游戏AI上验证:** 在实际的ATN模型上进行攻击和防御实验，使用游戏任务特定指标（如“有效操作”次数）评估效果。

## 环境配置 (Environment Setup)

本项目建议在支持GPU加速的Linux环境（推荐WSL2 + CUDA）下运行。以下是详细配置步骤，针对CUDA版本11.7：

1.  **安装 Windows Subsystem for Linux (WSL2):**
    确保你的Windows版本支持WSL2。打开PowerShell或CMD并运行：
    ```bash
    wsl --install
    ```
    如果已经安装，请确保是WSL2版本：
    ```bash
    wsl --set-version <你的Linux发行版名称> 2
    ```
    选择并安装一个Linux发行版（如Ubuntu）。

2.  **安装 NVIDIA 显卡驱动和 WSL2 CUDA Toolkit:**
    确保你的Windows主机安装了与WSL2兼容的最新NVIDIA驱动。
    在WSL2中安装与PyTorch版本和你的驱动兼容的CUDA Toolkit。针对CUDA 11.7，请参考NVIDIA官方文档中的“WSL2”部分获取详细安装步骤和命令：
    [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
    **务必选择适用于WSL-Ubuntu的CUDA 11.7版本。**

3.  **安装 Miniconda 或 Anaconda:**
    推荐使用conda进行Python环境管理，可以避免库冲突。
    在WSL终端中下载并运行Miniconda安装脚本（推荐，更轻量）：
    [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
    按照提示完成安装。安装完成后关闭并重新打开WSL终端。

4.  **创建并激活 conda 环境:**
    创建一个新的conda环境用于本项目：
    ```bash
    conda create -n seqadvgan_env python=3.9 # 示例使用 Python 3.9
    conda activate seqadvgan_env
    ```

5.  **安装 PyTorch 及其他依赖:**
    在已激活的conda环境中，根据你的CUDA版本（11.7）和操作系统，从PyTorch官方网站获取安装命令。
    **请访问 PyTorch 官方网站 ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))，选择正确的配置（PyTorch Build, Your OS, Package, Language, CUDA），然后复制生成的命令。**
    示例命令（请**务必**从官网获取最新和最匹配的命令）：
    ```bash
    # 示例：安装 PyTorch 1.13 (兼容 CUDA 11.7)
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch -c conda-forge
    ```
    安装其他必要的库：
    ```bash
    pip install numpy opencv-python matplotlib
    # 如果需要其他库，例如用于SSIM或LPIPS计算，请另行安装
    ```

6.  **克隆项目仓库:**
    ```bash
    git clone [你的项目仓库URL，创建仓库后填写]
    cd [你的项目文件夹]
    ```

7.  **获取 Bottleneck3d 代码:**
    将朋友提供的 `Bottleneck3d.py` 文件放到项目代码中合适的位置（例如 `models/bottleneck3d.py`）。

8.  **获取 ATN 模型:**
    将ATN模型文件（权重和/或架构定义）放到项目代码中合适的位置（例如 `models/atn_model/`），并确保你有加载该模型的代码和权限。

## 安装 (Installation)

本项目依赖于上述环境配置中安装的库。克隆仓库并完成环境配置后即可直接运行代码。你可以生成 `requirements.txt` 文件以便其他人复现环境：

```bash
pip freeze > requirements.txt
