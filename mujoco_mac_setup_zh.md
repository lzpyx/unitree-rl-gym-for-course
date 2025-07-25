# macOS MuJoCo 安装指南

本指南提供在 macOS 系统上设置 MuJoCo 与 unitree_rl_gym 的详细步骤说明。

---

## 目录
1. [创建虚拟环境](#1-创建虚拟环境)
2. [安装依赖项](#2-安装依赖项)
3. [运行 MuJoCo](#3-运行-mujoco)
4. [故障排除](#4-故障排除)

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序。推荐使用 Conda 创建虚拟环境。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适合创建和管理虚拟环境。

#### 1.1.1 检查您的 Mac 架构

首先，检查您的 Mac 是 Intel 还是 Apple Silicon：

```bash
uname -m
```

- 如果显示 `x86_64`：Intel Mac
- 如果显示 `arm64`：Apple Silicon Mac

#### 1.1.2 下载适用于 macOS 的 Miniconda

**适用于 Apple Silicon Mac (M1/M2/M3)：**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

**适用于 Intel Mac：**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

**替代方案：通过 Homebrew 安装（推荐）**
```bash
# 如果尚未安装 Homebrew，请先安装
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 添加到 PATH 中
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile

# 验证安装
brew --version

# 通过 Homebrew 安装 Miniconda
brew install --cask miniconda
```

#### 1.1.3 初始化 Conda

```bash
# 初始化 conda 并重新加载 shell 配置
conda init zsh
source ~/.zshrc

# 接受许可条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n unitree-rl-mujoco python=3.8 -y
```

### 1.3 激活虚拟环境

```bash
conda activate unitree-rl-mujoco
```

---

## 2. 安装依赖项

### 2.1 安装 PyTorch（Apple Silicon 的 MPS 支持）

PyTorch 是用于模型训练和推理的神经网络计算框架。对于 Apple Silicon Mac，我们需要安装支持 MPS 的版本：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch -y
```

**验证 MPS 支持：**
```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'MPS 可用: {torch.backends.mps.is_available()}')
print(f'MPS 已构建: {torch.backends.mps.is_built()}')

# 测试 MPS 加速
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(3, 3).to(device)
print(f'设备: {device}')
print('MPS 加速正常工作！')
"
```

### 2.2 安装兼容的 NumPy

在 macOS ARM64 上，较旧的 NumPy 版本可能没有预构建的轮子。安装兼容版本：

```bash
pip install "numpy>=1.19.0,<1.25.0"
```

### 2.3 安装 unitree_rl_gym

解压 unitree_rl_gym.zip 并导航到目录：

```bash
unzip unitree_rl_gym.zip
cd unitree_rl_gym
pip install -e .
```

---

## 3. 运行 MuJoCo

### 3.1 重要：使用 mjpython

在 macOS 上，MuJoCo 图形程序**必须**使用 `mjpython` 而不是 `python`：

```bash
# 检查 mjpython 是否可用
which mjpython

# 运行 MuJoCo 部署脚本
mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

### 3.2 运行自己的模型

要运行您自己训练的模型：

1. **下载训练权重** 从训练权重保存地址（文件存储）
   - 示例路径：`unitree_rl_gym/logs/g1/`
   - 将训练权重下载到本地机器

2. **更新配置文件**
   - 修改 `deploy/deploy_real/configs/g1.yaml`
   - 将 `policy_path` 改为指向本地权重文件路径

3. **运行部署**
   ```bash
   mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
   ```

### 3.3 常见问题和解决方案

**错误：`RuntimeError: launch_passive requires that the Python script be run under mjpython on macOS`**

**解决方案：** 使用 `mjpython` 而不是 `python` 运行脚本。

**错误：NumPy 编译失败**

**解决方案：** 使用兼容的 NumPy 版本 (>=1.19.0,<1.25.0)。

**错误：PyTorch CUDA 相关错误**

**解决方案：** 按照步骤 2.1 安装支持 MPS 的 PyTorch 版本。

---

## 4. 故障排除

### 4.1 如果 mjpython 不可用

```bash
# 重新安装 mujoco
pip uninstall mujoco -y
pip install mujoco==3.2.3
```

### 4.2 如果找不到 conda 命令

```bash
# 手动添加 PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# 或重新初始化
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

### 4.3 如果图形不显示

```bash
# 检查图形环境
echo $DISPLAY

# 在 macOS 上，通常不需要 DISPLAY
# 如果出现问题，尝试重启终端或系统
```

---

## 总结

按照本指南操作后，您应该拥有：
- 一个工作的 Python 3.8 Conda 环境
- 支持 Apple Silicon Mac MPS 的 PyTorch
- 兼容的 NumPy 版本
- 已安装的 unitree_rl_gym
- 使用 mjpython 运行的 MuJoCo

如需额外支持，请参考故障排除部分或查看官方文档。 