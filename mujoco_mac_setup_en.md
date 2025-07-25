# MuJoCo Installation Guide for macOS

This guide provides step-by-step instructions for setting up MuJoCo with unitree_rl_gym on macOS systems.

---

## Table of Contents
1. [Creating a Virtual Environment](#1-creating-a-virtual-environment)
2. [Installing Dependencies](#2-installing-dependencies)
3. [Running MuJoCo](#3-running-mujoco)
4. [Troubleshooting](#4-troubleshooting)

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments.

#### 1.1.1 Check Your Mac Architecture

First, check if your Mac is Intel or Apple Silicon:

```bash
uname -m
```

- If it shows `x86_64`: Intel Mac
- If it shows `arm64`: Apple Silicon Mac

#### 1.1.2 Download Miniconda for macOS

**For Apple Silicon Mac (M1/M2/M3):**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

**For Intel Mac:**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

**Alternative: Install via Homebrew (Recommended)**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile

# Verify installation
brew --version

# Install Miniconda via Homebrew
brew install --cask miniconda
```

#### 1.1.3 Initialize Conda

```bash
# Initialize conda and reload shell configuration
conda init zsh
source ~/.zshrc

# Accept the license terms
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n unitree-rl-mujoco python=3.8 -y
```

### 1.3 Activate the Virtual Environment

```bash
conda activate unitree-rl-mujoco
```

---

## 2. Installing Dependencies

### 2.1 Install PyTorch (MPS Support for Apple Silicon)

PyTorch is a neural network computation framework used for model training and inference. For Apple Silicon Macs, we need to install the MPS-enabled version:

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch -y
```

**Verify MPS Support:**
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')

# Test MPS acceleration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(3, 3).to(device)
print(f'Device: {device}')
print('MPS acceleration working!')
"
```

### 2.2 Install Compatible NumPy

On macOS ARM64, older NumPy versions may not have pre-built wheels. Install a compatible version:

```bash
pip install "numpy>=1.19.0,<1.25.0"
```

### 2.3 Install unitree_rl_gym

Unzip unitree_rl_gym.zip and navigate to the directory:

```bash
unzip unitree_rl_gym.zip
cd unitree_rl_gym
pip install -e .
```

---

## 3. Running MuJoCo

### 3.1 Important: Use mjpython

On macOS, MuJoCo graphical programs **must** use `mjpython` instead of `python`:

```bash
# Check if mjpython is available
which mjpython

# Run MuJoCo deployment script
mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

### 3.2 Running Your Own Model

To run your own trained model:

1. **Download trained weights** from the training weight save location (file storage)
   - Example path: `unitree_rl_gym/logs/g1/`
   - Download the trained weights to your local machine

2. **Update configuration file**
   - Modify `deploy/deploy_real/configs/g1.yaml`
   - Change the `policy_path` to point to your local weight file path

3. **Run the deployment**
   ```bash
   mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
   ```

### 3.3 Common Issues and Solutions

**Error: `RuntimeError: launch_passive requires that the Python script be run under mjpython on macOS`**

**Solution:** Use `mjpython` instead of `python` to run the script.

**Error: NumPy compilation failed**

**Solution:** Use compatible NumPy version (>=1.19.0,<1.25.0).

**Error: PyTorch CUDA-related errors**

**Solution:** Install MPS-enabled PyTorch version as shown in step 2.1.

---

## 4. Troubleshooting

### 4.1 If mjpython is not available

```bash
# Reinstall mujoco
pip uninstall mujoco -y
pip install mujoco==3.2.3
```

### 4.2 If conda command is not found

```bash
# Manual PATH addition
export PATH="$HOME/miniconda3/bin:$PATH"

# Or reinitialize
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

### 4.3 If graphics don't display

```bash
# Check graphics environment
echo $DISPLAY

# On macOS, DISPLAY is usually not needed
# If problems occur, try restarting terminal or system
```

---

## Summary

After following this guide, you should have:
- A working Conda environment with Python 3.8
- PyTorch with MPS support for Apple Silicon Macs
- Compatible NumPy version
- unitree_rl_gym installed
- MuJoCo running with mjpython

For additional support, refer to the troubleshooting section or check the official documentation. 