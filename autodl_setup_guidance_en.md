# AutoDL IsaacGym Training Setup Guide

## Overview
This guide provides step-by-step instructions for setting up IsaacGym training environment on AutoDL platform.

## Platform Access
- Website: https://www.autodl.com/console/homepage/personal
- Navigate to Container Instances → Rent New Instance

## Instance Configuration
1. **Region Selection**: Choose Inner Mongolia Zone B (other zones are also acceptable)
2. **GPU Selection**: Rent RTX 3090/4090
3. **Image Selection**: PyTorch / 2.3.0 / 3.12 (Ubuntu 22.04) / 12.1
4. **Action**: Create and power on the instance

## Instance Management
- **Start Training**: Click JupyterLab on the created instance, then click Terminal in the page
- **Shutdown**: Remember to shut down when not in use to avoid continuous billing

## File Storage Configuration (Network Shared Storage)
**Purpose**: Mount to different instances in the same region to avoid data loss when instances are occupied and cannot be powered on.

1. Click "File Storage" on the main interface
2. Select Inner Mongolia Zone B (same zone as your rented instance)
3. Initialize file storage
4. When creating instances in this region, file storage will be automatically mounted to `/root/autodl-fs` directory
5. Upload `unitree-rl-gym.zip` to file storage

## Environment Setup

### Step 1: Access Terminal
Click JupyterLab on your created instance, then click Terminal in the page.

### Step 2: Create and Activate Virtual Environment
```bash
conda create -n unitree-rl python=3.8 -y
conda init bash && source /root/.bashrc
conda activate unitree-rl
```

### Step 3: Install PyTorch
**Note**: This step may take some time
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 4: Extract and Navigate to Project
```bash
cd /autodl-fs/data
unzip unitree-rl-gym.zip
cd unitree_rl_gym
```

### Step 5: Install IsaacGym
```bash
cd isaacgym/python
pip install -e .
cd ../../
```

### Step 6: Install RSL_RL
```bash
cd rsl_rl
pip install -e .
cd ..
```

### Step 7: Install Unitree RL Gym
```bash
pip install -e .
```

### Step 8: Start Training
```bash
python legged_gym/scripts/train.py --task=g1 --headless
```

## Troubleshooting

### Python Library Issue
If you encounter Python library issues, run:
```bash
find / -name libpython3.8.so.1.0
```

Expected output:
```
/root/miniconda3/envs/unitree-rl/lib/libpython3.8.so.1.0
/root/miniconda3/pkgs/python-3.8.20-he870216_0/lib/libpython3.8.so.1.0
```

Then copy the library:
```bash
sudo cp /root/miniconda3/envs/unitree-rl/lib/libpython3.8.so.1.0 /usr/lib/
```

### NumPy Version Issue
If you encounter NumPy-related issues, downgrade to version 1.22:
```bash
pip install numpy==1.22
```

## Screen Session Management

### Create a Named Session
```bash
screen -S session_name
conda activate unitree-rl
python legged_gym/scripts/train.py --task=g1 --headless
```

### Detach from Session
Press `Ctrl + A`, then press `D`

### Connect to Specific Session
```bash
screen -r session_name
```

### Force Reconnect to Session (even if connected by another user)
```bash
screen -r -d
```

### Terminate Session
```bash
screen -X -S session_name quit
```

### List All Sessions
```bash
screen -ls
```