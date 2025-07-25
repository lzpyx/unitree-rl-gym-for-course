# AutoDL IsaacGym 训练环境配置指南

## 概述
本指南提供在AutoDL平台上搭建IsaacGym训练环境的详细步骤说明。

## 平台访问
- 网站：https://www.autodl.com/console/homepage/personal
- 导航：容器实例 → 租用新实例

## 实例配置
1. **区域选择**：选择内蒙B区（其他区域也可以）
2. **GPU选择**：租用RTX 3090/4090
3. **镜像选择**：PyTorch / 2.3.0 / 3.12 (Ubuntu 22.04) / 12.1
4. **操作**：创建并开机

## 实例管理
- **开始训练**：点击刚创建的实例的JupyterLab，在页面中点击终端(Terminal)
- **关机**：不使用时记得关机，避免持续运行产生费用

## 文件存储配置（网络共享存储）
**目的**：可挂载至同一地区的不同实例中，避免关机后实例被占用无法开机时，另一个实例无法访问已配置的代码和数据。

1. 点击主界面的"文件存储"
2. 选择内蒙B区（与租用的实例所在分区相同）
3. 初始化文件存储
4. 在该地区创建实例开机后，文件存储将自动挂载至实例的`/root/autodl-fs`目录
5. 将`unitree-rl-gym.zip`上传到文件存储中

## 环境配置

### 步骤1：访问终端
点击刚创建的实例的JupyterLab，在页面中点击终端(Terminal)。

### 步骤2：创建并激活虚拟环境
```bash
conda create -n unitree-rl python=3.8 -y
conda init bash && source /root/.bashrc
conda activate unitree-rl
```

### 步骤3：安装PyTorch
**注意**：这一步需要等待一段时间
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 步骤4：解压并导航到项目目录
```bash
cd /autodl-fs/data
unzip unitree-rl-gym.zip
cd unitree_rl_gym
```

### 步骤5：安装IsaacGym
```bash
cd isaacgym/python
pip install -e .
cd ../../
```

### 步骤6：安装RSL_RL
```bash
cd rsl_rl
pip install -e .
cd ..
```

### 步骤7：安装Unitree RL Gym
```bash
pip install -e .
```

### 步骤8：开始训练
```bash
python legged_gym/scripts/train.py --task=g1 --headless
```

## 故障排除

### Python库问题
如果遇到Python库问题，运行：
```bash
find / -name libpython3.8.so.1.0
```

预期输出：
```
/root/miniconda3/envs/unitree-rl/lib/libpython3.8.so.1.0
/root/miniconda3/pkgs/python-3.8.20-he870216_0/lib/libpython3.8.so.1.0
```

然后复制库文件：
```bash
sudo cp /root/miniconda3/envs/unitree-rl/lib/libpython3.8.so.1.0 /usr/lib/
```

### NumPy版本问题
如果遇到NumPy相关问题，将版本降至1.22：
```bash
pip install numpy==1.22
```

## Screen会话管理

### 创建指定名称的会话
```bash
screen -S session_name
conda activate unitree-rl
python legged_gym/scripts/train.py --task=g1 --headless
```

### 从会话中分离
按`Ctrl + A`，然后按`D`

### 连接到指定会话
```bash
screen -r session_name
```

### 强制重新连接到会话（即使被其他用户连接）
```bash
screen -r -d
```

### 终止会话
```bash
screen -X -S session_name quit
```

### 列出所有会话
```bash
screen -ls
```