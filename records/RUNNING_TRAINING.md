# UTNet训练运行指南

## 问题：SIGHUP信号错误

如果在使用`nohup`运行训练时遇到以下错误：
```
SignalException: Process got signal: 1 (SIGHUP)
```

这通常是因为：
- 终端关闭导致进程被终止
- SSH连接断开
- 系统资源限制

## 推荐解决方案

### 方案1：使用screen（推荐）

`screen`是一个终端复用器，可以让训练在后台持续运行，即使SSH断开也不会中断。

#### 安装screen（如果未安装）
```bash
# Ubuntu/Debian
sudo apt-get install screen

# CentOS/RHEL
sudo yum install screen
```

#### 使用screen运行训练
```bash
# 1. 创建新的screen会话
screen -S utnet_training

# 2. 进入UTNet目录
cd /data0/users/Robert/linweiquan/UTNet

# 3. 运行训练
torchrun --nproc_per_node=4 --master_port=29500 train.py --config config/config.yaml

# 4. 分离screen会话（训练继续运行）
# 按 Ctrl+A，然后按 D

# 5. 重新连接到screen会话
screen -r utnet_training

# 6. 查看所有screen会话
screen -ls

# 7. 终止screen会话
# 在screen会话内按 Ctrl+D，或者
screen -X -S utnet_training quit
```

### 方案2：使用tmux

`tmux`是另一个终端复用器，功能类似screen。

#### 安装tmux（如果未安装）
```bash
# Ubuntu/Debian
sudo apt-get install tmux

# CentOS/RHEL
sudo yum install tmux
```

#### 使用tmux运行训练
```bash
# 1. 创建新的tmux会话
tmux new -s utnet_training

# 2. 进入UTNet目录
cd /data0/users/Robert/linweiquan/UTNet

# 3. 运行训练
torchrun --nproc_per_node=4 --master_port=29500 train.py --config config/config.yaml

# 4. 分离tmux会话（训练继续运行）
# 按 Ctrl+B，然后按 D

# 5. 重新连接到tmux会话
tmux attach -t utnet_training

# 6. 查看所有tmux会话
tmux ls

# 7. 终止tmux会话
# 在tmux会话内输入 exit，或者
tmux kill-session -t utnet_training
```

### 方案3：使用nohup（不推荐，但可用）

如果必须使用nohup，确保：
1. 使用`nohup`命令并重定向输出
2. 在后台运行（使用`&`）
3. 使用`disown`命令让进程脱离终端

```bash
cd /data0/users/Robert/linweiquan/UTNet
nohup torchrun --nproc_per_node=4 --master_port=29500 train.py --config config/config.yaml > training.log 2>&1 &
disown
```

然后可以安全地关闭终端。查看日志：
```bash
tail -f training.log
```

## 推荐做法

**强烈推荐使用screen或tmux**，因为：
1. 可以随时重新连接查看训练进度
2. 可以交互式地查看输出
3. 更稳定，不容易被信号中断
4. 可以方便地管理多个训练任务

## 监控训练

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志（如果使用screen/tmux）
```bash
# screen
screen -r utnet_training

# tmux
tmux attach -t utnet_training
```

### 查看TensorBoard日志
```bash
# 在另一个终端
cd /data0/users/Robert/linweiquan/UTNet
tensorboard --logdir=./logs --port=6006
```

然后在浏览器中打开 `http://your-server-ip:6006`

## 停止训练

### 如果使用screen/tmux
1. 连接到会话
2. 按 `Ctrl+C` 停止训练
3. 输入 `exit` 退出会话

### 如果使用nohup
```bash
# 查找进程
ps aux | grep train.py

# 终止进程
kill <PID>

# 或者强制终止
kill -9 <PID>
```

