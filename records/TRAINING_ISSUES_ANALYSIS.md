# UTNet训练问题分析与解决方案

## 问题1: SignalException (SIGHUP)

### 错误信息
```
torch.distributed.elastic.multiprocessing.api.SignalException: Process 2095685 got signal: 1
```

### 原因
- **SIGHUP (signal 1)** 是进程被外部信号终止的信号
- 通常发生在：
  - 终端关闭
  - nohup进程被中断
  - SSH连接断开
  - 系统资源限制

### 解决方案
这不是代码问题，而是运行环境问题。建议：
1. 使用 `screen` 或 `tmux` 来运行训练，而不是直接使用nohup
2. 确保SSH连接稳定
3. 检查系统资源（内存、磁盘空间）

### 推荐运行方式
```bash
# 使用screen
screen -S utnet_training
cd /data0/users/Robert/linweiquan/UTNet
torchrun --nproc_per_node=4 --master_port=29500 train.py --config config/config.yaml

# 或者使用tmux
tmux new -s utnet_training
cd /data0/users/Robert/linweiquan/UTNet
torchrun --nproc_per_node=4 --master_port=29500 train.py --config config/config.yaml
```

---

## 问题2: Loss不收敛 (Loss一直不下降)

### 现象
- Epoch 0: Train Loss = 3953.7583
- Epoch 1: Train Loss = 3953.7605
- Loss在3953左右波动，完全没有下降趋势

### 根本原因分析

#### 1. **Loss计算问题：使用sum()而不是mean()**
   - 当前代码：`loss = loss_weighted.sum()` (Keypoint2DLoss, Keypoint3DLoss)
   - 问题：loss值与batch_size成正比，导致loss值过大且不稳定
   - 应该：使用mean()或除以batch_size

#### 2. **Loss权重过小**
   - 当前配置：`w_2d=0.01, w_3d_joint=0.05`
   - 如果loss值本身很大（如3953），乘以0.01后只有39.5，可能不足以驱动模型学习

#### 3. **没有梯度裁剪**
   - 大loss可能导致梯度爆炸或消失
   - 需要添加梯度裁剪

#### 4. **学习率可能不合适**
   - 虽然WiLoR也用1e-5，但UTNet的loss scale不同，可能需要调整

### 解决方案

#### 方案1: 修复Loss计算（推荐）
将loss从sum()改为mean()，并调整权重：

```python
# 在 Keypoint2DLoss 和 Keypoint3DLoss 中
# 从 loss = loss_weighted.sum() 
# 改为 loss = loss_weighted.mean() * batch_size  # 保持与sum()相同的scale
# 或者直接 loss = loss_weighted.mean()
```

#### 方案2: 调整Loss权重
增加loss权重，使loss scale合理：
```yaml
loss:
  w_2d: 1.0      # 从0.01增加到1.0
  w_3d_joint: 1.0  # 从0.05增加到1.0
  w_3d_vert: 0.5
  w_prior: 0.01
  w_aux: 0.1
```

#### 方案3: 添加梯度裁剪
在训练循环中添加梯度裁剪：
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

#### 方案4: 检查数据预处理
确保数据预处理正确，特别是：
- 2D keypoint归一化到[-0.5, 0.5]
- 3D joint单位（毫米）
- 图像归一化（ImageNet mean/std）

### 推荐修复步骤

1. **立即修复**：修改loss计算为mean()，并调整权重
2. **添加梯度裁剪**：防止梯度爆炸
3. **添加loss监控**：打印各个loss组件的值，便于调试
4. **检查数据**：验证数据加载和预处理是否正确

