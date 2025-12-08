# train_epoch函数修改说明

## 修改日期
2024-12-07

## 修改目的
1. 添加梯度裁剪，防止梯度爆炸
2. 添加详细的loss组件打印，便于诊断训练问题

## 修改内容

### 1. 梯度裁剪 (第302-308行)

**位置**: `train.py` 的 `train_epoch` 函数中，`loss.backward()` 之后

**代码**:
```python
# Gradient clipping (prevent gradient explosion)
# Get model parameters (handle DDP)
if isinstance(model, DDP):
    model_params = model.module.parameters()
else:
    model_params = model.parameters()
torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
```

**说明**:
- 在反向传播后、优化器更新前添加梯度裁剪
- 支持DDP和单GPU模式
- 最大梯度范数设置为1.0（常用值）

### 2. 详细的Loss组件打印 (第317-355行)

**位置**: `train.py` 的 `train_epoch` 函数中，每 `log_freq` 次迭代

**代码**:
```python
# Print detailed loss breakdown for diagnosis
loss_2d = losses.get('loss_2d', torch.tensor(0.0))
loss_3d_joint = losses.get('loss_3d_joint', torch.tensor(0.0))
loss_3d_vert = losses.get('loss_3d_vert', torch.tensor(0.0))
loss_aux = losses.get('loss_aux', torch.tensor(0.0))
# Prior loss may have sub-components (shape_prior, pose_prior)
loss_prior_shape = losses.get('shape_prior', torch.tensor(0.0))
loss_prior_pose = losses.get('pose_prior', torch.tensor(0.0))
total_loss_val = loss.item()

# Convert to float if tensor
if isinstance(loss_2d, torch.Tensor):
    loss_2d = loss_2d.item()
# ... (其他loss组件类似处理)

print(f'\n[Epoch {epoch}, Iter {batch_idx}] Loss breakdown:')
print(f'  2D keypoint loss: {loss_2d:.4f}')
print(f'  3D joint loss: {loss_3d_joint:.4f}')
if loss_3d_vert > 0:
    print(f'  3D vertex loss: {loss_3d_vert:.4f}')
print(f'  Prior loss (shape): {loss_prior_shape:.4f}')
if loss_prior_pose > 0:
    print(f'  Prior loss (pose): {loss_prior_pose:.4f}')
print(f'  Aux loss: {loss_aux:.4f}')
print(f'  Total loss: {total_loss_val:.4f}')
```

**说明**:
- 打印所有loss组件的值，便于诊断哪个loss占主导
- 处理tensor和float类型的loss值
- 条件打印vertex loss和pose prior（如果为0则不打印）

### 3. 学习率监控 (第605-608行, 613-614行)

**位置**: `train.py` 的训练循环中，每个epoch结束后

**代码**:
```python
# Print epoch summary with learning rate (only on rank 0)
if rank == 0:
    current_lr = optimizer.param_groups[0]['lr']
    print(f'\nEpoch {epoch}: Train Loss = {train_loss:.4f}, LR = {current_lr:.2e}')
```

**说明**:
- 在每个epoch结束后打印当前学习率
- 格式为科学计数法（如 `1.00e-05`）
- 只在rank 0进程打印（避免多GPU重复打印）

## 预期效果

### 梯度裁剪
- **防止梯度爆炸**: 当梯度范数超过1.0时，会被裁剪到1.0
- **稳定训练**: 避免因梯度过大导致的训练不稳定
- **性能影响**: 几乎无性能开销

### Loss组件打印
- **诊断问题**: 可以快速看出哪个loss组件占主导
- **监控训练**: 观察各个loss组件的变化趋势
- **调试方便**: 如果某个loss异常大，可以快速定位

### 学习率监控
- **跟踪学习率**: 观察学习率调度器的效果
- **调试学习率**: 如果loss不下降，可以检查学习率是否合适

## 使用示例

训练时会看到类似输出：

```
[Epoch 0, Iter 0] Loss breakdown:
  2D keypoint loss: 0.1234
  3D joint loss: 0.5678
  Prior loss (shape): 0.0012
  Aux loss: 0.0034
  Total loss: 0.6958

Epoch 0: Train Loss = 0.6958, LR = 1.00e-05
```

## 相关文件

- 修改文件: `train.py`
- 相关配置: `config/config.yaml` (log_freq参数控制打印频率)






