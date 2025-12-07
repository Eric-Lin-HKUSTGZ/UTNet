# UTNet损失函数与WiLoR对齐修改记录

## 修改日期
2025-12-07

## 修改目标
将 UTNet 中的 `Keypoint2DLoss` 和 `Keypoint3DLoss` 完全对齐到 WiLoR 的实现，确保损失计算方式完全一致。

## 主要差异分析

### 1. Keypoint2DLoss 差异

**WiLoR 实现特点：**
- 接受参数格式：`pred_keypoints_2d` (B, S, N, 2)，`gt_keypoints_2d` (B, S, N, 3)
- 从 GT 最后一维提取 confidence：`conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()`
- 损失计算：`(conf * self.loss_fn(pred, gt[:, :, :-1])).sum(dim=(1,2))`
- 返回：`loss.sum()`

**UTNet 原实现问题：**
- 虽然计算逻辑类似，但支持可选的 `mask` 参数和多种输入格式
- 代码风格不一致

**修改内容：**
- 完全采用 WiLoR 的实现代码
- 移除可选的 `mask` 参数
- 保持完全一致的代码结构和注释

### 2. Keypoint3DLoss 差异

**WiLoR 实现特点：**
- 接受参数格式：`pred_keypoints_3d` (B, S, N, 3)，`gt_keypoints_3d` (B, S, N, 4)
- `pelvis_id` 参数在 `forward` 方法中接收（默认 0）
- 使用 `gt_keypoints_3d.clone()` 避免修改原始数据
- 先对预测和 GT 分别减去 root joint，再提取 confidence
- 损失计算：`(conf * self.loss_fn(pred_relative, gt_relative)).sum(dim=(1,2))`

**UTNet 原实现问题：**
- `root_id` 参数在 `__init__` 中接收，不如 WiLoR 灵活
- 代码风格和细节不完全一致

**修改内容：**
- 完全采用 WiLoR 的实现代码
- 将 `root_id` 参数改为 `pelvis_id`，并移到 `forward` 方法
- 保持完全一致的代码结构和注释

## 数据格式调整

### 问题
UTNet 数据集返回的格式与 WiLoR 损失函数期望的格式不匹配：
- **2D keypoints**: 数据集返回 (B, J, 3)（第3维是深度），但只有 x, y 被使用；WiLoR 损失期望 (B, J, 3)（第3维是 confidence）
- **3D keypoints**: 数据集返回 (B, J, 3)；WiLoR 损失期望 (B, J, 4)（第4维是 confidence）

### 解决方案
在 `train.py` 的 `train_epoch` 和 `test` 函数中，添加 confidence 维度：

**修改前：**
```python
keypoints_2d = joint_img[:, :, :2].to(device)  # (B, J, 2)
targets = {
    'keypoints_2d': keypoints_2d,  # (B, J, 2)
    'keypoints_3d': batch['joints_3d_gt'].to(device),  # (B, J, 3)
    ...
}
```

**修改后：**
```python
# Prepare 2D keypoints with confidence (WiLoR format)
keypoints_2d_xy = joint_img[:, :, :2].to(device)  # (B, J, 2)
confidence_2d = torch.ones(batch_size, num_joints, 1, device=device, dtype=keypoints_2d_xy.dtype)
keypoints_2d_with_conf = torch.cat([keypoints_2d_xy, confidence_2d], dim=-1)  # (B, J, 3)

# Prepare 3D keypoints with confidence (WiLoR format)
keypoints_3d = batch['joints_3d_gt'].to(device)  # (B, J, 3)
confidence_3d = torch.ones(batch_size, num_joints, 1, device=device, dtype=keypoints_3d.dtype)
keypoints_3d_with_conf = torch.cat([keypoints_3d, confidence_3d], dim=-1)  # (B, J, 4)

targets = {
    'keypoints_2d': keypoints_2d_with_conf,  # (B, J, 3) with confidence
    'keypoints_3d': keypoints_3d_with_conf,  # (B, J, 4) with confidence
    ...
}
```

**Evaluator 调用：**
```python
# Evaluator expects tensors without confidence
gt_keypoints_3d = targets['keypoints_3d'][:, :, :3]  # Remove confidence
evaluator(pred_keypoints_3d, gt_keypoints_3d)
```

## 修改文件列表

1. **UTNet/src/losses/loss.py**
   - 修改 `Keypoint2DLoss` 类，完全对齐 WiLoR
   - 修改 `Keypoint3DLoss` 类，完全对齐 WiLoR
   - 更新 `UTNetLoss` 中的损失函数初始化

2. **UTNet/train.py**
   - 在 `train_epoch` 函数中添加 confidence 维度处理
   - 在 `test` 函数中添加 confidence 维度处理
   - 修改 evaluator 调用，去除 confidence 维度

## 对齐效果

### Keypoint2DLoss
✅ 完全一致的参数格式：`pred` (B, N, 2), `gt` (B, N, 3)  
✅ 完全一致的 confidence 提取方式  
✅ 完全一致的损失计算和聚合方式  
✅ 完全一致的代码风格

### Keypoint3DLoss
✅ 完全一致的参数格式：`pred` (B, N, 3), `gt` (B, N, 4)  
✅ 完全一致的 root joint 归一化方式  
✅ 完全一致的 confidence 提取方式  
✅ 完全一致的损失计算和聚合方式  
✅ 完全一致的代码风格

## 预期效果

1. **损失计算一致性**：确保 UTNet 的损失计算与 WiLoR 完全相同
2. **训练稳定性**：统一的损失函数有助于训练更加稳定
3. **可复现性**：与 WiLoR 对齐后，更容易复现 WiLoR 的训练效果
4. **代码可维护性**：统一的代码风格使得后续维护更加容易

## 注意事项

1. 所有关节点的 confidence 都设置为 1.0，因为 HO3D 和 DexYCB 数据集的所有标注关节点都是可见的
2. 如果未来使用其他数据集（可能有遮挡或不可见关节点），需要在数据集加载器中提供真实的 confidence 值
3. Evaluator 期望输入不带 confidence 的 3D keypoints，因此在调用时需要切片去除最后一维

## 验证建议

运行一个 epoch 的训练，检查：
- ✅ 损失值是否在合理范围内（不应该出现异常大或 NaN）
- ✅ 各个损失组件（2D、3D joint、prior、aux）的值是否符合预期
- ✅ MPJPE 和 PA-MPJPE 指标是否正常计算（不应该为 0 或 NaN）
- ✅ 损失是否开始下降



