# 修复 UTNet MPJPE: inf mm 指标计算错误

## 修改日期
2025-12-07

## 问题描述

训练 UTNet 时，验证阶段的指标显示：
```
MPJPE: inf mm, PA-MPJPE: inf mm, Avg Metric: inf mm
```

## 问题根源

### 1. HO3D_v3 数据集结构
- **Train split**: 83325 个样本，包含完整的 21 个关节点 GT 标注
- **Evaluation split**: 仅包含 root joint（单个3D点），没有完整的 21 个关节点 GT

### 2. UTNet 原实现的问题

在 `UTNet/dataloader/ho3d_dataset.py:228`：
```python
# For evaluation split:
hand_joints_3d = np.tile(root_joint_3d.reshape(1, 3), (21, 1))  # 所有关节点=root joint
```

在 `UTNet/dataloader/ho3d_dataset.py:804`：
```python
gt3Dcrop = np.zeros_like(joint_xyz)  # 全零！
```

**后果链**：
1. 所有 21 个关节点位置相同（都是 root joint）→ 方差为 0
2. Evaluator 在 `metrics/evaluator.py:129` 过滤：`valid_mask = (gt_var > 1e-8)` → 所有样本被过滤
3. Counter 保持为 0 → `get_metrics_dict()` 返回空字典 → `train.py` 中 `metrics_dict.get('mpjpe', float('inf'))` 返回 inf

### 3. WiLoR 的正确做法

查看 WiLoR 配置文件 `pretrained_models/dataset_config.yaml:31-34`：
- 只有 `HO3D-TRAIN`（83325 个样本），没有 `HO3D-VAL` 或 `HO3D-EVAL`
- WiLoR 使用 train split 进行训练，并从 `FREIHAND-TRAIN` 进行验证
- **不使用 evaluation split 进行评估**（因为没有完整 GT）

## 解决方案

### 修改策略

对齐 WiLoR 的做法：将 HO3D 的 train split 划分为 train (90%) 和 val (10%)，不使用 evaluation split 进行训练过程中的验证。

### 修改文件列表

1. **`UTNet/train.py`**
   - 修改 `create_dataloader` 函数，添加 train/val 划分逻辑
   - 将所有 `test_loader` 改为 `val_loader`
   - 将所有 `test_evaluator` 改为 `val_evaluator`
   - 将所有 `test_metrics` 改为 `val_metrics`
   - 更新打印信息中的 "Test Loss" 为 "Val Loss"

2. **`UTNet/config/config.yaml`**
   - 添加 `val_split_ratio: 0.1` 参数

## 详细修改

### 1. 修改 `create_dataloader` 函数

**位置**: `UTNet/train.py:45-117`

**修改内容**:
- 添加 `val_split_ratio` 参数说明
- 对于 HO3D 数据集，当 split 为 'train' 或 'val' 时，都从 train split 加载数据
- 使用固定随机种子 (42) 对数据进行划分：
  - train: 前 90% (约 75,000 样本)
  - val: 后 10% (约 8,300 样本)
- 使用 `torch.utils.data.Subset` 实现子集划分
- 添加样本数量打印信息

**关键代码**:
```python
# For HO3D, we split train into train/val since evaluation has no full GT
if dataset_name == 'ho3d' and split in ['train', 'val']:
    actual_split = 'train'  # Always load from train split
else:
    actual_split = split

# ... create dataset ...

# Split train/val for HO3D
if dataset_name == 'ho3d' and split in ['train', 'val']:
    total_size = len(dataset)
    indices = list(range(total_size))
    
    # Use fixed random seed for reproducibility
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(total_size * (1 - val_split_ratio))
    
    if split == 'train':
        indices = indices[:split_idx]
    else:  # split == 'val'
        indices = indices[split_idx:]
    
    from torch.utils.data import Subset
    dataset = Subset(dataset, indices)
```

### 2. 修改主训练循环

**位置**: `UTNet/train.py:641-784`

**修改内容**:
- 第 643 行: `test_loader` → `val_loader`
- 第 710 行: `test_dataset_length` → `val_dataset_length`
- 第 711 行: `test_evaluator` → `val_evaluator`
- 第 719 行: `test_evaluator` → `val_evaluator`
- 第 743 行: `test_evaluator.counter` → `val_evaluator.counter`
- 第 745 行: `test_metrics = test(model, test_loader, ...)` → `val_metrics = test(model, val_loader, ...)`
- 第 750 行: `test_loss` → `val_loss`, `test_metrics` → `val_metrics`
- 第 755 行: 打印信息 "Test Loss" → "Val Loss"
- 第 774 行: checkpoint 中的 `'test_loss'` → `'val_loss'`

### 3. 添加配置参数

**位置**: `UTNet/config/config.yaml:55-67`

**修改内容**:
```yaml
training:
  # ... existing params ...
  val_split_ratio: 0.1  # 10% of train split for validation (only for HO3D)
```

## 验证

修改后，训练日志应该显示：

```
HO3D train split: 74993 samples (90.0% of train split)
HO3D val split: 8332 samples (10.0% of train split)

Epoch 0: Train Loss = xxxxx, Val Loss = xxxxx, LR = 1.00e-05
  MPJPE: xx.xxx mm, PA-MPJPE: xx.xxx mm, Avg Metric: xx.xxx mm
```

指标不再显示 `inf`，而是正常的数值。

## 注意事项

1. **Evaluation split 的用途**：
   - 保留用于最终推理和提交到 HO3D 官方评估服务器
   - 不能用于训练过程中的指标计算（因为没有完整 GT）

2. **数据泄漏防护**：
   - 使用固定随机种子 (42) 确保每次运行时 train/val 划分一致
   - 避免验证集数据污染训练集

3. **DexYCB 数据集**：
   - 不受此修改影响
   - 继续使用原有的 train/test split（test split 有完整 GT）

4. **可复现性**：
   - 固定随机种子保证了 train/val 划分的可复现性
   - 所有实验使用相同的数据划分

## 对齐 WiLoR

这次修改使 UTNet 的数据使用方式与 WiLoR 保持一致：
- ✅ 只使用 train split 进行训练和验证
- ✅ 不使用 evaluation split 计算训练指标
- ✅ 使用固定比例划分 train/val
- ✅ 保持数据划分的可复现性

## 预期效果

1. **指标正常显示**：MPJPE 和 PA-MPJPE 显示合理的毫米值，不再是 inf
2. **训练监控**：可以通过验证集指标监控模型训练进度
3. **最佳模型保存**：根据验证集的 avg_metric 保存最佳模型
4. **与 WiLoR 对齐**：数据使用策略与 SOTA 模型保持一致






