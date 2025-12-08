# Checkpoint保存逻辑修改说明

## 修改日期
2024-12-07

## 修改目的
1. 每个epoch训练完都进行一次测试
2. 计算评估指标mpjpe和pa_mpjpe
3. 如果两者平均值达到最小值，则保存模型

## 修改内容

### 1. 导入Evaluator (第31行)

**代码**:
```python
from metrics.evaluator import Evaluator
from typing import Optional
```

**说明**: 导入评估器类用于计算mpjpe和pa_mpjpe指标

### 2. 修改test函数 (第377-500行)

**主要变化**:
- 添加`evaluator`参数，支持传入Evaluator实例
- 在测试过程中计算mpjpe和pa_mpjpe指标
- 返回包含所有指标的字典，而不是只返回loss

**返回格式**:
```python
{
    'test_loss': float,      # 测试loss
    'mpjpe': float,          # MPJPE指标 (mm)
    'pa_mpjpe': float,       # PA-MPJPE指标 (mm)
    'avg_metric': float      # (mpjpe + pa_mpjpe) / 2.0 (mm)
}
```

**关键代码**:
```python
# Compute evaluation metrics
pred_keypoints_3d = predictions['keypoints_3d']  # (B, J, 3) in mm
gt_keypoints_3d = targets['keypoints_3d']  # (B, J, 3) in mm

# Evaluator expects tensors
evaluator(pred_keypoints_3d, gt_keypoints_3d)

# Get metrics
metrics_dict = evaluator.get_metrics_dict()
mpjpe = metrics_dict.get('mpjpe', float('inf'))
pa_mpjpe = metrics_dict.get('pa_mpjpe', float('inf'))
avg_metric = (mpjpe + pa_mpjpe) / 2.0
```

### 3. 修改训练循环 (第640-720行)

**主要变化**:
1. **每个epoch都测试**: 移除了`if (epoch + 1) % test_freq == 0`的条件，每个epoch都进行测试
2. **初始化Evaluator**: 在训练循环开始前创建Evaluator实例
3. **跟踪最佳模型**: 使用`best_avg_metric`和`best_epoch`跟踪最佳性能
4. **基于指标保存**: 当`avg_metric < best_avg_metric`时保存最佳模型

**关键代码**:
```python
# Initialize evaluator for test set (only on rank 0)
if rank == 0:
    test_dataset_length = len(test_loader.dataset)
    test_evaluator = Evaluator(
        dataset_length=test_dataset_length,
        keypoint_list=None,  # Use all keypoints
        root_joint_idx=0,  # Wrist joint
        metrics=['mpjpe', 'pa_mpjpe'],
        save_predictions=False
    )
else:
    test_evaluator = None

# Track best average metric (mpjpe + pa_mpjpe) / 2
best_avg_metric = float('inf')
best_epoch = -1

# In training loop:
# Test every epoch
test_metrics = test(model, test_loader, criterion, device, epoch, writer, 
                  rank, distributed, evaluator=test_evaluator)

if rank == 0:
    avg_metric = test_metrics.get('avg_metric', float('inf'))
    
    # Save best model based on average metric
    if avg_metric < best_avg_metric:
        best_avg_metric = avg_metric
        best_epoch = epoch
        # Save best checkpoint
        best_checkpoint_path = save_dir / 'best_model.pth'
        torch.save(best_checkpoint, best_checkpoint_path)
```

### 4. 保存的Checkpoint内容

**最佳模型checkpoint** (`best_model.pth`):
```python
{
    'epoch': int,                    # Epoch number
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # Scheduler state (if exists)
    'train_loss': float,             # Training loss
    'test_loss': float,              # Test loss
    'mpjpe': float,                  # MPJPE metric (mm)
    'pa_mpjpe': float,               # PA-MPJPE metric (mm)
    'avg_metric': float             # Average metric (mm)
}
```

**定期checkpoint** (`checkpoint_epoch_{epoch+1}.pth`):
- 仍然按照`save_freq`保存
- 包含基本的训练状态信息

## 预期效果

### 1. 每个Epoch都测试
- **之前**: 每`test_freq`个epoch测试一次
- **现在**: 每个epoch都测试，可以更及时地监控模型性能

### 2. 基于指标保存最佳模型
- **之前**: 只保存定期checkpoint，不跟踪最佳模型
- **现在**: 自动保存`avg_metric = (mpjpe + pa_mpjpe) / 2.0`最小的模型

### 3. 详细的评估信息
- 每个epoch打印:
  ```
  Epoch 0: Train Loss = 0.6958, Test Loss = 0.6234, LR = 1.00e-05
    MPJPE: 12.345 mm, PA-MPJPE: 8.901 mm, Avg Metric: 10.623 mm
    ✓ Saved best model (avg_metric=10.623 mm) to checkpoints/best_model.pth
  ```

## 使用说明

### 加载最佳模型
```python
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best model: Epoch {checkpoint['epoch']}, "
      f"MPJPE={checkpoint['mpjpe']:.3f} mm, "
      f"PA-MPJPE={checkpoint['pa_mpjpe']:.3f} mm, "
      f"Avg={checkpoint['avg_metric']:.3f} mm")
```

### 查看训练过程中的最佳模型
训练结束后会打印:
```
Best model: Epoch 15, Avg Metric = 10.623 mm
```

## 注意事项

1. **Evaluator重置**: 每个epoch开始测试前，需要重置evaluator的counter:
   ```python
   test_evaluator.counter = 0
   ```

2. **分布式训练**: 只有rank 0进程计算和保存指标，其他进程返回dummy值

3. **内存使用**: 每个epoch都测试会增加训练时间，但可以更及时地发现最佳模型

4. **指标单位**: mpjpe和pa_mpjpe的单位是毫米(mm)

## 相关文件

- 修改文件: `train.py`
- 评估器: `metrics/evaluator.py`
- 指标计算: `metrics/pose_metrics.py`
- 配置文件: `config/config.yaml` (test_freq参数现在主要用于定期checkpoint，不影响测试频率)






