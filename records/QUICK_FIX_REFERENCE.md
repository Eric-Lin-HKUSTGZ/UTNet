# UTNet 训练问题快速修复参考

## 🚨 MANO 加载失败

### 症状
- 3D joint loss 异常高（70000+）且不下降
- 模型预测全是 0
- 训练完全不收敛

### 快速修复
在 `train.py` 开头添加兼容性补丁：

```python
import numpy as np
import inspect

# NumPy 兼容性
if not hasattr(np, 'bool'): np.bool = np.bool_
if not hasattr(np, 'int'): np.int = np.int_
if not hasattr(np, 'float'): np.float = np.float64
if not hasattr(np, 'complex'): np.complex = np.complex128
if not hasattr(np, 'object'): np.object = np.object_
if not hasattr(np, 'unicode'): np.unicode = np.str_
if not hasattr(np, 'str'): np.str = np.str_

# inspect 兼容性
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
```

确保 `utnet.py` 中 `use_pca=False`：

```python
mano_model = smplx.create(
    model_path=model_path,
    model_type='mano',
    gender='neutral',
    num_hand_joints=num_hand_joints,
    use_pca=False,  # ✅ 必须是 False
    flat_hand_mean=True
)
```

---

## 🚨 分布式训练验证指标 `inf`

### 症状
- `MPJPE: inf mm, PA-MPJPE: inf mm`
- 但 `Avg Metric` 显示正常值

### 快速修复
确保 `test()` 函数中 `metrics_dict` 包含所有指标：

```python
metrics_dict = {
    'test_loss': avg_loss,
    'mpjpe': mpjpe,           # ✅ 必须包含
    'pa_mpjpe': pa_mpjpe,     # ✅ 必须包含
    'avg_metric': avg_metric
}
```

---

## 🚨 3D Joint Loss 计算错误

### 症状
- Loss 值异常大
- 训练不稳定

### 快速修复
确保使用相对关节位置（相对于 root joint）：

```python
# ✅ 正确
pred_rel = pred_keypoints_3d - pred_keypoints_3d[:, [root_id], :]
gt_rel = gt_keypoints_3d - gt_keypoints_3d[:, [root_id], :]
loss = criterion(pred_rel, gt_rel)

# ❌ 错误
loss = criterion(pred_keypoints_3d, gt_keypoints_3d)
```

---

## 🚨 数据预处理不一致

### 症状
- 2D keypoint loss 不下降
- 预测结果偏差很大

### 快速修复
确保图像归一化使用 ImageNet 均值和标准差：

```python
# ✅ 正确
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
imgRGB = (imgRGB - mean) / std

# 2D keypoints 归一化到 [-0.5, 0.5]
keypoints_norm = keypoints / img_size - 0.5

# ❌ 错误
imgRGB = imgRGB / 255.0  # 直接除以 255
keypoints_norm = keypoints / (img_size / 2) - 1  # 归一化到 [-1, 1]
```

---

## 🚨 DDP 未使用参数错误

### 症状
```
RuntimeError: Expected to have finished reduction in the prior iteration 
before starting a new one.
```

### 快速修复
```python
model = DDP(
    model,
    device_ids=[device.index],
    output_device=device.index,
    find_unused_parameters=True  # ✅ 设置为 True
)
```

---

## 🚨 验证集没有完整 GT

### 症状
- 使用 HO3D evaluation split 时 `MPJPE: inf mm`
- 所有样本被过滤掉

### 快速修复
从 train split 划分验证集：

```python
# ✅ 正确：从 train split 划分
if dataset_name == 'ho3d' and split in ['train', 'val']:
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    if split == 'train':
        dataset = Subset(dataset, indices[:train_size])
    else:  # val
        dataset = Subset(dataset, indices[train_size:])

# ❌ 错误：直接使用 evaluation split
dataset = HO3DDataset(split='evaluation')  # 没有完整 GT
```

---

## 📝 环境要求

### Python 环境
- Python 3.8+（推荐 3.8，避免 3.12+ 的兼容性问题）
- PyTorch 1.10+
- CUDA 11.3+

### 关键依赖
```bash
pip install smplx scipy chumpy
```

### MANO 数据
确保 `mano_data/` 目录结构正确：
```
mano_data/
├── mano/
│   └── MANO_RIGHT.pkl
└── MANO_RIGHT.pkl  # 用于 faces 加载
```

---

## 🧪 快速测试

### 测试 MANO 加载
```bash
cd /data0/users/Robert/linweiquan/UTNet
python -c "
import torch
from src.utils.mano_utils import MANOWrapper
import smplx

mano = smplx.create('mano_data', 'mano', 'neutral', num_hand_joints=15, use_pca=False, flat_hand_mean=True)
wrapper = MANOWrapper(mano)

# Test forward
global_orient = torch.eye(3).unsqueeze(0).unsqueeze(0)
hand_pose = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 15, 3, 3)
betas = torch.zeros(1, 10)

out = wrapper(global_orient, hand_pose, betas)
print(f'✅ MANO works! Vertices: {out[\"vertices\"].shape}')
"
```

### 测试训练（单 GPU）
```bash
python train.py --config config/config.yaml --gpu 0
```

### 测试训练（多 GPU）
```bash
torchrun --nproc_per_node=4 train.py --config config/config.yaml
```

---

## 📊 正常训练日志示例

```
[Epoch 0, Iter 0] Loss breakdown:
  2D keypoint loss: 334.8019       # ✅ 正常范围（300-400）
  3D joint loss: 73632.9062        # ✅ 会逐渐下降
  Prior loss: 70.4138              # ✅ 正常范围（50-100）
  Aux loss: 0.0000                 # ✅ 可以为 0
  Total loss: 3826.6260            # ✅ 会逐渐下降

Pred 3D joints mean/std/range: mean=41.34, std=24.57, range=[-89.23, 112.57]  # ✅ 非零

Epoch 0: Train Loss = 3826.18, Val Loss = 3668.85, LR = 1.00e-05
  MPJPE: 68.234 mm, PA-MPJPE: 67.988 mm, Avg Metric: 68.111 mm  # ✅ 正常变化
```

---

## 🔗 详细文档

- 完整修复报告：`records/MANO模型加载问题修复报告.md`
- 实现总结：`IMPLEMENTATION_SUMMARY.md`
- 训练调试指南：`TRAINING_DEBUG_GUIDE.md`

---

**最后更新**: 2025-12-07

