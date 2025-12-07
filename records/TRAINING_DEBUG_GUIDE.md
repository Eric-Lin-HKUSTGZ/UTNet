# UTNet 训练调试指南

## 最近修复（2025-12-07）

### 修复内容
1. **分布式验证评估 Bug**：修复了验证指标完全不变的问题
2. **添加调试代码**：用于诊断 3D joint loss 过高的原因

详细修改记录见：`records/修复分布式验证评估和添加调试.md`

## 训练方式

### 标准训练（启用验证指标修复）

```bash
cd /data0/users/Robert/linweiquan/UTNet
torchrun --nproc_per_node=4 train.py --config config/config.yaml
```

**预期行为**：
- ✅ 验证指标（MPJPE, PA-MPJPE）会随训练变化
- ✅ 训练日志会显示 "Evaluated on 8332 samples from all 4 GPUs"
- ✅ 2D loss 应该逐渐下降

### 调试模式 1：基础数据统计（第一个 epoch 自动启用）

第一个 epoch 的前 3 个 batch 会自动打印调试信息：

```bash
torchrun --nproc_per_node=4 train.py --config config/config.yaml 2>&1 | tee training_debug.log
```

**查看内容**：
```
[Debug Batch 0]
  RGB range: [-2.118, 2.640]
  Depth range: [0.000, 1.000]
  GT 3D joints mean: XXX.XXX, std: YYY.YYY
  Pred 3D joints mean: AAA.AAA, std: BBB.BBB
  GT root joint: tensor([...])
  Pred root joint: tensor([...])
  GT centered by root mean: 0.000, std: ZZZ.ZZZ
```

**诊断关键点**：
- ✅ GT 3D joints mean 应该在 0-500 mm 范围
- ✅ GT centered by root mean 应该**接近 0**（≤ 1e-5）
- ❌ 如果 GT centered mean 不为 0 → **GT 可能已经是相对坐标，loss 不应再次中心化**

### 调试模式 2：Loss 函数内部调试

当怀疑 loss 计算有问题时启用：

```bash
export UTNET_DEBUG_LOSS=1
torchrun --nproc_per_node=4 train.py --config config/config.yaml 2>&1 | tee training_debug_loss.log
```

**查看内容**：
```
[Debug 3D Loss]
  Before centering - Pred mean: XXX.XXX, std: YYY.YYY
  Before centering - GT mean: AAA.AAA, std: BBB.BBB
  After centering - Pred mean: 0.000, std: YYY.YYY
  After centering - GT mean: 0.000, std: BBB.BBB
  Per-element loss mean: CCC.CCC, max: DDD.DDD
  Loss value: EEEEE.EEE
```

**诊断关键点**：
- ✅ After centering 的 mean 应该都接近 0
- ✅ Per-element loss mean 应该在 10-100 mm 范围（合理）
- ❌ 如果 Per-element loss mean > 1000 → **尺度或单位问题**
- ❌ 如果 Loss value > 100000 → **重复中心化或数据错误**

### 单GPU调试（更快）

如果需要快速测试：

```bash
python train.py --config config/config.yaml
```

## 常见问题诊断

### 问题 1：验证指标仍然完全不变

**症状**：
```
Epoch 0: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm
Epoch 1: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm
Epoch 2: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm
```

**检查步骤**：
1. 查看日志中是否有 "Evaluated on XXXX samples from all X GPUs"
   - ❌ 如果没有 → 修改未生效，检查代码版本
   - ✅ 如果有 → 继续下一步

2. 检查 samples 数量是否正确
   - ❌ 如果 ~2083 → 仍然只评估了 1 个 GPU 的数据
   - ✅ 如果 ~8332 → 数据收集正常，问题在别处

3. 检查 2D loss 是否下降
   - ✅ 如果 2D loss 下降 → 模型在学习，但 3D 部分有问题
   - ❌ 如果 2D loss 不降 → 整个模型未训练，检查优化器/学习率

### 问题 2：3D joint loss 过高（~75000-80000）且不下降

**诊断流程**：

#### 步骤 1：检查数据统计（Batch 0-2）

运行训练查看第一个 epoch 的输出：

```bash
# 重点关注这些数值
GT 3D joints mean: ???  # 应该 < 500
GT centered by root mean: ???  # 应该 ≈ 0
```

**情况 A**：GT centered by root mean **不为 0**（例如 125.345）
→ **数据已经是相对坐标，loss 函数不应再次中心化**

**修复**：修改 `dataloader/ho3d_dataset.py`，确保返回绝对坐标：
```python
# 检查 __getitem__ 返回的 joints_3d_gt
# 应该是相对于相机坐标系的绝对位置，不是相对于 root joint
```

**情况 B**：GT 3D joints mean **过大**（例如 > 1000）
→ **单位转换错误，可能重复乘以 1000**

**修复**：检查 `dataloader/ho3d_dataset.py:797`：
```python
joint_xyz = data['joints_coord_cam'].reshape([21, 3])[HO3D2MANO, :] * 1000
# 检查原始数据单位，可能已经是 mm，不需要 * 1000
```

#### 步骤 2：启用 Loss 调试

```bash
export UTNET_DEBUG_LOSS=1
torchrun --nproc_per_node=4 train.py --config config/config.yaml
```

查看第一个 batch 的 loss 计算细节：

**情况 C**：Per-element loss mean **过大**（> 500）
→ **模型输出尺度错误**

**修复**：检查 `src/models/utnet.py:222, 314`：
```python
# 确认 MANO 输出单位
coarse_joints = coarse_joints * 1000.0  # meters → mm
# 检查 mano_output.joints 的原始单位是否真的是 meters
```

**情况 D**：Loss value **异常大**（> 100000）
→ **重复中心化**（GT 已经是相对坐标，loss 再次中心化导致错误）

**修复**：
1. 选项 1：修改数据加载器返回绝对坐标
2. 选项 2：修改 `Keypoint3DLoss` 移除中心化步骤（如果 GT 总是相对坐标）

### 问题 3：2D loss 下降，但 3D loss 不降

**可能原因**：
1. **损失权重不平衡**：检查 `config.yaml`
   ```yaml
   loss:
     w_2d: 0.01       # 2D 权重
     w_3d_joint: 0.05  # 3D 权重太小？
   ```
   
   尝试增加 3D 权重：
   ```yaml
   loss:
     w_2d: 0.01
     w_3d_joint: 0.5  # 增加到 0.5
   ```

2. **学习率问题**：模型的 2D 部分（ViT backbone）学习得很好，但 3D 部分（GCN refinement）学习率不够。

3. **梯度消失**：3D 部分的梯度被截断了。检查梯度裁剪是否过于激进：
   ```python
   # train.py:363
   torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
   # 尝试增加到 max_norm=5.0 或 10.0
   ```

## 下一步行动

### 立即操作

运行一个 epoch，查看调试输出：

```bash
cd /data0/users/Robert/linweiquan/UTNet
torchrun --nproc_per_node=4 train.py --config config/config.yaml 2>&1 | tee training_debug_$(date +%Y%m%d_%H%M%S).log
```

**预期时间**：约 20-30 分钟（一个 epoch）

### 根据输出决定

1. **如果验证指标开始变化** → ✅ Bug 修复成功
   - 继续训练，观察 loss 下降趋势
   - 如果 3D loss 仍不降，启用 Loss 调试模式

2. **如果验证指标仍然不变** → ❌ 需要进一步诊断
   - 检查代码是否正确更新
   - 检查分布式训练是否正常工作

3. **如果出现新的错误** → 🔧 修复错误
   - 可能是 `dist.gather` 相关的问题
   - 检查是否所有进程的数据形状一致

## 联系与支持

如果遇到问题，提供以下信息：

1. **日志文件**：`training_debug_*.log`
2. **调试输出**：第一个 epoch 的前 3 个 batch 的输出
3. **Loss 趋势**：前几个 epoch 的 loss 值
4. **环境信息**：GPU 数量、PyTorch 版本

## 参考文档

- 详细修改记录：`records/修复分布式验证评估和添加调试.md`
- WiLoR 对齐记录：`records/损失函数WiLoR对齐修改记录.md`
- 损失函数详解：`records/UTNet损失函数详解.md`

