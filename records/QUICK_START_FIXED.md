# UTNet 修复后快速开始

## 🎯 已完成的修复（2025-12-07）

✅ **分布式验证评估 Bug**：修复了验证指标完全不变的问题  
🔧 **调试工具**：添加了诊断 3D loss 过高的调试代码  
📚 **完整文档**：创建了详细的修复记录和使用指南

## 🚀 立即开始训练

```bash
cd /data0/users/Robert/linweiquan/UTNet

# 标准 4-GPU 训练
torchrun --nproc_per_node=4 train.py --config config/config.yaml 2>&1 | tee training_fixed_$(date +%Y%m%d_%H%M%S).log
```

## 👀 观察修复效果

### 1. 验证指标应该会变化

**之前（Bug）**：
```
Epoch 0: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm
Epoch 1: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm  ← 完全相同
Epoch 2: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm  ← 完全相同
```

**修复后（预期）**：
```
Epoch 0: MPJPE: 102.152 mm, PA-MPJPE: 48.320 mm
  Evaluated on 8332 samples from all 4 GPUs  ← 新增信息
Epoch 1: MPJPE: 98.456 mm, PA-MPJPE: 46.123 mm   ← 指标变化
  Evaluated on 8332 samples from all 4 GPUs
Epoch 2: MPJPE: 89.234 mm, PA-MPJPE: 41.567 mm   ← 继续变化
  Evaluated on 8332 samples from all 4 GPUs
```

### 2. 自动调试输出（前 3 个 batch）

第一个 epoch 会自动显示数据统计：

```
[Debug Batch 0]
  RGB range: [-2.118, 2.640]
  Depth range: [0.000, 1.000]
  GT 3D joints mean: XXX.XXX, std: YYY.YYY
  Pred 3D joints mean: AAA.AAA, std: BBB.BBB
  GT root joint: tensor([...])
  Pred root joint: tensor([...])
  GT centered by root mean: 0.000, std: ZZZ.ZZZ  ← 关键！应该接近 0
```

## 🔍 诊断 3D Loss 问题

### 如果 3D loss 仍然很高（~75000）

查看调试输出中的 `GT centered by root mean`：

#### ✅ 如果 ≈ 0（例如 < 1e-5）
→ 数据正确，问题在其他地方，启用详细调试：

```bash
export UTNET_DEBUG_LOSS=1
torchrun --nproc_per_node=4 train.py --config config/config.yaml
```

#### ❌ 如果 ≠ 0（例如 > 1.0）
→ **数据已经是相对坐标，loss 不应再次中心化**

**快速修复**：
```python
# 编辑 dataloader/ho3d_dataset.py
# 找到返回 joints_3d_gt 的地方
# 确保返回的是绝对坐标（相对于相机），不是相对于 root joint
```

或者联系我获取具体修复代码。

## 📊 预期收敛指标

训练成功的标志：
- ✅ 2D loss 逐渐下降（已观察到：584 → 290）
- ✅ 3D joint loss 逐渐下降（目标：< 500）
- ✅ MPJPE 逐渐降低（目标：< 30 mm）
- ✅ PA-MPJPE 逐渐降低（目标：< 15 mm）

## 📖 详细文档

如果需要更多信息，请查看：

1. **实施总结**：`IMPLEMENTATION_SUMMARY.md` - 完整的修复说明
2. **调试指南**：`TRAINING_DEBUG_GUIDE.md` - 如何诊断问题
3. **修复记录**：`records/修复分布式验证评估和添加调试.md` - 技术细节

## ⚠️ 常见问题

### Q: 验证指标仍然不变？

A: 检查日志中是否有 "Evaluated on 8332 samples from all 4 GPUs"
- 如果**没有** → 代码未更新，检查 `train.py`
- 如果**有**但指标不变 → 可能模型根本没学习，检查优化器和学习率

### Q: 出现 "RuntimeError: NCCL error" 或通信错误？

A: 分布式通信问题，尝试：
```bash
# 方法 1: 减少 GPU 数量
torchrun --nproc_per_node=2 train.py --config config/config.yaml

# 方法 2: 单 GPU 模式（用于快速测试）
python train.py --config config/config.yaml
```

### Q: 3D loss 一直很高（>70000）？

A: 按照上面的诊断步骤：
1. 查看 `GT centered by root mean`
2. 如果 ≠ 0，修复数据加载器
3. 如果 ≈ 0，启用 `UTNET_DEBUG_LOSS=1` 进一步诊断

## 💡 下一步

1. **运行一个 epoch**（约 20-30 分钟）
2. **检查输出**：
   - ✅ "Evaluated on 8332 samples" 是否出现
   - ✅ 验证指标是否变化
   - 📊 数据统计是否正常
3. **根据结果决定**：
   - 如果一切正常 → 继续训练，观察收敛
   - 如果有问题 → 查看详细文档或提供日志寻求帮助

## 📞 需要帮助？

提供以下信息：
1. 训练日志（特别是第一个 epoch）
2. `GT centered by root mean` 的值
3. 验证指标的变化趋势

---

**祝训练顺利！** 🎉

