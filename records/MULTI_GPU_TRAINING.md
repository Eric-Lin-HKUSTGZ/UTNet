# UTNet 多GPU训练指南

## 概述

UTNet现在支持多GPU分布式训练，使用PyTorch的`DistributedDataParallel` (DDP)实现。

## 使用方法

### 方法1: 使用torchrun (推荐，PyTorch 1.9+)

```bash
# 使用4个GPU训练
torchrun --nproc_per_node=4 --master_port=29500 train.py --config config/config.yaml

# 使用8个GPU训练
torchrun --nproc_per_node=8 --master_port=29500 train.py --config config/config.yaml
```

### 方法2: 使用提供的脚本

```bash
# 使用默认4个GPU
bash scripts/train_multi_gpu.sh

# 指定GPU数量和配置文件
bash scripts/train_multi_gpu.sh 8 config/config.yaml
```

### 方法3: 使用torch.distributed.launch (旧版PyTorch)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --config config/config.yaml
```

## 单GPU训练

单GPU训练仍然支持，无需修改：

```bash
python train.py --config config/config.yaml
```

或者指定GPU：

```bash
python train.py --config config/config.yaml --gpu 0
```

## 关键特性

1. **自动分布式初始化**: 代码会自动检测是否在分布式环境中运行
2. **数据并行**: 使用`DistributedSampler`确保每个GPU处理不同的数据子集
3. **梯度同步**: DDP自动在所有GPU之间同步梯度
4. **日志和保存**: 只在rank 0 (主进程)上保存checkpoint和写入TensorBoard日志
5. **Loss聚合**: 自动聚合所有GPU的loss值

## 配置说明

多GPU训练会自动：
- 将batch size分配到各个GPU（总batch size = config中的batch_size × num_gpus）
- 使用DistributedSampler进行数据采样
- 在rank 0上保存模型和日志

## 注意事项

1. **Batch Size**: 如果使用4个GPU，实际总batch size = config中的batch_size × 4
2. **学习率**: 可能需要根据GPU数量调整学习率（通常线性缩放：lr × num_gpus）
3. **内存**: 每个GPU的内存使用量相同，确保有足够的GPU内存
4. **端口**: 如果多个训练任务同时运行，需要修改`--master_port`避免冲突

## 故障排除

### 端口冲突
如果遇到端口占用错误，修改`--master_port`：
```bash
torchrun --nproc_per_node=4 --master_port=29501 train.py --config config/config.yaml
```

### CUDA out of memory
- 减小config中的batch_size
- 或者使用更少的GPU

### 进程挂起
确保所有GPU都可用：
```bash
nvidia-smi  # 检查GPU状态
```

## 性能优化建议

1. **数据加载**: 增加`num_workers`可以提高数据加载速度
2. **混合精度**: 可以添加AMP (Automatic Mixed Precision)进一步加速
3. **梯度累积**: 如果GPU内存不足，可以使用梯度累积来模拟更大的batch size

