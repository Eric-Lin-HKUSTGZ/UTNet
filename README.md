# UTNet: Unified Transformer Network for Hand Pose Estimation

单阶段端到端的手部姿态估计网络，整合随机模态采样、ViT主干、NAF上采样和GCN精细化。

## 特性

- **随机模态采样** (OmniVGGT风格): 训练时随机丢弃Depth，提高模型鲁棒性
- **ViT主干**: 基于WiLoR的ViT架构，支持模态融合token
- **NAF多尺度上采样**: 内容自适应的多尺度特征上采样
- **GCN精细化**: 使用图卷积网络精细化MANO参数
- **单阶段端到端训练**: 所有模块在同一backward中联合优化

## 目录结构

```
UTNet/
├── config/
│   └── config.yaml              # 配置文件
├── dataloader/
│   ├── __init__.py
│   └── dex_ycb_dataset.py       # Dex-YCB数据集加载器
├── src/
│   ├── models/
│   │   ├── utnet.py             # 主模型
│   │   ├── backbone/
│   │   │   └── vit.py           # ViT主干
│   │   ├── tokenization/
│   │   │   ├── rgb_embed.py     # RGB token化
│   │   │   ├── depth_embed.py   # Depth token化
│   │   │   └── modality_fusion.py # 模态融合
│   │   ├── naf_upsampler.py     # NAF上采样模块
│   │   └── gcn_refinement.py   # GCN精细化模块
│   ├── losses/
│   │   └── loss.py              # 损失函数
│   └── utils/
│       ├── detection.py         # 手部检测
│       ├── geometry.py          # 几何工具
│       └── mano_utils.py        # MANO工具
└── train.py                     # 训练脚本
```

## 安装

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 下载MANO模型:
- 从 [MANO官网](https://mano.is.tue.mpg.de/) 下载MANO模型
- 将模型放在指定路径

3. 下载WiLoR预训练检测器:
- 检测器路径: `/data0/users/Robert/linweiquan/WiLoR/pretrained_models/detector.pt`
- 确保该路径存在

## 配置

编辑 `config/config.yaml` 设置:
- 数据集路径
- MANO模型路径
- 模型超参数
- 训练参数

## 训练

```bash
python train.py --config config/config.yaml
```

## 数据格式

使用Dex-YCB数据集，参考 `KeypointFusion/dataloader/loader.py` 的数据格式。

数据集应包含:
- RGB图像
- Depth图像 (可选)
- MANO参数标注
- 2D/3D关键点标注

## 模型架构

1. **数据采样和模态配置**: 随机模态采样 (OmniVGGT风格)
2. **手部检测与裁剪**: 使用WiLoR预训练检测器
3. **Token化和模态融合**: RGB + Depth token融合
4. **ViT主干 + 粗参数预测**: 输出粗MANO参数
5. **NAF-GPPR精细化**: 多尺度上采样 + GCN精细化
6. **MANO解码和最终输出**: 最终手部网格和关键点

## 损失函数

- 2D关键点重投影损失
- 3D关节损失
- 3D顶点损失 (可选)
- MANO参数正则
- 辅助损失 (粗预测监督)

## 参考

- WiLoR: `/data0/users/Robert/linweiquan/WiLoR`
- OmniVGGT: `/data0/users/Robert/linweiquan/OmniVGGT`
- NAF: `/data0/users/Robert/linweiquan/NAF`
- KeypointFusion: `/data0/users/Robert/linweiquan/KeypointFusion`
