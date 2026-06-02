# physics-informed-elasticity-2d
PINN solver for 2D linear elasticity on a unit square: joint displacement/stress prediction with PDE, constitutive, and BC losses; manufactured-solution benchmark; Adam + optional LBFGS; auto error metrics and field plots. PyTorch.


# Physics-Informed 2D Linear Elasticity (PINN)

基于物理信息神经网络（Physics-Informed Neural Network, PINN）的 **二维线弹性正问题** 求解代码。在单位正方形域 \([0,1]^2\) 上，网络同时预测位移场与应力场，并通过 PDE 残差、本构关系及边界条件进行联合训练。

## 问题描述

求解二维线弹性静力学问题，网络输出 5 个量：

| 输出 | 含义 |
|------|------|
| \(u_x, u_y\) | 位移分量 |
| \(\sigma_{xx}, \sigma_{yy}, \sigma_{xy}\) | 应力分量 |

控制方程包括：

- **平衡方程**：\(\nabla \cdot \sigma = f\)
- **本构关系**：Hooke 定律（Lamé 常数 \(\lambda, \mu\)）
- **几何关系**：应变-位移关系

本代码采用 **manufactured solution** 验证：解析解已知，可在边界上施加精确位移与应力，并在域内用解析体力检验 PDE 残差。

### 默认材料与几何参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 域 | \([0,1] \times [0,1]\) | 单位正方形 |
| \(\lambda\) | 1.0 | Lamé 第一常数 |
| \(\mu\) | 0.5 | 剪切模量 |
| \(Q\) | 4 | 解析解参数 |

## 方法概述

```
输入 (x, y) → 全连接网络 (tanh) → 输出 (u_x, u_y, σ_xx, σ_yy, σ_xy)
                    ↓
        ┌───────────┴───────────┐
   PDE 残差损失            边界条件损失
 (平衡 + 本构)         (解析 u, σ on ∂Ω)
        └───────────┬───────────┘
                    ↓
         Adam (20,000 epochs) → LBFGS 精修
```

- **配点**：域内 10,000 个随机点，每条边界 100 个点
- **优化器**：Adam + StepLR 自适应学习率，可选 LBFGS 二阶精修
- **损失函数**：均方误差（MSE）

## 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0（建议 CUDA 版本以加速训练）
- NumPy, Matplotlib, tqdm

## 安装

```bash
git clone <your-repo-url>
cd physics-informed-elasticity-2d

pip install -r requirements.txt
```

或手动安装：

```bash
pip install torch numpy matplotlib tqdm
```

## 使用方法

直接运行主脚本：

```bash
python physics-informed-elasticity-2d.py
```

训练过程中终端会显示各项损失，例如：

```
pde: 0.01234, bc: 0.00567, init: 0, total: 0.018007
```

训练结束后自动：

1. 在测试网格上计算 **相对 L2 误差**（Ux, Uy, Sxx, Syy, Sxy）
2. 生成并保存三张对比图（`Displacement.png`、`Stress.png`、`Strain.png`）

### 跳过 LBFGS 精修（更快）

在 `main()` 中修改：

```python
model.train(epochs=20000, lossf=get_pinn_loss, refine_with_lbfgs=False)
```

### 自定义网络与训练参数

在 `main()` 中可调整：

```python
layers = [2, 60, 60, 60, 60, 5]   # [输入, 隐藏层..., 输出]
model = PINNs(
    layers=layers,
    activation='tanh',
    sadap=True,          # Adam 学习率衰减
    initial_lr=0.001,
)
model.train(epochs=20000, lossf=get_pinn_loss)
```

## 输出文件

| 文件 | 内容 |
|------|------|
| `Displacement.png` | 位移场解析解 / 预测 / 绝对误差 |
| `Stress.png` | 应力场解析解 / 预测 / 绝对误差 |
| `Strain.png` | 应变场解析解 / 预测 / 绝对误差 |

## 项目结构

```
physics-informed-elasticity-2d/
├── physics-informed-elasticity-2d.py   # 主程序（模型、损失、训练、评估）
├── requirements.txt                  # Python 依赖
└── README.md
```

## 主要模块

| 模块 | 功能 |
|------|------|
| `NeuralNetwork` | 全连接前馈网络 |
| `PINNs` | 训练流程（Adam + LBFGS） |
| `pinn_pde_loss` | 域内 PDE 与本构残差 |
| `boundary_loss_BC` | 边界位移与应力约束 |
| `evaluate_and_plot` | L2 误差评估与可视化 |


