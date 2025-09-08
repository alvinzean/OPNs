# OPNTensorMatrix

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](#)  
[![License](https://img.shields.io/badge/license-MIT-green)](#)  
> 使用 PyTorch 实现的 OPN（Ordered Pair Number）张量矩阵类，支持专属数学运算、比较、广播、矩阵操作等功能。

## 目录
- [项目简介](#项目简介)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [核心用法实例](#核心用法实例)
- [内部结构一览](#内部结构一览)
- [贡献指南](#贡献指南)
- [内部方法与私有方法说明](#内部方法与私有方法说明)

---

## 项目简介
`OPNTensorMatrix` 是一个基于 PyTorch 的张量类，它将每个元素表示为 `(a, b)` 的有序对，支持以下特点：
- 自定义的 OPN 运算（乘法、幂、逆、范数等）
- 支持广播机制、比较操作（`<`, `>`, `==` 等）
- 高效的矩阵与向量乘、元素级操作
- 完整的初始化与设备支持（CPU/GPU）  
此项目适合研究者、数学运算爱好者，以及希望在 PyTorch 上实现复杂数结构的开发者。

---

## 特性
- **初始化支持多种输入**：`torch.Tensor`、NumPy 数组、Python 列表、`OPNTensor` 或其他 `OPNTensorMatrix`
- **核心数学运算**：包括 OPN 专属的乘法、幂运算（支持正负整数幂）、逆运算、范数、共轭
- **矩阵运算支持**：包括标准矩阵乘法、矩阵与向量乘法、向量点积等多种形态
- **广播与比较**：支持形状广播计算，包含 `<`, `>`, `==`, `!=`，并可设置容差阈值
- **实用工具**：如 `sum`, `argmax`, `argmin`, `clip`, `to_numpy`, `to(device)` 等方法便于操作和调试

---

## 安装
```bash
pip install torch numpy
```
项目本身为本地模块，无需额外安装：
```
git clone https://github.com/your-repo/OPNTensorMatrix.git
cd OPNTensorMatrix
```

---
## 快速开始
```
from src.gpuopns.opn import OPNTensor
from your_module import OPNTensorMatrix

# 初始化样例
data = [[1.0, 2.0], [3.0, 4.0]]
m = OPNTensorMatrix(data)

# 矩阵乘法
n = OPNTensorMatrix([[5.0, 6.0], [7.0, 8.0]])
prod = m @ n

# 幂运算
m2 = m ** 2

# 逆运算
inv = m.inverse()

# 转换为 NumPy
arr = m.to_numpy()
```
---
## 核心用法实例
| 用例    | 示例代码                                                              | 说明           |
| ----- | ----------------------------------------------------------------- | ------------ |
| 幂运算   | `m2 = m ** 3`                                                     | 支持标量或按元素的幂运算 |
| 矩阵乘法  | `C = A @ B`                                                       | 从标准矩阵到点积都支持  |
| 比较操作  | `mask = m > n`                                                    | 支持广播比较与容差机制  |
| 累积与检索 | `s = OPNTensorMatrix.sum(m)`<br>`idx = OPNTensorMatrix.argmax(m)` | 元素求和与最大值索引   |

---

## 主要方法/属性说明：

- 初始化：支持多类型输入，支持指定 device（CPU/GPU）

- 运算符重载：__add__, __mul__, __pow__, __matmul__, inverse, norm, conjugate, 等

- 比较与广播：gt, lt, eq, broadcast_add, broadcast_mul

- 工具函数：sum, argmax, argmin, clip, to_numpy, to(), detach, clone

- 私有：_broadcast_compare 管理比较操作的形状与容错

---

## 贡献指南

欢迎贡献！你可以：

- 提交 Issue 反馈问题

- Fork 本项目并提交 Pull Request

- 优化性能（如向量化、数值稳定性）

- 提供更多测试用例、增强文档覆盖

## 内部方法与私有方法说明

下面列出 `OPNTensorMatrix` 类中的所有非公共接口方法，并说明它们的功能与使用注意点。这部分有助于理解实现细节，便于调试和贡献。

| 方法名 | 类型 | 功能说明 | 注意事项 |
|--------|------|----------|----------|
| `_broadcast_compare` | 私有 | 统一广播 `self` 与 `other` 的 `a` 和 `b` 分量，用于比较运算前的数据对齐 | 检查类型是否为 `OPNTensorMatrix`，若不可广播则抛出 `ValueError` |
| `eq`, `ne`, `gt`, `lt`, `ge`, `le` | 公共比较 | 元素级比较（支持广播），比较逻辑基于 `(a + b)` 与 `a` 分量的优先级，并使用容差阈值 (`__zero_threshold__`) 控制精度 | 内部依赖 `_broadcast_compare`, 避免重复实现；请了解比较规则以正确使用 |
| `clip` | 公共裁剪 | 元素级上下界裁剪，可接受标量、`OPNTensor` 或 `OPNTensorMatrix` 输入类型，会对输入进行自动包装与广播 | 原对象 `.data` 会被修改，存在副作用；若需保留原值，请用 `.copy()` |
| `clip_c` | 特殊裁剪 | 先将每个元素 `c = a + b` 限制到 [min_v, max_v]，再按比例调整分量值，以保持方向一致 | 特别处理 `c == 0` 避免除零错误 |
| `to_opn_list` | 公共辅助 | 将当前矩阵逐元素转换为 `OPNTensor` 对象列表（嵌套结构） | 合适用于进一步逐元素操作，但大型张量可能效率较低 |
| `to_numpy` | 公共辅助 | 将内部 `torch.Tensor` 转为 NumPy 数组（CPU） | 如果数据在 GPU 上需 `.cpu()`，该方法自动处理 |
| `to` | 公共 | 将数据移动到指定设备（如 `"cuda"`） | 返回新的 `OPNTensorMatrix` 对象，不改变原对象 |
| `detach` | 公共 | 从计算图中分离张量，用于取消梯度追踪 | 返回新实例，适合在训练结束后保存结果 |
| `clone` | 公共 | 克隆当前对象（数据共享），但子对象不共享内存 | 若需完全复制，请结合 `.detach()` |
| `copy`, `__copy__` | 公共 | 返回数据完整克隆的新对象 | `_copy__` 是 Python 规范的接口，`copy()` 则是简写 |
| `__getitem__` | 公共 | 支持索引切片操作（不允许索引到最后一维 OPN 分量） | 若切片仅剩一维，返回包含该一行的 OPN 张量 |
| `__setitem__` | 公共 | 支持通过索引将数据赋值为 OPN 格式 | 输入类型多样，若形状不兼容或类型错误会提示 |
| `__repr__` / `__str__` | 公共 | 提供可读的对象描述，包含形状、设备和数据 | `__str__` 会打印实际数据，适合调试 |
| `__len__` | 公共 | 返回矩阵第一维长度，即行数 | 依赖 `.shape[0]` |
| 数学运算重载 (`__add__`, `__sub__`, `__neg__`, `__mul__`, `__rmul__`, `__matmul__`, `__pow__`) | 公共 | 实现 OPN 专属运算，如加法、乘法、矩阵乘法、幂运算等 | 多涉及广播、不同维度组合，仔细阅读注释很重要 |
| `inverse` | 公共 | 逐元素计算 (a, b) 的逆元 `(a/(a²−b²), −b/(a²−b²))` | 若 `a² ≈ b²` 会抛出 `ZeroDivisionError`，与 `__zero_threshold__` 兼容 |
| `norm` | 公共 | 计算每个 OPN 元素的范数：`sqrt(a² − b²)` | 当数值接近零可能出现负值，建议在用户层加入 `clamp(min=0)` |
| `conjugate` | 公共 | 共轭操作：将 b 分量取反 | 小心会原地修改 `.data`，建议在链式调用时显式 `.clone()` |
| `transpose` | 公共 | 矩阵维度转置，保持 OPN 最后维度 | 如果涉及 GPU 与 CPU 之间要注意设备一致性 |
| 属性 `shape`, `ndim`, `device` | 公共 | `shape`: 数据（除最后一维）形状；`ndim`: 维数（剔除 OPN 分量） | 有助于动态判断运算逻辑适用性 |
| `broadcast_add` | 公共对广播加法 | 支持将标量、单个 `OPNTensor` 或形状兼容的 `OPNTensorMatrix` 加到当前对象中 | 标量加法只作用于 `a` 分量；若形状不匹配会抛出 `ValueError`；返回新对象，不修改原矩阵 |
| `broadcast_mul` | 公共广播乘法 | 支持将标量、`OPNTensor` 或形状兼容的 `OPNTensorMatrix` 广播乘到当前对象 | 内部处理类似 OPN 乘法逻辑，自动处理广播；若类型或形状不支持，会抛出异常；返回新对象 |
| `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` | 运算符重载（公共） | 依次调用对应的 `eq`, `ne`, `lt`, `le`, `gt`, `ge` 方法，实现 Python 常用比较运算 | 这些只是为了兼容 Python 语法糖，具体逻辑在相应方法中实现 |
| `__abs__`, `abs` | 公共数学运算 | 返回按元素绝对值的新矩阵（若 `< 0`，自动乘以 `-1`） | 通过广播构造临时矩阵进行判断，不修改原对象；逻辑与 `abs()` 方法一致 |
| `dot` | 公共 | 实现 OPN 张量乘法操作，支持标量或矩阵间按 OPN 规则的乘法 | 与 `__mul__` 实现类似；支持广播；返回新对象 |