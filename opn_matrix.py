from torch.cuda import device

from .opn import OPNTensor
import numpy as np
import torch

class OPNTensorMatrix:
    """
    OPN（有序对数）张量的矩阵类，支持批量操作。

    属性:
        data (torch.Tensor): 底层张量存储，形状为 (..., 2)
        device (str/torch.device): 张量所在的设备
        __zero_threshold__ (float): 零值比较的阈值
    """

    __zero_threshold__ = 1e-8
    __inf_threshold__ = 1e8

    @classmethod
    def set_zero_threshold(cls, threshold):
        """设置全局零值比较阈值"""
        cls.__zero_threshold__ = threshold

    @classmethod
    def set_inf_threshold(cls, threshold):
        """设置全局零值比较阈值"""
        cls.__inf_threshold__ = threshold

    @classmethod
    def unique(cls, opn_matrix):
        """
        对形状为(n, 2)的torch矩阵进行unique操作
        Args:
            opn_matrix: torch.Tensor, 形状为(n, 2)的矩阵
        Returns:
            unique_tensor: 去重后的矩阵
            inverse_indices: 原始矩阵中每个元素在去重后矩阵中的索引
        """
        assert opn_matrix.data.ndim == 2 and opn_matrix.data.shape[-1] == 2, "输入必须是(n,2)形状"
        return cls._unique_gpu_safe(opn_matrix)

    @classmethod
    def bincount(cls, opn_matrix):
        """
        统计每个唯一元素出现的次数
        Args:
            opn_matrix: OPNTensorMatrix, 形状为(n, 2)的矩阵
        Returns:
            unique_tensor: 去重后的矩阵
            counts: 每个唯一元素出现的次数
        """
        assert opn_matrix.data.ndim == 2 and opn_matrix.data.shape[-1] == 2, "输入必须是(n,2)形状"
        return cls._bincount_gpu_safe(opn_matrix)

    @classmethod
    def _bincount_gpu_safe(cls, opn_matrix):
        unique_tensor, inverse_indices = cls._unique_gpu_safe(opn_matrix)
        counts = torch.bincount(inverse_indices)
        return unique_tensor, counts

    @classmethod
    def _unique_gpu_safe(cls, opn_matrix):
        """安全的去重方法"""
        seen = {}
        unique_list = []
        inverse_indices = []
        for i, row in enumerate(opn_matrix.data):
            # 将tensor转换为可哈希的元组
            row_tuple = tuple(row.tolist())
            if row_tuple not in seen:
                seen[row_tuple] = len(unique_list)
                unique_list.append(row)
            inverse_indices.append(seen[row_tuple])
        unique_tensor = torch.stack(unique_list)
        inverse_indices = torch.tensor(inverse_indices, device=opn_matrix.device)
        return OPNTensorMatrix(unique_tensor,device=opn_matrix.device), inverse_indices

    @classmethod
    def cat(cls, opn_matrix_list):
        for opn_matrix in opn_matrix_list:
            if not isinstance(opn_matrix, OPNTensorMatrix):
                raise TypeError(f"不支持的opn_matrix类型: {type(opn_matrix)}")
        return OPNTensorMatrix(torch.cat([opn_matrix.data for opn_matrix in opn_matrix_list]),device=opn_matrix_list[0].device)


    @classmethod
    def zero(cls, shape=(), device="cpu"):
        """创建全零OPN张量矩阵"""
        if isinstance(shape, int):
            shape = (shape,)
        return cls(torch.zeros((*shape, 2), device=device, dtype=torch.float64),device=device)

    @classmethod
    def argsort(cls, opn_matrix):
        assert opn_matrix.data.ndim == 2 and opn_matrix.data.shape[-1] == 2, "输入必须是(n,2)形状"
        a = opn_matrix.data[:, 0]
        b = opn_matrix.data[:, 1]
        sum_ab = a + b
        compare_key = torch.stack([-sum_ab, a], dim=-1)
        idx2 = torch.argsort(compare_key[:, 1], stable=True).to(device=opn_matrix.device)
        tensor = compare_key[idx2]
        idx1 = torch.argsort(tensor[:, 0], stable=True).to(device=opn_matrix.device)
        return idx2[idx1]

    @classmethod
    def argmax(cls, opn_matrix):
        return cls.argsort(opn_matrix)[-1]

    @classmethod
    def argmin(cls, opn_matrix):
        return cls.argsort(opn_matrix)[0]

    @classmethod
    def sum(cls, opn_matrix, dim=None, keepdim=False):
        if dim is None:
            sum_a = torch.sum(opn_matrix.data[..., 0])
            sum_b = torch.sum(opn_matrix.data[..., 1])
        else:
            sum_a = torch.sum(opn_matrix.data[..., 0], dim=dim, keepdim=keepdim)
            sum_b = torch.sum(opn_matrix.data[..., 1], dim=dim, keepdim=keepdim)
        if dim is None or not keepdim:
            result_data = torch.stack([sum_a, sum_b])
        else:
            result_data = torch.stack([sum_a, sum_b], dim=-1)
        return cls(result_data, device=opn_matrix.device)

    @classmethod
    def maximum(cls, opn_matrix, val: OPNTensor):
        val_a = torch.full_like(opn_matrix.data[..., 0], val.a.item())
        val_b = torch.full_like(opn_matrix.data[..., 1], val.b.item())
        val_data = torch.stack([val_a, val_b], dim=-1)
        sum_matrix = opn_matrix.data[..., 0] + opn_matrix.data[..., 1]
        sum_val = val_data[..., 0] + val_data[..., 1]
        a_matrix = opn_matrix.data[..., 0]
        a_val = val_data[..., 0]
        mask = sum_matrix < sum_val
        mask = mask | ((sum_matrix == sum_val) & (a_matrix < a_val))
        result_data = torch.where(mask.unsqueeze(-1), val_data, opn_matrix.data)
        return cls(result_data, device=opn_matrix.device)

    @classmethod
    def clip(cls, opn_matrix, min_v=None, max_v=None):
        if min_v is None and max_v is None:
            return opn_matrix.to(opn_matrix.device)
        if min_v is not None:
            if isinstance(min_v, (int, float)):
                min_data = torch.full_like(opn_matrix.data, min_v)
                min_v = cls(min_data, device=opn_matrix.device)
            elif isinstance(min_v, OPNTensor):
                min_data = torch.stack([
                    min_v.a * torch.ones_like(opn_matrix.data[..., 0]),
                    min_v.b * torch.ones_like(opn_matrix.data[..., 1])
                ], dim=-1)
                min_v = cls(min_data, device=opn_matrix.device)
            elif not isinstance(min_v, OPNTensorMatrix):
                raise TypeError(f"不支持的min_v类型: {type(min_v)}")
        if max_v is not None:
            if isinstance(max_v, (int, float)):
                max_data = torch.full_like(opn_matrix.data, max_v)
                max_v = cls(max_data, device=opn_matrix.device)
            elif isinstance(max_v, OPNTensor):
                max_data = torch.stack([
                    max_v.a * torch.ones_like(opn_matrix.data[..., 0]),
                    max_v.b * torch.ones_like(opn_matrix.data[..., 1])
                ], dim=-1)
                max_v = cls(max_data, device=opn_matrix.device)
            elif not isinstance(max_v, OPNTensorMatrix):
                raise TypeError(f"不支持的max_v类型: {type(max_v)}")
        result = opn_matrix
        if min_v is not None:
            if min_v.shape != opn_matrix.shape:
                try:
                    min_v = cls(torch.broadcast_to(min_v.data, opn_matrix.data.shape),device=opn_matrix.device)
                except RuntimeError as e:
                    raise ValueError(f"min_v形状{min_v.shape}无法广播到{opn_matrix.shape}") from e
            mask = result.lt(min_v)
            result.data[mask] = min_v.data[mask]
        if max_v is not None:
            if max_v.shape != opn_matrix.shape:
                try:
                    max_v = cls(torch.broadcast_to(max_v.data, opn_matrix.data.shape),device=opn_matrix.device)
                except RuntimeError as e:
                    raise ValueError(f"max_v形状{max_v.shape}无法广播到{opn_matrix.shape}") from e
            mask = result.gt(max_v)
            result.data[mask] = max_v.data[mask]
        return result

    @classmethod
    def clip_c(cls, opn_matrix, min_v, max_v):
        c = opn_matrix.data[..., 0] + opn_matrix.data[..., 1]
        d = torch.where(c > max_v, max_v, c)
        e = torch.where(d < min_v, min_v, d)
        safe_c = torch.where(c == 0, torch.ones_like(c), c)
        return opn_matrix * e.unsqueeze(-1) / safe_c.unsqueeze(-1)

    @classmethod
    def inf(cls, shape=(), device="cpu"):
        """创建全inf OPN张量矩阵"""
        if isinstance(shape, int):
            shape = (shape,)
        return cls(torch.full((*shape, 2), -1 * OPNTensorMatrix.__inf_threshold__, device=device, dtype=torch.float64), device=device)

    @classmethod
    def one(cls, shape=(), device="cpu"):
        """创建单位OPN张量矩阵 (0, -1)"""
        if isinstance(shape, int):
            shape = (shape,)
        data = torch.zeros((*shape, 2), device=device, dtype=torch.float64)
        data[..., 1] = -1  # 设置b分量为-1
        return cls(data, device=device)

    def __init__(self, data, device=None):
        """
        初始化OPNTensorMatrix，支持以下输入:
        - torch.Tensor (..., 2)
        - np.ndarray (..., 2)
        - OPNTensor兼容数据的列表
        - 另一个OPNTensorMatrix
        """
        self.device = device if device is not None else "cpu"

        if isinstance(data, OPNTensorMatrix):
            self.data = data.data.to(device=self.device)
        elif isinstance(data, torch.Tensor):
            if data.shape[-1] == 2:
                self.data = data.to(device=self.device, dtype=torch.float64)
            else:
                raise ValueError(f"最后一维大小必须为2，当前形状 {data.shape}")
        elif isinstance(data, np.ndarray):
            if data.shape[-1] == 2:
                self.data = torch.from_numpy(data).to(device=self.device, dtype=torch.float64)
            else:
                raise ValueError(f"最后一维大小必须为2，当前形状 {data.shape}")
        elif isinstance(data, (list, tuple)):
            try:
                array_data = np.array(data, dtype=np.float32)
                if array_data.shape[-1] == 2:
                    self.data = torch.from_numpy(array_data).to(device=self.device, dtype=torch.float64)
                else:
                    # 尝试解释为OPNTensor列表
                    opn_list = [OPNTensor(d, device=self.device) for d in data]
                    stacked = torch.stack([opn.data for opn in opn_list])
                    self.data = stacked.to(device=self.device)
            except Exception as e:
                raise ValueError(f"无法将输入列表转换为张量: {e}")
        else:
            raise TypeError(f"不支持输入类型 {type(data)}，预期 torch.Tensor, np.ndarray 或 list")

    @property
    def shape(self):
        """返回矩阵形状（不包括最后一维）"""
        return self.data.shape[:-1]

    @property
    def ndim(self):
        """维度数量（不包括OPN维度）"""
        return self.data.ndim - 1

    @property
    def is_cuda(self):
        return self.device.lower() == 'cuda'

    @property
    def dtype(self):
        return self.data.dtype

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return OPNTensorMatrix(self.data.clone(), device=self.device)

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"OPNTensorMatrix(shape={self.shape}, device='{self.device}')"

    def __str__(self):
        return (f"OPNTensorMatrix(\n"
                f"  device={self.device},\n"
                f"  shape={self.shape},\n"
                f"  dtype={self.data.dtype},\n"
                f"  data=\n{self.data}\n)")

    def __format__(self, format_spec):
        return f"{self.data}"

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        # 确保不索引最后一维
        if len(idx) >= self.data.ndim:
            raise IndexError("不能直接索引OPN分量（最后一维）")

        sliced = self.data[idx]
        if len(sliced.shape) == 1:
            return OPNTensorMatrix(sliced.unsqueeze(0), device=self.device)
        return OPNTensorMatrix(sliced, device=self.device)

    def __setitem__(self, idx, value):
        if isinstance(value, OPNTensorMatrix):
            value_data = value.data.to(self.device)
        elif isinstance(value, torch.Tensor):
            value_data = value.to(self.device)
        elif isinstance(value, (list, np.ndarray, OPNTensor)):
            value_data = OPNTensorMatrix(value, device=self.device).data
        else:
            raise TypeError(f"不支持设置类型 {type(value)}")

        if value_data.shape[-1] != 2:
            raise ValueError("最后一维大小必须为2")
        try:
            self.data[idx] = value_data
        except:
            raise ValueError(f"赋值维度不同 目标维度:{self.data[idx].shape}, 右维度:{value_data.shape}")

    # 数学运算
    def __neg__(self):
        return OPNTensorMatrix(-self.data, device=self.device)

    def __add__(self, other):
        if isinstance(other, OPNTensorMatrix):
            try:
                return OPNTensorMatrix(self.data + other.data, device=self.device)
            except:
                raise ValueError(f"相加维度不同 左元素:{self.shape}, 右元素:{other.shape}")
        elif other == 0:
            return OPNTensorMatrix(self.data, device=self.device)
        else:
            raise TypeError(f"无法将OPNTensorMatrix与 {type(other)} 相加")

    def __sub__(self, other):
        if isinstance(other, OPNTensorMatrix):
            return OPNTensorMatrix(self.data - other.data, device=self.device)
        elif other == 0:
            return OPNTensorMatrix(self.data, device=self.device)
        else:
            raise TypeError(f"无法从OPNTensorMatrix中减去 {type(other)}")

    def __mul__(self, other):
        """矩阵乘法或标量乘法"""
        if isinstance(other, (int, float, torch.Tensor, np.number)):
            # 标量乘法
            return OPNTensorMatrix(self.data * other, device=self.device)
        elif isinstance(other, OPNTensorMatrix):
            # OPN乘法: (a,b)*(c,d) = (-ad-bc, -ac-bd)
            a, b = self.data[..., 0], self.data[..., 1]
            c, d = other.data[..., 0], other.data[..., 1]

            # 如果需要则广播形状
            if a.shape != c.shape:
                a, b, c, d = torch.broadcast_tensors(a, b, c, d)

            new_a = -a * d - b * c
            new_b = -a * c - b * d
            result = torch.stack([new_a, new_b], dim=-1)
            return OPNTensorMatrix(result, device=self.device)
        else:
            raise TypeError(f"无法将OPNTensorMatrix与 {type(other)} 相乘")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        """OPNTensorMatrix 的矩阵乘法运算

        支持以下形状组合:
        - (m,n,2) @ (n,k,2) -> (m,k,2)  # 标准矩阵乘法
        - (m,n,2) @ (n,2) -> (m,2)      # 矩阵与向量乘法
        - (m,2) @ (m,n,2) -> (n,2)      # 向量与矩阵乘法
        - (n,2) @ (n,2) -> (2)          # 向量点积

        参数:
            other: OPNTensorMatrix 实例

        返回:
            OPNTensorMatrix: 乘法结果

        异常:
            TypeError: 如果 other 不是 OPNTensorMatrix 类型
            ValueError: 如果矩阵形状不兼容
        """
        if not isinstance(other, OPNTensorMatrix):
            raise TypeError(f"不能与 {type(other)} 类型进行矩阵乘法")

        # 获取输入形状并确保最后一维为2
        shape_self = self.data.shape
        shape_other = other.data.shape

        # 处理不同维度的输入
        if shape_self[-1] != 2 or shape_other[-1] != 2:
            raise ValueError(f"最后一维大小必须为2, 输入形状为 {shape_self} 和 {shape_other}")

        # 统一提升维度以便处理
        self_data = self.data if self.data.ndim >= 2 else self.data.unsqueeze(0)
        other_data = other.data if other.data.ndim >= 2 else other.data.unsqueeze(0)

        # 处理四种情况
        if self_data.ndim == 3 and other_data.ndim == 3:
            # 情况1: (m,n,2) @ (n,k,2) -> (m,k,2)
            m, n1, _ = self_data.shape
            n2, k, _ = other_data.shape
            if n1 != n2:
                raise ValueError(f"形状 {shape_self} 和 {shape_other} 不匹配")

            a_self, b_self = self_data[..., 0], self_data[..., 1]
            a_other, b_other = other_data[..., 0], other_data[..., 1]

            aa = a_self @ a_other
            ab = a_self @ b_other
            ba = b_self @ a_other
            bb = b_self @ b_other

        elif self_data.ndim == 3 and other_data.ndim == 2:
            # 情况2: (m,n,2) @ (n,2) -> (m,2)
            m, n1, _ = self_data.shape
            n2, _ = other_data.shape
            if n1 != n2:
                raise ValueError(f"形状 {shape_self} 和 {shape_other} 不匹配")

            a_self, b_self = self_data[..., 0], self_data[..., 1]
            a_other, b_other = other_data[..., 0], other_data[..., 1]

            aa = (a_self * a_other.unsqueeze(0)).sum(dim=1)
            ab = (a_self * b_other.unsqueeze(0)).sum(dim=1)
            ba = (b_self * a_other.unsqueeze(0)).sum(dim=1)
            bb = (b_self * b_other.unsqueeze(0)).sum(dim=1)

        elif self_data.ndim == 2 and other_data.ndim == 3:
            # 情况3: (m,2) @ (m,n,2) -> (n,2)
            m1, _ = self_data.shape
            m2, n, _ = other_data.shape
            if m1 != m2:
                raise ValueError(f"形状 {shape_self} 和 {shape_other} 不匹配")

            a_self, b_self = self_data[..., 0], self_data[..., 1]
            a_other, b_other = other_data[..., 0], other_data[..., 1]

            aa = (a_self.unsqueeze(1) * a_other).sum(dim=0)
            ab = (a_self.unsqueeze(1) * b_other).sum(dim=0)
            ba = (b_self.unsqueeze(1) * a_other).sum(dim=0)
            bb = (b_self.unsqueeze(1) * b_other).sum(dim=0)

        elif self_data.ndim == 2 and other_data.ndim == 2:
            # 情况4: (n,2) @ (n,2) -> (2)
            n1, _ = self_data.shape
            n2, _ = other_data.shape
            if n1 != n2:
                raise ValueError(f"形状 {shape_self} 和 {shape_other} 不匹配")

            a_self, b_self = self_data[..., 0], self_data[..., 1]
            a_other, b_other = other_data[..., 0], other_data[..., 1]

            aa = (a_self * a_other).sum()
            ab = (a_self * b_other).sum()
            ba = (b_self * a_other).sum()
            bb = (b_self * b_other).sum()
        else:
            raise ValueError(f"不支持的形状组合: {shape_self} 和 {shape_other}")

        # 按照 OPN 乘法规则组合结果
        new_a = -ab - ba  # -ad-bc
        new_b = -aa - bb  # -ac-bd

        # 构建结果并恢复原始维度
        result = torch.stack([new_a, new_b], dim=-1)
        if shape_self == (2,) and shape_other == (2,):  # 向量点积情况
            result = result.squeeze()

        return OPNTensorMatrix(result, device=self.device)

    def transpose(self, dim0=-2, dim1=-1):
        """转置矩阵维度"""
        # 转置矩阵维度
        if dim0 < 0:
            dim0 -= 1
        if dim1 < 0:
            dim1 -= 1
        data = self.data.transpose(dim0, dim1)
        return OPNTensorMatrix(data, device=self.device)

    def conjugate(self):
        """共轭操作: (a, b) -> (a, -b)"""
        data = self.data
        data[..., 1] = -data[..., 1]
        return OPNTensorMatrix(data, device=self.device)

    def norm(self):
        """计算每个OPN元素的范数: sqrt(a² - b²)"""
        a, b = self.data[..., 0], self.data[..., 1]
        return torch.sqrt(a ** 2 - b ** 2)

    def is_zero(self):
        """检查元素是否为零 (a == ±b)"""
        a, b = self.data[..., 0], self.data[..., 1]
        return torch.isclose(a, b, rtol=self.__zero_threshold__) | torch.isclose(a, -b, rtol=self.__zero_threshold__)

    def to_opn_list(self):
        """转换为OPNTensor对象列表"""
        def recursive_convert(data):
            if data.ndim == 1:
                return OPNTensor(data, device=self.device)
            return [recursive_convert(x) for x in data]
        return recursive_convert(self.data)

    def to_numpy(self):
        """转换为numpy数组"""
        return self.data.cpu().numpy()

    def to(self, device):
        """移动到不同设备"""
        return OPNTensorMatrix(self.data.to(device), device = device)

    def detach(self):
        """从计算图中分离"""
        return OPNTensorMatrix(self.data.detach(), device=self.device)

    def clone(self):
        """创建副本"""
        return OPNTensorMatrix(self.data, device=self.device)

    def _broadcast_compare(self, other):
        """内部方法：处理广播比较的公共逻辑"""
        if not isinstance(other, OPNTensorMatrix):
            raise TypeError(f"比较对象必须为OPNTensorMatrix, 实际为 {type(other)}")

        # 获取输入数据并确保至少是2维
        self_data = self.data if self.data.ndim >= 2 else self.data.unsqueeze(0)
        other_data = other.data if other.data.ndim >= 2 else other.data.unsqueeze(0)

        try:
            # 广播两个张量的分量
            a_self, a_other = torch.broadcast_tensors(self_data[..., 0], other_data[..., 0])
            b_self, b_other = torch.broadcast_tensors(self_data[..., 1], other_data[..., 1])
        except RuntimeError as e:
            raise ValueError(f"形状无法广播比较: {self.data.shape} vs {other.data.shape}") from e
        a_self = a_self.to(self.device)
        b_self = b_self.to(self.device)
        a_other = a_other.to(self.device)
        b_other = b_other.to(self.device)

        return a_self, b_self, a_other, b_other

    def eq(self, other):
        """
        元素级相等比较（支持广播）
        """
        a_self, b_self, a_other, b_other = self._broadcast_compare(other)
        a_eq = torch.isclose(a_self, a_other, rtol=self.__zero_threshold__)
        b_eq = torch.isclose(b_self, b_other, rtol=self.__zero_threshold__)
        return torch.logical_and(a_eq, b_eq)

    def ne(self, other):
        """
        元素级不等比较（支持广播）
        """
        return torch.logical_not(self.eq(other))

    def gt(self, other):
        """
        元素级大于比较（支持广播）
        """
        a_self, b_self, a_other, b_other = self._broadcast_compare(other)
        self_sum = a_self + b_self
        other_sum = a_other + b_other
        return torch.logical_or(
            torch.lt(self_sum, other_sum),
            torch.logical_and(
                torch.isclose(self_sum, other_sum, rtol=self.__zero_threshold__),
                torch.gt(a_self, a_other)
            )
        )

    def lt(self, other):
        """
        元素级小于比较（支持广播）
        """
        a_self, b_self, a_other, b_other = self._broadcast_compare(other)
        self_sum = a_self + b_self
        other_sum = a_other + b_other
        return torch.logical_or(
            torch.gt(self_sum, other_sum),
            torch.logical_and(
                torch.isclose(self_sum, other_sum, rtol=self.__zero_threshold__),
                torch.lt(a_self, a_other)
            )
        )

    def le(self, other):
        """元素级小于等于比较（支持广播）"""
        return torch.logical_not(self.gt(other))

    def ge(self, other):
        """元素级大于等于比较（支持广播）"""
        return torch.logical_not(self.lt(other))

    def __pow__(self, power):
        """
        OPN幂运算 (a,b)**n

        规则:
        - n=0: 返回单位元 (0,-1)
        - n=1: 返回自身
        - n>1: 递归计算
        - n<0: 计算倒数后正幂
        - 支持广播到每个元素

        参数:
            power: 标量或与self.shape匹配的张量

        返回:
            OPNTensorMatrix: 计算结果
        """
        if isinstance(power, (int, float, torch.Tensor, np.number)):
            power = torch.as_tensor(power, device=self.device)

        if not isinstance(power, torch.Tensor):
            raise TypeError(f"不支持的类型 {type(power)}，应为标量或torch.Tensor")

        # 处理标量情况
        if power.ndim == 0:
            if power == 0:
                return OPNTensorMatrix.one(self.shape, device=self.device)
            if power == 1:
                return self
            if power < 0:
                return self.__pow__(-power).inverse()

            # 正整数的幂运算
            result = self
            for _ in range(int(power.item()) - 1):
                result = result * self
            return result

        # 处理张量幂的情况（逐元素）
        a, b = self.data[..., 0], self.data[..., 1]
        shape = self.shape

        # 广播power到相同形状
        if power.shape != shape:
            try:
                power = torch.broadcast_to(power, shape)
            except RuntimeError:
                raise ValueError(f"无法广播power形状 {power.shape} 到 {shape}")

        # 初始化结果
        result_data = torch.empty_like(self.data)

        # 处理不同幂值
        zero_mask = (power == 0)
        one_mask = (power == 1)
        pos_mask = (power > 1) & ~zero_mask & ~one_mask
        neg_mask = (power < 0)

        # 应用不同规则
        result_data[zero_mask] = torch.tensor([0, -1], device=self.device)  # 单位元
        result_data[one_mask] = self.data[one_mask]

        # 正幂计算
        if pos_mask.any():
            a_pos = a[pos_mask]
            b_pos = b[pos_mask]
            n_pos = power[pos_mask].unsqueeze(-1)  # 保持维度

            # 使用OPN幂公式
            a_plus_b = a_pos + b_pos
            a_minus_b = a_pos - b_pos
            pow_plus = a_plus_b ** n_pos
            pow_minus = a_minus_b ** n_pos

            head = ((-1) ** (n_pos + 1) / 2) * pow_plus
            tail = 0.5 * pow_minus

            result_data[pos_mask, 0] = head + tail
            result_data[pos_mask, 1] = head - tail

        # 负幂计算
        if neg_mask.any():
            inv = self[neg_mask].inverse()
            n_neg = -power[neg_mask].unsqueeze(-1)
            result_data[neg_mask] = (inv ** n_neg).data

        return OPNTensorMatrix(result_data, device=self.device)

    def inverse(self):
        """
        计算OPN矩阵的逆 (逐元素)
        1/(a,b) = (a/(a²-b²), -b/(a²-b²))
        """
        a, b = self.data[..., 0], self.data[..., 1]
        denominator = a ** 2 - b ** 2

        if torch.any(torch.abs(denominator) < self.__zero_threshold__):
            raise ZeroDivisionError("存在零因子 (a²≈b²)")

        inv_a = a / denominator
        inv_b = -b / denominator
        return OPNTensorMatrix(torch.stack([inv_a, inv_b], dim=-1), device=self.device)

    def broadcast_add(self, other):
        """
        广播加法
        支持与标量或形状兼容的张量相加

        参数:
            other: 标量、OPNTensor或OPNTensorMatrix

        返回:
            OPNTensorMatrix: 广播相加结果
        """
        if isinstance(other, (int, float)):
            # 标量加法视为加到a分量
            new_data = self.data
            new_data[..., 0] += other
            return OPNTensorMatrix(new_data, device=self.device)

        if isinstance(other, OPNTensor):
            # 单个OPNTensor广播到所有元素
            other_data = torch.broadcast_to(other.data, self.data.shape)
            return OPNTensorMatrix(self.data + other_data, device=self.device)

        if isinstance(other, OPNTensorMatrix):
            # 标准广播加法
            try:
                return OPNTensorMatrix(self.data + other.data, device=self.device)
            except RuntimeError as e:
                raise ValueError(f"形状不匹配无法广播: {self.shape} vs {other.shape}") from e

        raise TypeError(f"不支持的类型 {type(other)}")

    def broadcast_mul(self, other):
        """
        广播乘法
        支持与标量或形状兼容的张量相乘

        参数:
            other: 标量、OPNTensor或OPNTensorMatrix

        返回:
            OPNTensorMatrix: 广播相乘结果
        """
        if isinstance(other, (int, float, torch.Tensor, np.number)):
            # 标量乘法
            return OPNTensorMatrix(self.data * other, device=self.device)

        if isinstance(other, OPNTensor):
            # 单个OPNTensor广播乘法
            a_other, b_other = other.a.item(), other.b.item()
            a_self, b_self = self.data[..., 0], self.data[..., 1]

            new_a = -a_self * b_other - b_self * a_other
            new_b = -a_self * a_other - b_self * b_other
            return OPNTensorMatrix(torch.stack([new_a, new_b], dim=-1), device=self.device)

        if isinstance(other, OPNTensorMatrix):
            # 标准OPN乘法广播
            a_self, b_self = self.data[..., 0], self.data[..., 1]
            a_other, b_other = other.data[..., 0], other.data[..., 1]

            # 广播形状
            try:
                a_self, b_self, a_other, b_other = torch.broadcast_tensors(
                    a_self, b_self, a_other, b_other)
            except RuntimeError as e:
                raise ValueError(f"形状不匹配无法广播: {self.shape} vs {other.shape}") from e

            new_a = -a_self * b_other - b_self * a_other
            new_b = -a_self * a_other - b_self * b_other
            return OPNTensorMatrix(torch.stack([new_a, new_b], dim=-1), device=self.device)

        raise TypeError(f"不支持的类型 {type(other)}")

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __abs__(self):
        return self * torch.where(self < OPNTensorMatrix.zero(1,device=self.device), -1, 1)

    def abs(self):
        return self * torch.where(self < OPNTensorMatrix.zero(1,device=self.device), -1, 1)

    def dot(self, other):
        """矩阵乘法或标量乘法"""
        if isinstance(other, (int, float, torch.Tensor, np.number)):
            # 标量乘法
            return OPNTensorMatrix(self.data * other, device=self.device)
        elif isinstance(other, OPNTensorMatrix):
            # OPN乘法: (a,b)*(c,d) = (-ad-bc, -ac-bd)
            a, b = self.data[..., 0], self.data[..., 1]
            c, d = other.data[..., 0], other.data[..., 1]

            # 如果需要则广播形状
            if a.shape != c.shape:
                a, b, c, d = torch.broadcast_tensors(a, b, c, d)

            new_a = -a * d - b * c
            new_b = -a * c - b * d
            result = torch.stack([new_a, new_b], dim=-1)
            return OPNTensorMatrix(result, device=self.device)
        else:
            raise TypeError(f"无法将OPNTensorMatrix与 {type(other)} 相乘")
