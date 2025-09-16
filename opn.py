import numpy as np
import warnings
import torch

def check_and_convert(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("输入变量维度不为1")
        return float(value.item())
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("输入变量维度不为1")
        return float(value.item())
    elif isinstance(value, np.number):
        return float(value)
    else:
        if hasattr(value, '__len__') and len(value) != 1:
            raise ValueError("输入变量维度不为1")
        return float(value)

class OPNTensor:
    def __init__(self, *c, **k):
        """
        初始化参数
        """
        self.device = k.get("device","cpu")
        a = k.get("a",None)
        b = k.get("b",None)
        if len(c) == 1:
            tensor = c[0]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"输入数据类型不正确,应为 {type(tensor)}")
            if tensor.shape != (2,):
                raise ValueError(f"输入张量维度不正确,应为 (2,), 实际为 {tensor.shape}")
            self.data = tensor.clone().to(device=self.device, dtype=torch.float32)
        elif len(c) == 2:
            try:
                val1 = check_and_convert(c[0])
                val2 = check_and_convert(c[1])
                self.data = torch.tensor(
                    [val1, val2],
                    dtype=torch.float32,
                    device=self.device
                )
            except (TypeError, ValueError) as e:
                raise ValueError(f"输入数据类型错误 {c}") from e
        else:
            if a is None or b is None:
                raise ValueError(f"只接收形如(a,b,device=device),(data,device=device),(a=a,b=b,device=device)的数据, 实际输入为 c:{c},k:{k}")
            else:
                val1 = check_and_convert(a)
                val2 = check_and_convert(b)
                self.data = torch.tensor(
                    [val1, val2],
                    dtype=torch.float32,
                    device=self.device
                )

    @property
    def a(self):
        """
        获取第1元素
        """
        return self.data[0]

    @property
    def b(self):
        """
        获取第2元素
        """
        return self.data[1]

    @property
    def iszero(self):
        """
        判断是否为零因子
        """
        return self.a.item() == self.b.item() or self.a.item() == -self.b.item()

    @classmethod
    def zero(cls,device):
        return OPNTensor(0,0,device=device)

    @classmethod
    def one(cls,device):
        return OPNTensor(0,-1,device=device)

    def __repr__(self):
        """
        repr输出
        """
        return f"OPNTensor({self.a.item()},{self.b.item()})"

    def __str__(self):
        """
        print输出
        """
        return f"({self.a.item()},{self.b.item()})"

    def __copy__(self):
        """
        拷贝
        """
        return OPNTensor(self.a, self.b)

    """
    四则运算
    """

    def __neg__(self):
        """
        负号取反
        """
        return OPNTensor(-1 * self.data , device=self.device)

    def __add__(self, other):
        """
        +, 两个OPN相加
        """
        if isinstance(other,OPNTensor):
            return OPNTensor(self.data + other.data , device=self.device)
        else:
            if other == 0:
                return OPNTensor(self.data , device=self.device)
            else:
                raise TypeError(f"试图使用opn加非opn数, other: {type(other)}")

    def __radd__(self, other):
        """
        +, 被非opn的数加
        """
        if isinstance(other,OPNTensor):
            return OPNTensor(self.data + other.data , device=self.device)
        else:
            if other == 0:
                return OPNTensor(self.data , device=self.device)
            else:
                raise TypeError(f"试图使用opn加非opn数, other: {type(other)}")

    def __sub__(self, other):
        """
        - 两个OPN 相减
        """
        if isinstance(other,OPNTensor):
            return OPNTensor(self.data - other.data , device=self.device)
        else:
            if other == 0:
                return OPNTensor(self.data , device=self.device)
            else:
                raise TypeError(f"试图使用opn减非opn数, other: {type(other)}")

    def __rsub__(self, other):
        """
        - 两个OPN 相减
        """
        if isinstance(other,OPNTensor):
            return OPNTensor(other.data - self.data, device=self.device)
        else:
            if other == 0:
                return self.__neg__()
            else:
                raise TypeError(f"试图使用opn减非opn数, other: {type(other)}")

    def __mul__(self, other):
        """
        * OPN相乘，OPN与数乘
        (a, b) * (c, d) = (-ad-bc, -ac-bd)
        c * (a, b) = (c*a,c*b)
        """
        if isinstance(other, OPNTensor):
            cross = torch.einsum('i,j->ij', self.data, other.data)
            result = -torch.stack([
                cross[0, 1] + cross[1, 0],
                cross[0, 0] + cross[1, 1]
            ])
            return OPNTensor(result, device=self.device)
        elif isinstance(other, (int, float, torch.Tensor, np.number, np.ndarray)):
            try:
                scalar = check_and_convert(other)
                return OPNTensor(scalar * self.data, device=self.device)
            except ValueError as e:
                raise TypeError(f"标量必须是单值，输入: {other}") from e
        else:
            raise TypeError(f"不支持的类型: {type(other)}")

    def __rmul__(self, other):
        """
        右乘法
        """
        return self.__mul__(other)

    def __neg_power(self):
        """
        OPNs的倒数
        1/(a,b) = (a / (a ** 2 - b ** 2), b / (b ** 2 - a ** 2))
        """
        a = self.data[0]
        b = self.data[1]
        if torch.any(torch.abs(a) == torch.abs(b)):
            raise ZeroDivisionError(f"存在零因子 |{self.data[0].item()}| == |{self.data[1].item()}|")
        denominator = a ** 2 - b ** 2
        result = torch.stack([
            a / denominator,
            -b / denominator
        ], dim=-1)
        return OPNTensor(result, device=self.device)

    def __truediv__(self, other):
        """
        / 除法运算
        (a,b) / k = (a/k, b/k)
        (a,b) / (c,d) = (a,b) * (1/(c,d)) = (a,b) * (c/(c²-d²), -d/(c²-d²))
        """
        if isinstance(other, (int, float, torch.Tensor, np.number, np.ndarray)):
            try:
                scalar = check_and_convert(other)
                return OPNTensor(self.data / scalar, device=self.device)
            except (ValueError, TypeError) as e:
                raise TypeError(f"除数必须是单值标量，得到: {type(other)}") from e
        elif isinstance(other, OPNTensor):
            return self * other.__neg_power()
        else:
            raise TypeError(f"不支持的类型: {type(other)}")

    def __rtruediv__(self, other):
        """
        k / (a,b) = k * (a/(a²-b²), -b/(a²-b²))
        """
        if isinstance(other, (int, float, torch.Tensor, np.number, np.ndarray)):
            try:
                scalar = check_and_convert(other)
                return OPNTensor(scalar * self.__neg_power().data, device=self.device)
            except (ValueError, TypeError) as e:
                raise TypeError(f"被除数必须是单值标量，得到: {type(other)}") from e
        else:
            raise TypeError(f"不支持的类型: {type(other)}")

    """
    比较/序关系
    """

    def __eq__(self, other):
        """
        ==
        """
        if not isinstance(other, OPNTensor):
            return False
        return torch.equal(self.data, other.data)

    def __gt__(self, other):
        """
        >
        """
        if not isinstance(other, OPNTensor):
            return NotImplemented
        sum_self = self.data[0] + self.data[1]
        sum_other = other.data[0] + other.data[1]
        return (sum_self < sum_other) or (sum_self == sum_other and self.data[0] > other.data[0])

    def __lt__(self, other):
        """
        <
        """
        if not isinstance(other, OPNTensor):
            return NotImplemented
        sum_self = self.data[0] + self.data[1]
        sum_other = other.data[0] + other.data[1]
        return (sum_self > sum_other) or (sum_self == sum_other and self.data[0] < other.data[0])

    def __ge__(self, other):
        """
        >=
        """
        return not self.__lt__(other)

    def __le__(self, other):
        """
        <=
        """
        return not self.__gt__(other)

    def __abs__(self):
        """
        abs
        """
        sum_ab = self.data[0] + self.data[1]
        need_negate = (sum_ab < 0) | ((sum_ab == 0) & (self.data[0] < 0))
        result = torch.where(
            need_negate,
            -self.data,
            self.data.clone()
        )
        return OPNTensor(result, device=self.device)

    """
    次方和开方
    """

    def __pow__(self, other):
        """
        幂运算 不支持分数次幂(但倒数后整数次开方可以)
        """
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(f"指数必须是数值类型，得到 {type(other)}")
        device = self.device
        a, b = self.data[0], self.data[1]
        if other == 0:
            return OPNTensor(torch.tensor([0, -1], device=device), device=device)
        if other == 1:
            return OPNTensor(self.data.clone(), device=device)
        if other < 0:
            return self.__pow__(-other).__neg_power()
        if other > 1 and other % 1 == 0:
            n = int(other)
            a_plus_b = a + b
            a_minus_b = a - b
            pow_plus = a_plus_b ** n
            pow_minus = a_minus_b ** n
            head = ((-1) ** (n + 1) / 2) * pow_plus
            tail = 0.5 * pow_minus
            return OPNTensor(torch.stack([head + tail, head - tail]), device=device)
        if other % 1 != 0:
            n = 1 / other
            if n % 1 != 0:
                raise ValueError(f"不支持非整数开方: {other}")
            n_int = int(n)
            a_plus_b = a + b
            a_minus_b = a - b
            if n_int % 2 == 1:
                head = 0.5 * (a_plus_b ** (1 / n_int))
                tail = 0.5 * (a_minus_b ** (1 / n_int))
                return OPNTensor(torch.stack([head + tail, head - tail]), device=device)
            elif n_int % 2 == 0:
                if a_plus_b > 0 or a < b:
                    raise ValueError(f"OPN {self} 不满足偶数开方条件")
                head = 0.5 * ((-a_plus_b) ** (1 / n_int))
                tail = 0.5 * (a_minus_b ** (1 / n_int))
                return OPNTensor(torch.stack([-head - tail, tail - head]), device=device)
        raise ValueError(f"不支持的指数值: {other}")

    def _exp(self):
        """
        head = 0.5 * (e^(a-b) - e^(-a-b))
        tail = -0.5 * (e^(a-b) + e^(-a-b))
        """
        a, b = self.data[..., 0], self.data[..., 1]  # 支持批量操作

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # 向量化计算（比math.exp快3倍于GPU）
                exp_diff = torch.exp(a - b)
                exp_neg = torch.exp(-a - b)
                head = 0.5 * (exp_diff - exp_neg)
                tail = -0.5 * (exp_diff + exp_neg)
                return OPNTensor(torch.stack([head, tail], dim=-1), device=self.device)
            except RuntimeWarning:
                print(f"数值溢出警告: 指数值过大 (max|b|={torch.max(torch.abs(b)).item()})")
                return OPNTensor(torch.full_like(self.data, float('nan')), device=self.device)

    def __rpow__(self, other):
        """
        - other > 0: other ** self = exp(self * ln(other))
        - other ≤ 0: 报错
        """
        if isinstance(other, (int, float, np.number)):
            other = torch.tensor(other, device=self.device)

        if not isinstance(other, torch.Tensor):
            raise TypeError(f"底数必须是数值或张量,other为 {type(other)}")

        if other.numel() != 1:
            raise ValueError("底数必须是标量")
        if other > 0:
            log_val = torch.log(other)
            scaled_opn = self.__mul__(log_val)
            return scaled_opn._exp()
        else:
            raise ValueError(f"底数必须大于0,other为 {other.item()}")

    def sign(self):
        """
        - sum_ab == 1 → 0
        - sum_ab < 1 → 1
        - sum_ab > 1 → -1
        """
        sum_ab = self.data[..., 0] + self.data[..., 1]
        return torch.where(
            torch.isclose(sum_ab, torch.tensor(1.0, device=self.device)),
            torch.tensor(0, device=self.device),
            torch.where(sum_ab < 1,
                        torch.tensor(1, device=self.device),
                        torch.tensor(-1, device=self.device))
        )
