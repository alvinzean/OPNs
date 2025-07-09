import math
from opns_pack.opns import OPNs


def pow(opn, n):
    """
    Power
    """
    return opn ** n


def sqrt(opn):
    """
    Square root
    """
    return opn ** (1 / 2)


def exp(opn):
    return opn._exp()


def ln(opn):
    if opn.a + opn.b < 0 and opn.a > opn.b:
        head = 0.5 * math.log((opn.b - opn.a) / (opn.b + opn.a))
        tail = -0.5 * math.log(opn.b ** 2 - opn.a ** 2)
        return OPNs(head, tail)
    else:
        raise ValueError('Error: For ln() operation, OPN{} should be positive and '
                         'satisfying the first term is greater than the second term!'.format(opn))


def log(opn, base=math.e):
    if base > 0 and base != 1 and ln(opn):
        return ln(opn) * (math.log(base) ** (-1))
    else:
        raise ValueError('Error: The operation log{} for this OPNs does not exist; '
                         'or the base must be greater than 0 and not equal to 1.'.format(opn))


def log2(opn):
    return log(opn, 2)


def log10(opn):
    return log(opn, 10)


def sin(opn):
    head = math.sin(opn.a) * math.cos(opn.b)
    tail = math.cos(opn.a) * math.sin(opn.b)
    return OPNs(head, tail)


def cos(opn):
    head = math.sin(opn.a) * math.sin(opn.b)
    tail = - math.cos(opn.a) * math.cos(opn.b)
    return OPNs(head, tail)


def tan(opn):
    head1 = math.tan(opn.a) * (1 + math.tan(opn.b) ** 2)
    head2 = math.tan(opn.b) * (1 + math.tan(opn.a) ** 2)
    tail = 1 - (math.tan(opn.a) ** 2) * (math.tan(opn.b) ** 2)
    first_entry = head1 / tail
    second_entry = head2 / tail
    return OPNs(first_entry, second_entry)


def cot(opn):
    head1 = math.tan(opn.a) * (1 + math.tan(opn.b) ** 2)
    head2 = math.tan(opn.b) * (1 + math.tan(opn.a) ** 2)
    tail1 = math.tan(opn.a) ** 2 - math.tan(opn.b) ** 2
    tail2 = math.tan(opn.b) ** 2 - math.tan(opn.a) ** 2
    first_entry = head1 / tail1
    second_entry = head2 / tail2
    return OPNs(first_entry, second_entry)


def asin(opn):
    if -1 <= opn.a + opn.b <= 1 and -1 <= opn.a - opn.b <= 1:
        first_entry = 0.5 * (math.asin(opn.a + opn.b) + math.asin(opn.a - opn.b))
        second_entry = 0.5 * (math.asin(opn.a + opn.b) - math.asin(opn.a - opn.b))
        return OPNs(first_entry, second_entry)
    else:
        raise ValueError(
            'Error: For asin() operation, OPN{} should satisfy -1 ≤ ux + vx ≤ 1 and -1 ≤ ux - vx ≤ 1'.format(opn))


def acos(opn):
    if -1 <= opn.a + opn.b <= 1 and -1 <= opn.a - opn.b <= 1:
        first_entry = 0.5 * (math.acos(opn.a + opn.b) + math.acos(opn.a - opn.b)) + math.pi / 2
        second_entry = 0.5 * (math.acos(opn.a + opn.b) - math.acos(opn.a - opn.b)) + math.pi / 2
        return OPNs(first_entry, second_entry)
    else:
        raise ValueError(
            'Error: For acos() operation, OPN{} should satisfy -1 ≤ ux + vx ≤ 1 and -1 ≤ ux - vx ≤ 1'.format(opn))


def atan(opn):
    x, y = opn.a, opn.b
    head = (x ** 2 - y ** 2 - 1 + math.sqrt((x ** 2 - y ** 2 - 1) ** 2 + 4 * x ** 2)) / (2 * x)
    tail = (y ** 2 - x ** 2 - 1 + math.sqrt((y ** 2 - x ** 2 - 1) ** 2 + 4 * y ** 2)) / (2 * y)
    return OPNs(head, tail)
