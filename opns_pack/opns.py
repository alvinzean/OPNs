import math
import warnings

"""
Encapsulate OPNs as a class
"""

def tran(x, m):
    if x < 0:
        return -math.pow(-x, m)
    else:
        return math.pow(x, m)


class OPNs:
    """
    OPNs class
    """
    absolute_zero = (0, 0)

    @property
    def zero(self):
        """
        Used to check if it can be used as a denominator
        """
        return self.a == self.b or self.a == -self.b

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        """
        Print out OPNs
        """
        return f"({self.a}, {self.b})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        """
        Create a new copy
        Used when the result should remain unchanged.
        If not copied, modifying the result would change the original.
        """
        return OPNs(self.a, self.b)

    """
    Arithmetic operations
    """

    def __neg__(self):
        """
        Unary negation
        -OPNs
        """
        return OPNs(-self.a, -self.b)

    def __add__(self, other):
        """
        +, Add two OPNs
        """
        if hasattr(other, '__radd__') and not isinstance(other, OPNs):
            return other.__radd__(self)
        return OPNs(self.a + other.a, self.b + other.b)

    def __radd__(self, other):
        """
        Right-hand addition
        Used when built-in functions like sum() call 0 + OPNs
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        """
        -, Subtract two OPNs
        """
        if hasattr(other, '__rsub__'):
            return other.__rsub__(self)
        return OPNs(self.a - other.a, self.b - other.b)

    def __mul__(self, other):
        """
        *, Multiplication of OPNs or scalar multiplication
        (a, b) * (c, d) = (-ad - bc, -ac - bd)
        """
        if isinstance(other, (int, float)):
            return OPNs(self.a * other, self.b * other)
        elif isinstance(other, OPNs):
            e = -self.a * other.b - self.b * other.a
            f = -self.a * other.a - self.b * other.b
            return OPNs(e, f)
        else:
            return other.__rmul__(self)

    def __rmul__(self, other):
        """
        Right-hand multiplication
        """
        if other == 1:
            return self
        return self.__mul__(other)

    def __neg_power(self):
        """
        Multiplicative inverse of an OPNs
        """
        if self.a == self.b or self.a == -self.b:
            raise ZeroDivisionError(f'The multiplicative inverse of this OPNs {self} does not exist')
        else:
            c = self.a / (self.a ** 2 - self.b ** 2)
            d = self.b / (self.b ** 2 - self.a ** 2)
            return OPNs(c, d)

    def __truediv__(self, other):
        """
        /, Division
        opns1 / opns2: equals opns1 * (inverse of opns2)
        """
        if isinstance(other, (int, float)):
            return OPNs(self.a / other, self.b / other)
        return self.__mul__(other.__neg_power())

    def __rtruediv__(self, other):
        """
        Right-hand division
        real / OPNs: equals real * (inverse of OPNs)
        """
        return self.__neg_power().__mul__(other)

    """
    Comparison / Ordering
    """

    def __eq__(self, other):
        """
        ==
        """
        return self.a == other.a and self.b == other.b

    def __gt__(self, other):
        """
        >, Greater than
        """
        sub_opns = self.__sub__(other)
        return (sub_opns.a + sub_opns.b < 0) or (sub_opns.a + sub_opns.b == 0 and sub_opns.a > 0)

    def __lt__(self, other):
        """
        <, Less than
        """
        sub_opns = self.__sub__(other)
        return (sub_opns.a + sub_opns.b) > 0 or (sub_opns.a + sub_opns.b == 0 and sub_opns.a < 0)

    def __gl__(self, other):
        """
        >=, Greater than or equal
        """
        return self.__gt__(other) or self.__eq__(other)

    def __le__(self, other):
        """
        <=, Less than or equal
        """
        return self.__lt__(other) or self.__eq__(other)

    def __abs__(self):
        if self.a + self.b > 0 or (self.a + self.b == 0 and self.a < 0):
            return OPNs(-self.a, -self.b)
        return self.__copy__()

    '''2. Power and nth root of OPNs'''

    """
    Power and root operations
    """

    def __pow__(self, other):
        """
        Power operation (fractional powers not supported)
        (But inverse followed by root with integer power is allowed)
        """
        if other == 0:  # exponent = 0
            return OPNs(0, -1)
        if other == 1:  # exponent = 1
            return self.__copy__()
        if other > 1:  # exponent > 0
            head = (((-1) ** (other + 1)) / 2) * ((self.a + self.b) ** other)
            tail = 0.5 * ((self.a - self.b) ** other)
            c = head + tail
            d = head - tail
            return OPNs(c, d)
        # Root operation
        if other < 0:  # negative exponent = inverse of positive power
            return self.__pow__(-other).__neg_power()
        if other % 1 != 0:
            n = 1 / other
            if n % 1 != 0:
                raise ValueError("Invalid root '{}': root() only supports integer roots.".format(n))
            elif n % 2 == 1:
                head = 0.5 * tran(self.a + self.b, 1 / n)
                tail = 0.5 * tran(self.a - self.b, 1 / n)
                first_entry = head + tail
                second_entry = head - tail
                new_opns = OPNs(first_entry, second_entry)
                return new_opns
            elif n % 2 == 0 and self.a + self.b <= 0 and self.a >= self.b:
                head = 0.5 * ((-self.a - self.b) ** (1 / n))
                tail = 0.5 * ((self.a - self.b) ** (1 / n))
                first_entry = head + tail
                second_entry = head - tail
                new_opns = OPNs(-first_entry, -second_entry)
                return new_opns
            else:
                raise Exception(
                    "Error: When n is even, if OPNs is negative, or the first term of OPNs is smaller than the second "
                    "term, "
                    "the OPNs {} cannot open roots!".format(self))

    def _exp(self):
        # Use catch_warnings to manage the context of warnings
        with warnings.catch_warnings():
            # Use filterwarnings to control behavior
            warnings.filterwarnings('error')  # Turn warnings into exceptions

            try:
                head = 0.5 * (math.e ** (self.a - self.b) - math.e ** (-self.a - self.b))
                tail = -0.5 * (math.e ** (self.a - self.b) + math.e ** (-self.a - self.b))
                return OPNs(head, tail)
            except RuntimeWarning:
                # Handle RuntimeWarning here
                print(f"Runtime warning: exponent too large. Current value: {self.b}")


    def __rpow__(self, other):
        """
        Real number raised to OPNs power
        """
        if other > 0:
            return (self.__mul__(math.log(other)))._exp()
        else:
            raise ValueError('Error: the real number \'{}\' should be greater than 0'.format(other))

