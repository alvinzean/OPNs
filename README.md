# OPNs Library

## Overview

The OPNs Library is a Python package that defines a special numerical domain called OPNs (Ordered pair of normalized real numbers). This library implements custom arithmetic operations, including addition, subtraction, multiplication, division, exponentiation, root extraction, and trigonometric functions, all following the unique rules of the OPNs domain.

## Features

- Custom OPNs arithmetic operations
- OPNs class
- OPNs-specific math functions

## Installation

To install the OPNs Library, clone this repository:

```bash
git clone https://github.com/alvinzean/OPNs.git
cd OPNs
```

## Usage

- ### OPNs Class(opns.py)
The **OPNs** class includes the definition of OPNs and their basic operations. It overrides the standard arithmetic operators to support custom rules.

#### Basic OPNs Operations

```python
from opns import OPNs

a = OPNs(3, 4)
b = OPNs(-3, -4)

print(a + b)  # OPNs addition
print(a - b)  # OPNs subtraction
print(a * b)  # OPNs multiplication
print(a / b)  # OPNs division
print(a ** 2) # OPNs exponentiation
print(a == b) # OPNs comparison
print(a < b)  # OPNs less than
```

- ### OPNs Math Functions(opns_math.py)
The **opns_math** module provides various mathematical functions for OPNs, similar to Python's math library. It includes functions for logarithms, trigonometric functions, and more (will be continuously updated).

#### Example Usage

```python
from opns import OPNs
import opns_math

a = OPNs(3, 4)

print(opns_math.log(a))     # OPNs logarithm
print(opns_math.sin(a))     # OPNs sine
print(opns_math.asin(a))    # OPNs arcsine
print(opns_math.exp(a))     # OPNs exponential
```

## OPNs Mathematical Formulas

An OPNs is defined as $\alpha=(\mu_{\alpha}, \nu_{\alpha})$, with both $\mu_{\alpha}$ and $\nu_{\alpha}$ in the interval (0,1). In the actual operation, we removed the restriction of two terms in OPNs with values between 0 and 1. Here are some examples of mathematical formulas for OPNs:

- ### Addition

Given two OPNs, $\alpha=(\mu_{\alpha}, \nu_{\alpha})$ and $\beta=(\mu_{\beta}, \nu_{\beta})$, their addition is defined as:

$$\alpha+\beta=(\mu_{\alpha}+\mu_{\beta}, \nu_{\alpha}+\nu_{\beta})$$

- ### Multiplication

The multiplication of two OPNs is defined as:
$$\alpha\cdot\beta = (-\mu_{\alpha}\nu_{\beta}-\nu_{\alpha}\mu_{\beta},-\mu_{\alpha}\mu_{\beta}-\nu_{\alpha}\nu_{\beta})$$

- ### Exponentiation

The exponentiation of an OPNs $\alpha$ raised to the power of $n$ is defined as:
$$\alpha^{n} = \left (\frac{(-1)^{n+1}}{2} \left ( \mu_{\alpha} + \nu_{\alpha} \right )^n + \frac{1}{2}\left ( \mu_{\alpha} - \nu_{\alpha} \right )^n, \frac{(-1)^{n+1}}{2} \left ( \mu_{\alpha} + \nu_{\alpha} \right )^n - \frac{1}{2}\left ( \mu_{\alpha} - \nu_{\alpha} \right )^n  \right )$$

## Reference and Background

For detailed definitions and calculation rules of Ordered Pair of Normalized Real Numbers (OPNs), please refer to the foundational paper:
﻿
Zhou, Lei. *"Ordered pair of normalized real numbers."* Information Sciences 538 (2020): 290–313.
[https://doi.org/10.1016/j.ins.2020.05.036](https://doi.org/10.1016/j.ins.2020.05.036)
﻿
## Recent Updates

We have updated this repository to include the core implementation of the OPNs matrix class—`opns_numpy`—which provides Numpy-like functionality tailored for OPNs data structures and arithmetic. This module enables convenient construction, manipulation, and computation of OPNs matrices, and is designed to support further development in machine learning and numerical applications under the OPNs framework.
﻿
## Future Updates

We are actively maintaining and improving the `opns_numpy` library. Future plans include:
﻿
* Enhanced matrix operations and broadcasting support
* automatic optimization of model parameters
* Optimized performance for large-scale OPNs computations


More updates will be provided in the future, including additional OPNs algorithm models, and even a parallel version that can run on GPUs.

## How to use it

### Install dependency

Run `pip install  -r requirement.txt` to install the main libraries we needed.

### Use test dataset

We provide several datasets, including energy, wine, yacht, bike, etc.

Each of them can be used like: `python test.py --dataset [dataset_name]`. 

For example, we run `bike` dataset like this: `python test.py --dataset bike`.

Each dataset needed a corresponding parameters configuration. You can find it at `utils/test_data_params.json`.

### Use personal dataset

**You can use your custom dataset**. 

1. Add it to our dataset folder, which is named **dataset**. 
2. Then, add the parameters about your dataset to a `json` file like `test_data_params.json`. If you don’t provide parameters, we will use a default configuration.
3. Add your dataset_name to`parser.add_argument('--dataset', ...)` in  `test.py` .
4. run `python test.py --dataset [YOUR dataset_name]`
5. You can use additional parameter `--memory` to analyze memory usage.

👍 You can follow our Github for the latest updates on this project. [OPNs](https://github.com/alvinzean/OPNs)

