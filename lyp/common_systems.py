"""
    Useful systems definitions
    ==========================

    functions to define common dynamical systems.
"""

from system import DynamicalSystem
from sympy import symbols

def lorenz(parms=[10., 28., 8/3]):
    x, y, z = symbols('x, y, z')
    sigma, rho, beta = symbols('sigma rho beta')

    variables = [x, y, z]
    
    parameters = [sigma, rho, beta]

    lorenz = [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

    system = DynamicalSystem(variables, parameters, lorenz)
    system.base_parameters(parms)
    
    return system

def rossler(parms=[0.2, 0.2, 5.7]):
    x, y, z = symbols('x, y, z')
    a, b, c = symbols('a b c')

    variables = [x, y, z]
    
    parameters = [a, b, c]

    lorenz = [
        - y - z,
        x + a * y,
        b + z * (x - c)
    ]

    system = DynamicalSystem(variables, parameters, lorenz)
    system.base_parameters(parms)
    
    return system
 

if __name__ == "__main__":
    import numpy as np
    lorenz = lorenz()

    y0 = np.array([1, 2, 3])
    print(lorenz.convert_location(y0))

    print(lorenz.jacobian.sub_parameters(matrix_fmt=True))
    print(lorenz.funcs())
