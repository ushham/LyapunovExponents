"""
    Useful systems definitions
    ==========================

    functions to define common dynamical systems.
"""

from system import DynamicalSystem
from sympy import symbols

def lorenz(parms=[10., 28., 8/3]):
    system = DynamicalSystem()
    x, y, z = symbols('x, y, z')
    sigma, rho, beta = symbols('sigma rho beta')

    variables = [x, y, z]
    
    parameters = [sigma, rho, beta]

    lorenz = [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

    system.model_definition(variables, parameters, lorenz)
    system.base_parameters(parms)
    
    return system
 

if __name__ == "__main__":
    import numpy as np
    lorenz = lorenz()
    # print(lorenz.subs_functions(('x', 1)))
    y0 = np.array([1, 2, 3])
    print(lorenz.convert_location(y0))

    print(lorenz.jacobian.sub_parameters(matrix_fmt=True))
    print(lorenz.jacobian.parm_vals)
