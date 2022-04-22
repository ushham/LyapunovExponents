from numba import njit
import numpy as np
from sympy import symbols, lambdify


def apply_func(func, loc):
    ns = np.empty(len(func))
    for n, f in enumerate(func):
        ns[n] = f(*loc)
    return ns


def rungekutta4(func, y0, h):
    """
        RK4 method for single 

    Parameters
    ----------
    f: DynamicalSystem object

    y0: numpy array

    h: float

    """

    fk1 = apply_func(func, y0)
    fk2 = apply_func(func, y0 + fk1 * h / 2.)
    fk3 = apply_func(func, y0 + fk2 * h / 2.)
    fk4 = apply_func(func, y0 + fk3 * h)

    y = y0 + (h / 6.) * (fk1 + 2*fk2 + 2*fk3 + fk4)
    
    return y

# Function not yet working
def rungekutta4_coupled(func1, func2, y0, z0, h):
    #RK4 numerical solver for a coupled system {x' = f(x), P' = J(x)P}
    fk1 = apply_func(func1, y0)
    gk1 = g(z0, y0)

    fk2 = f(y0 + fk1 * h / 2.)
    gk2 = g(z0 + gk1 * h / 2., y0 + fk1 * h / 2.)

    fk3 = f(y0 + fk2 * h / 2.)
    gk3 = g(z0 + gk2 * h / 2., y0 + fk2 * h / 2.)

    fk4 = f(y0 + fk3 * h)
    gk4 = g(z0 + gk3 * h, y0 + fk3 * h)


    y = y0 + (h / 6.) * (fk1 + 2*fk2 + 2*fk3 + fk4)
    z = z0 + (h / 6.) * (gk1 + 2*gk2 + 2*gk3 + gk4)
    
    return y, z


if __name__ == "__main__":
    from common_systems import lorenz
    system = lorenz()

    y0 = np.array([1, 2, 3])
    res = np.empty((100, 3))

    res[0, :] = y0

    functions = system.num_funcs()
    jacobian = system.jacobian()
    jac_func = jacobian.num_funcs()



"""
next steps

make coupled rk4
make jacobian a function
start working on the class to calculate maximal element 

"""