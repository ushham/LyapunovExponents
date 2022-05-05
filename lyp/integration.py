from numba import njit
import numpy as np
from sympy import symbols, lambdify


def apply_func(func, loc):
    ns = np.empty(len(func))
    for n, f in enumerate(func):
        ns[n] = f(*loc)
    return ns

def apply_matrix_func(funcs, basis, location, inverted=False):
        """
            Function to calculate the system P' = J(location) P.
            This is done by calculating how the Jacobian (at the gievn location)
            alters the given basis directions.
            This function is designed to provide the tendencies to then be used by an integrator to calculate the value of P.
        """
        jac = apply_func(func=funcs, loc=location)
        jac = np.reshape(jac, basis.shape).T
        jac = np.array(jac)

        if inverted:
            out_matrix = -(jac.T @ basis)
        else:
            out_matrix = jac @ basis

        return out_matrix

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

def rungekutta4_coupled(func1, func2, y0, z0, h):
    #RK4 numerical solver for a coupled system {x' = f(x), P' = J(x)P}
    #if the step size is negative we need to invert the matrix equation
    
    invert_mat = (h < 0)

    fk1 = apply_func(func1, y0)
    gk1 = apply_matrix_func(func2, z0, y0, inverted=invert_mat)

    fk2 = apply_func(func1, y0 + fk1 * h / 2.)
    gk2 = apply_matrix_func(func2, z0 + gk1 * h / 2., y0 + fk1 * h / 2., inverted=invert_mat)

    fk3 = apply_func(func1, y0 + fk2 * h / 2.)
    gk3 = apply_matrix_func(func2, z0 + gk2 * h / 2., y0 + fk2 * h / 2., inverted=invert_mat)

    fk4 = apply_func(func1, y0 + fk3 * h)
    gk4 = apply_matrix_func(func2, z0 + gk3 * h, y0 + fk3 * h, inverted=invert_mat)


    y = y0 + (h / 6.) * (fk1 + 2*fk2 + 2*fk3 + fk4)
    z = z0 + (h / 6.) * (gk1 + 2*gk2 + 2*gk3 + gk4)
    
    return y, z


if __name__ == "__main__":
    from common_systems import lorenz
    system = lorenz()

    y0 = np.array([1, 2, 3])
    m0 = np.diag([1, 1, 1])
    res = np.empty((100, 3))

    res[0, :] = y0

    functions = system.jacobian.funcs()
    func0 = functions[0]
    func0(*y0)
    print(apply_matrix_func(functions, m0, y0))
    




"""
next steps

make coupled rk4
make jacobian a function
start working on the class to calculate maximal element 

"""
