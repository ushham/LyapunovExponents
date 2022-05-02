"""

    Definition of different methods of calculating the Lyapunov Exponents and vectors
"""

from common_systems import lorenz
from integration import rungekutta4
from system import DynamicalSystem
from sympy import symbols, Matrix
import numpy as np
from abc import ABC


class LyapunovCalculations():

    def __init__(self, system) -> None:
        self.system = system

        self.dim = system.__len__()

    @staticmethod
    def _normalise_vector(vec, dist):
        scaling_factor = dist / np.linalg.norm(vec)
        return scaling_factor * vec

    @staticmethod
    def _calculate_ly_exp(vec1, vec2, delta0):
        err_vec = vec2 - vec1
        delta = np.linalg.norm(err_vec)

        exponent = np.log(delta / delta0)
        return exponent, err_vec

    def _gram_schmidt(self, basis):
        """
            Take a set of basis vectors and orthonormalise these.
            Here the first basis vector is kept in the same direction.
            The orthogonalised basis vectors and the scaling coefficients are returned.
        """
        dimentions = self.system.dimension()
        out_vecs = np.empty_like(basis)
        exponents = np.empty(dimentions)
        
        for i in range(dimentions):
            vec = basis[i]
            for j in range(i):
                vec -= np.dot(vec, out_vecs[j]) * out_vecs[j]
            print(vec)

            # TODO: Issue with np.linalg.norm
            exponents[i] = np.linalg.norm(vec)
            out_vecs[i] = vec / exponents[i]

        return out_vecs, exponents

    def _phi_dot(self, basis, location):
        """
            Function to calculate the system P' = J(location) P.
            This is done by calculating how the Jacobian (at the gievn location)
            alters the given basis directions.
            This function is designed to provide the tendencies to then be used by an integrator to calculate the value of P.
        """

        jac = self.system.jacobian.sub_values(location=location, matrix_fmt=True)
        jac = np.array(jac)

        return jac @ basis      

    def trajectory(self, ic=None, max_time=100, time_step=0.01):
        if ic is None:
            ic = np.random.rand(self.dim[0])

        min_time = 0
        steps = int((max_time - min_time) / time_step)
        traj = np.empty((steps, self.dim[0]))
        traj[0, :] = ic

        functions = self.system.num_funcs()

        for i in range(1, steps):
            traj[i, :] = rungekutta4(functions, traj[i-1, :], time_step)

        return traj

    def max_lyapunov_exp(self, ic=None, max_time=100, time_step=0.01, delta=0.001, drop_sec=None):
        """ Function to return an array holding the evolution of the maximum lyapunov exponent

        """
        # Run trajectory to ensure ic is on the attractor
        ic = self.trajectory(ic)[-1, :]

        min_time = 0
        steps = int((max_time - min_time) / time_step)

        traj = np.empty((steps, self.dim[0]))
        traj_del = np.empty_like(traj)

        ly_exp = np.empty(steps-1)

        traj[0] = ic
        traj_del[0] = ic + delta

        functions = self.system.num_funcs()

        for i in range(1, steps):
            traj[i] = rungekutta4(functions, traj[i-1], time_step)
            traj_del[i] = rungekutta4(functions, traj_del[i-1], time_step)

            ly_exp[i-1], err_vec = self._calculate_ly_exp(traj[i], traj_del[i], delta0=delta)
            traj_del[i] = traj[i] + self._normalise_vector(err_vec, delta)

        time_div = np.cumsum(ly_exp) / np.linspace(min_time, max_time, steps-1)

        # Drop starting section
        perc_to_drop = 0.25 if drop_sec is None else drop_sec
        return time_div[int(perc_to_drop * steps):]

    def gram_schmidt_method():
        """
            Calculate all the Lyapunov exponents by calculating the limit of the backwards Lyapunov vectors.
        """
        # Finish coding the gs function
        # link up RK4 method with coupled system
        # make variables to store the exponents and the vectors
        # check if the vectors calculated here are of any use
        return 0

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    system = lorenz()
    diag = LyapunovCalculations(system)
    # res = diag.max_lyapunov_exp(drop_sec=0.01)
    # plt.plot(res)
    # plt.show()
    print(diag._gram_schmidt(np.array(system.jacobian.sub_values(location=[1, 1, 1], matrix_fmt=True))))
    