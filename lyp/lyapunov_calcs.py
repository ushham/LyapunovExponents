"""

    Definition of different methods of calculating the Lyapunov Exponents and vectors
"""

from common_systems import lorenz
from integration import rungekutta4, rungekutta4_coupled
from system import DynamicalSystem
from sympy import symbols, Matrix
import numpy as np
from abc import ABC


class LyapunovCalculations():

    def __init__(self, system) -> None:
        self.system = system

        self.dim = system.model_dims

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
            vec = basis[:, i]
            for j in range(i):
                vec -= np.dot(vec, out_vecs[:, j]) * out_vecs[:, j]

            exponents[i] = np.linalg.norm(vec)
            out_vecs[:, i] = vec / exponents[i]

        return out_vecs, exponents      

    def trajectory(self, ic=None, min_time=0, max_time=100, time_step=0.01):
        if ic is None:
            ic = np.random.rand(self.dim[1])

        steps = int((max_time - min_time) / time_step)
        traj = np.empty((steps, self.dim[1]))
        traj[0, :] = ic

        functions = self.system.funcs()

        for i in range(1, steps):
            traj[i, :] = rungekutta4(functions, traj[i-1, :], time_step)

        return traj

    def max_lyapunov_exp(self, ic=None, min_time=0, max_time=100, time_step=0.01, delta=0.001, drop_sec=None):
        """ Function to return an array holding the evolution of the maximum lyapunov exponent

        """
        # Run trajectory to ensure ic is on the attractor
        ic = self.trajectory(ic)[-1]

        steps = int((max_time - min_time) / time_step)

        traj = np.empty((steps, self.dim[1]))
        traj_del = np.empty_like(traj)

        ly_exp = np.empty(steps-1)

        traj[0] = ic
        traj_del[0] = ic + delta

        functions = self.system.funcs()

        for i in range(1, steps):
            traj[i] = rungekutta4(functions, traj[i-1], time_step)
            traj_del[i] = rungekutta4(functions, traj_del[i-1], time_step)

            ly_exp[i-1], err_vec = self._calculate_ly_exp(traj[i], traj_del[i], delta0=delta)
            traj_del[i] = traj[i] + self._normalise_vector(err_vec, delta)

        time_div = np.cumsum(ly_exp) / np.linspace(min_time+1, max_time, steps-1)

        # Drop starting section
        perc_to_drop = 0.25 if drop_sec is None else drop_sec
        return time_div[int(perc_to_drop * steps):]

    def gram_schmidt_method(self, min_time=0, max_time=100, time_step=0.01, ini_point=None):
        """
            Calculate all the Lyapunov exponents by calculating the limit of the backwards Lyapunov vectors.
        """

        # check if the vectors calculated here are of any use
        num_steps = int((max_time - min_time) / time_step)
        

        # Run the trajectory from a random initial point for the given time to ensure trajectory is on the attractor
        ini_pont = self.trajectory(ic=ini_point, min_time=min_time, max_time=max_time, time_step=time_step)[-1]

        traj = np.empty((num_steps, self.dim[1]))
        basis = np.empty((num_steps, self.dim[0], self.dim[1]))
        lypunov_exp = np.empty_like(traj)

        ini_basis = np.zeros(self.dim)
        np.fill_diagonal(ini_basis, 1)

        traj[0] = ini_pont
        basis[0] = ini_basis

        sys_funcs, jac_funcs = self.system.funcs(), self.system.jacobian.funcs()
        ly1 = 0
        for n in range(1, num_steps):
            traj[n], basis[n] = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n-1], basis[n-1], time_step)
            basis[n], lypunov_exp[n-1] = self._gram_schmidt(basis[n])

            ly1 += np.log(lypunov_exp[n-1, 0])

        # First row is zero due to starting loop from 1
        lypunov_exp[-1] = lypunov_exp[-2]
        return lypunov_exp, ly1

    def reverse_gram_schmidt(self, min_time=0, max_time=100, time_step=0.01, ini_point=None):
        """
            Calculate the Lyapunov exponents and the forward Lyapunov vectors by integrating the system backwards in time.
        """

        # check if the vectors calculated here are of any use
        num_steps = int((max_time - min_time) / time_step)
        

        # Run the trajectory from a random initial point for the given time to ensure trajectory is on the attractor
        ini_pont = self.trajectory(ic=ini_point, min_time=min_time, max_time=max_time, time_step=time_step)[-1]

        traj = np.empty((num_steps, self.dim[1]))
        basis = np.empty((num_steps, self.dim[0], self.dim[1]))
        lypunov_exp = np.empty_like(traj)

        ini_basis = np.zeros(self.dim)
        np.fill_diagonal(ini_basis, 1)

        traj[0] = ini_pont
        basis[0] = ini_basis

        sys_funcs, jac_funcs = self.system.funcs(), self.system.jacobian.funcs()
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    system = lorenz()
    diag = LyapunovCalculations(system)
    res = diag.max_lyapunov_exp(drop_sec=0.01)
    plt.plot(res)

    # vec = diag._gram_schmidt(np.array(system.jacobian.sub_values(location=[1, 1, 1], matrix_fmt=True, numpy_fmt=True)))
    ll, ly1 = diag.gram_schmidt_method(ini_point=np.array([19, 20, 50]))
    time = np.linspace(1, 100, 10000)
    ll = np.log(ll)
    ll = np.cumsum(ll, axis=0)
   
    plt.plot(ll[:, 0] / time[:])
    plt.show()