"""

    Definition of different methods of calculating the Lyapunov Exponents and vectors
"""

from common_systems import lorenz
from integration import rungekutta4, rungekutta4_coupled
import numpy as np
from abc import ABC, abstractmethod
from non_covarient_lyp import Fowards

class LyapunovCalculations(ABC):
    def __init__(self, system) -> None:
        self.system = system
        self.dim = system.model_dims
        self.backwards = None
        self.forwards =  None

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

    @staticmethod
    def _process_lyp_exp(exp_arr, min_time, max_time, time_step):
        time = np.linspace(min_time, max_time, exp_arr.shape[0])
        exp_arr = np.cumsum(np.log(exp_arr), axis=0)

        for i in range(exp_arr.shape[1]):
            exp_arr[:, i] = np.nan_to_num(exp_arr[:, i] / time) #- np.sum(exp_arr[:, :i], axis=1)
        return exp_arr

    @abstractmethod
    def _gram_schmidt(self, basis):
        """
            Take a set of basis vectors and orthonormalise these.
            Here the first basis vector is kept in the same direction.
            The orthogonalised basis vectors and the scaling coefficients are returned.
        """
        
        pass     

    @abstractmethod
    def trajectory(self, ic, min_time, max_time, time_step):
        
        pass

    @abstractmethod
    def max_lyapunov_exp(self, ic, min_time, max_time, time_step, delta, drop_sec):
        """ Function to return an array holding the evolution of the maximum lyapunov exponent

        """
        pass

    @abstractmethod
    def gram_schmidt_method(self, min_time, max_time, time_step, ini_point):
        """
            Calculate all the Lyapunov exponents by calculating the limit of the backwards Lyapunov vectors.
        """

        pass

    @abstractmethod
    def reverse_gram_schmidt(self, min_time, max_time, time_step, ini_point):
        """
            Calculate the Lyapunov exponents and the forward Lyapunov vectors by integrating the system backwards in time.
        """

        pass

class Lyapunov(LyapunovCalculations):

    # def __init__(self, system) -> None:

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
        for n in range(1, num_steps):
            traj[n], basis[n] = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n-1], basis[n-1], time_step)
            basis[n], lypunov_exp[n-1] = self._gram_schmidt(basis[n])

        # Last row is zero due to starting loop from 1
        lypunov_exp[-1] = lypunov_exp[-2]
        return self._process_lyp_exp(lypunov_exp, min_time=min_time, max_time=max_time, time_step=time_step)

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
        basis[-1] = ini_basis

        sys_funcs, jac_funcs = self.system.funcs(), self.system.jacobian.funcs()
        # We then run a trajectory forwards and save this as backwards integration leads to instability
        for n in range(1, num_steps):
            traj[n], basis_temp = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n-1], ini_basis, time_step)

        for n in range(num_steps-2, -1, -1):
            traj_temp, basis[n] = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n+1], basis[n+1], -time_step)
            # basis[n] = np.linalg.inv(basis[n])
            basis[n], lypunov_exp[n] = self._gram_schmidt(basis[n])

        # First row is zero due to 
        lypunov_exp[-1] = lypunov_exp[-2]
        return self._process_lyp_exp(lypunov_exp, min_time=min_time, max_time=max_time, time_step=time_step)

    def lyapunov_exp(self, runs = 1, backwards=True):
        lyp = list()
        for i in range(runs):
            if backwards:
                lyp.append(self.gram_schmidt_method())
            else:
                lyp.append(self.reverse_gram_schmidt())

        return np.array(lyp)

    @property
    def forwards(self):
        return Fowards()
        

        


# TODO: Read papers about calculating the covarient vectors
# TODO: Move most of functions to 
            

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    system = lorenz()
    diag = Lyapunov(system)
    res = diag.max_lyapunov_exp(drop_sec=0.01)
    # plt.plot(res)

    ll = diag.reverse_gram_schmidt()
    # ll = diag.lyapunov_exp()
    print(ll)
    plt.plot(ll[200:])
    # plt.ylim(-1, 2)
    plt.show()