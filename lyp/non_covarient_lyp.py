from abc import ABC, abstractmethod
from base_lyp import Lyp
from integration import rungekutta4, rungekutta4_coupled
import numpy as np

class NonCovarientLyp(Lyp):
    def __init__(self) -> None:
        self.exp = None
        self.vec = None
    
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

    # @abstractmethod
    # def exponents(self):
    #     pass
    
    # @abstractmethod
    # def vectors(self):
    #     pass



class Fowards(NonCovarientLyp):
    def __init__(self) -> None:
        pass

    def exponents(self):
        return 0
    
    def vectors(self):
        return 0