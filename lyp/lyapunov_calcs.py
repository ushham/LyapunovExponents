"""

    Definition of different methods of calculating the Lyapunov Exponents and vectors
"""

from common_systems import lorenz
from integration import rungekutta4, rungekutta4_coupled
import numpy as np
from abc import ABC, abstractmethod

class Lyp(ABC):

    _name = ""

    def __init__(self, system) -> None:
        self.system = system
        self.dim = system.model_dims

        self.min_time = 0
        self.max_time = 100
        self.time_step = 0.01
        self.ini_point = None

        self.num_steps = int((self.max_time - self.min_time) / self.time_step)

    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val

    def __str__(self):
        s = ""
        for key, val in zip(self.__dict__.keys(), self.__dict__.values()):
            if key == 'system':
                pass
            else:
                s += "'"+key+"': "+str(val)+",\n"
        return s

    def _list_params(self):
        return self._name +" Parameters:\n"+self.__str__()

    def print_params(self):
        """Print the parameters contained in the container."""
        print(self._list_params())

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
    def _process_lyp_exp(exp_arr, min_time, max_time, num_steps):
        time = np.linspace(min_time, max_time, num_steps)
        time[time==0] = np.nan

        exp_arr = np.cumsum(np.log(exp_arr), axis=0, dtype=np.float64)

        for i in range(exp_arr.shape[1]):
            exp_arr[:, i] = np.divide(exp_arr[:, i], time)
        return exp_arr

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

class NonCovarientLyp(Lyp):
    _name = "Non-Covarient"
    def __init__(self, system) -> None:
        Lyp.__init__(self, system)
        self.exp = None
        self.vec = None

    @abstractmethod
    def _set_data(self):
        pass

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

    def _gram_schmidt_method(self, min_time, max_time, time_step, ini_point):
        """
            Calculate all the Lyapunov exponents by calculating the limit of the backwards Lyapunov vectors.
        """
        
        # Run the trajectory from a random initial point for the given time to ensure trajectory is on the attractor
        ini_pont = self.trajectory(ic=ini_point, min_time=min_time, max_time=max_time, time_step=time_step)[-1]

        traj = np.empty((self.num_steps, self.dim[1]))
        basis = np.empty((self.num_steps, self.dim[0], self.dim[1]))
        lypunov_exp = np.empty_like(traj)

        ini_basis = np.zeros(self.dim)
        np.fill_diagonal(ini_basis, 1)

        traj[0] = ini_pont
        basis[0] = ini_basis

        sys_funcs, jac_funcs = self.system.funcs(), self.system.jacobian.funcs()
        for n in range(1, self.num_steps):
            traj[n], basis[n] = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n-1], basis[n-1], time_step)
            basis[n], lypunov_exp[n-1] = self._gram_schmidt(basis[n])

        # Last row is zero due to starting loop from 1
        lypunov_exp[-1] = lypunov_exp[-2]
        return self._process_lyp_exp(lypunov_exp, min_time=min_time, max_time=max_time, num_steps=self.num_steps), basis


    def _reverse_gram_schmidt(self, min_time, max_time, time_step, ini_point):
        """
            Calculate the Lyapunov exponents and the forward Lyapunov vectors by integrating the system backwards in time.
        """

        # Run the trajectory from a random initial point for the given time to ensure trajectory is on the attractor
        ini_pont = self.trajectory(ic=ini_point, min_time=min_time, max_time=max_time, time_step=time_step)[-1]

        traj = np.empty((self.num_steps, self.dim[1]))
        basis = np.empty((self.num_steps, self.dim[0], self.dim[1]))
        lypunov_exp = np.empty_like(traj)

        ini_basis = np.zeros(self.dim)
        np.fill_diagonal(ini_basis, 1)

        traj[0] = ini_pont
        basis[-1] = ini_basis

        sys_funcs, jac_funcs = self.system.funcs(), self.system.jacobian.funcs()
        # We then run a trajectory forwards and save this as backwards integration leads to instability
        for n in range(1, self.num_steps):
            traj[n], basis_temp = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n-1], ini_basis, time_step)

        for n in range(self.num_steps-2, -1, -1):
            traj_temp, basis[n] = rungekutta4_coupled(sys_funcs, jac_funcs, traj[n+1], basis[n+1], -time_step)
            # basis[n] = np.linalg.inv(basis[n])
            basis[n], lypunov_exp[n] = self._gram_schmidt(basis[n])

        # First row is zero due to 
        lypunov_exp[-1] = lypunov_exp[-2]
        return self._process_lyp_exp(lypunov_exp, min_time=min_time, max_time=max_time, num_steps=self.num_steps), basis



class Fowards(NonCovarientLyp):

    def __init__(self, system) -> None:
        NonCovarientLyp.__init__(self, system)

    def _set_data(self):
        self.exp, self.vec = self._reverse_gram_schmidt(min_time=self.min_time, max_time=self.max_time, time_step=self.time_step, ini_point=self.ini_point)
        self.time_steps = self.exp.shape[0]

    def exponents(self):
        if self.exp is None:
            self._set_data()
        return self.exp
    
    def vectors(self):
        if self.vec is None:
            self._set_data()
        return self.vec

    def exponents_average(self):
        if self.exp is None:
            self._set_data()
        return np.mean(self.exp[int(self.num_steps * 0.8):], axis=0)
        

class Backwards(NonCovarientLyp):

    def __init__(self, system) -> None:
        NonCovarientLyp.__init__(self, system)

    def _set_data(self):
        self.exp, self.vec = self._gram_schmidt_method(min_time=self.min_time, max_time=self.max_time, time_step=self.time_step, ini_point=self.ini_point)
        self.time_steps = self.exp.shape[0]

    def exponents(self):
        if self.exp is None:
            self._set_data()
        return self.exp
    
    def vectors(self):
        if self.vec is None:
            self._set_data()
        return self.vec

    def exponents_average(self):
        if self.exp is None:
            self._set_data()
        return np.mean(self.exp[int(self.num_steps * 0.8):], axis=0)
        

class Lyapunov(Lyp):

    _name = "Lyapunov"

    def __init__(self, system) -> None:
        Lyp.__init__(self, system)


    @property
    def forwards(self):
        return Fowards(self.system)

    @property
    def backwards(self):
        return Backwards(self.system)


# TODO: Read papers about calculating the covarient vectors
# TODO: Make a visualisation class


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    system = lorenz()
    diag = Lyapunov(system)
    print(diag.forwards.exponents_average())




    # # ll = diag.lyapunov_exp()
    # plt.plot(fwd[200:])
    # plt.ylim(-1, 2)
    # plt.show()