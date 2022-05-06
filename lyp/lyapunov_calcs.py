"""

    Definition of different methods of calculating the Lyapunov Exponents and vectors
"""
import numpy as np
from common_systems import lorenz
from abc import ABC, abstractmethod
from integration import rungekutta4, rungekutta4_coupled
import visualisations as plot
# from numba import njit

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

    @abstractmethod
    def plot_exponents(self, y_lims=False):
        pass
    
    def trajectory(self, ic=None):
        # The trajectory is run for 2x the number of steps and the first half are then dumped
        if ic is None:
            ic = np.random.rand(self.dim[1])

        traj = np.empty((self.num_steps * 2, self.dim[1]))
        traj[0, :] = ic

        functions = self.system.funcs()

        for i in range(1, self.num_steps * 2):
            traj[i, :] = rungekutta4(functions, traj[i-1, :], self.time_step)

        return traj[self.num_steps:]

class NonCovarientLyp(Lyp):
    _name = "Non-Covarient"
    def __init__(self, system) -> None:
        Lyp.__init__(self, system)
        self.exp = None
        self.vec = None

        
        self.delta = 0.001    # Small pertibation for initial vector
        self.drop_perc = 0.8  # Percentage of array to drop to calculate ave Lyapunov exp

    @abstractmethod
    def _set_data(self):
        pass

    def max_lyapunov_exp(self, ic=None, min_time=None, max_time=None, time_step=None, delta=None, drop_sec=None):
        """ Function to return an array holding the evolution of the maximum lyapunov exponent
        """
        min_time = self.min_time if min_time is None else min_time
        max_time = self.max_time if max_time is None else max_time
        time_step = self.time_step if time_step is None else time_step
        delta = self.delta if delta is None else delta

        # Run trajectory to ensure ic is on the attractor


        ic = self.trajectory(ic)[-1]

        traj = np.empty((self.num_steps, self.dim[1]))
        traj_del = np.empty_like(traj)

        ly_exp = np.empty(self.num_steps-1)

        traj[0] = ic
        traj_del[0] = ic + delta

        functions = self.system.funcs()

        for i in range(1, self.num_steps):
            traj[i] = rungekutta4(functions, traj[i-1], time_step)
            traj_del[i] = rungekutta4(functions, traj_del[i-1], time_step)

            ly_exp[i-1], err_vec = self._calculate_ly_exp(traj[i], traj_del[i], delta0=delta)
            traj_del[i] = traj[i] + self._normalise_vector(err_vec, delta)

        time_div = np.cumsum(ly_exp) / np.linspace(min_time+1, max_time, self.num_steps-1)

        # Drop starting section
        perc_to_drop = 0.8 if drop_sec is None else drop_sec
        return np.mean(time_div[int(perc_to_drop * self.num_steps):])


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
        ini_pont = self.trajectory(ic=ini_point)[-1]

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
        ini_pont = self.trajectory(ic=ini_point)[-1]

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

    def plot_exponents(self, exp=None, y_lims=False, show=True):
        if self.exp is None and exp is None:
            self._set_data()

        plot.plot_exponents(self.exp, y_lims=y_lims, show=show)

    def exponents_average(self):
        if self.exp is None:
            self._set_data()
        return np.mean(self.exp[int(self.num_steps * 0.8):], axis=0)

    def exponent_alter_param(self, parameter, start, end, resolution):
        exp_hold = list()

        params = np.array(self.system.parameters)
        orig_params = self.system.parm_vals
        idx = np.where(params == parameter)[0][0]

        alter_par = np.linspace(start, end, resolution)
        for p in alter_par:
            orig_params[idx] = p
            self.system.update_parameters(orig_params)
            self._set_data()
            exp_hold.append(self.exponents_average())
        return np.array(exp_hold)


class CovarientLyp(Lyp):
    _name = "Covarient"
    def __init__(self, system) -> None:
        Lyp.__init__(self, system)
        self.exp = None
        self.vec = None


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

        
class Lyapunov(Lyp):

    _name = "Lyapunov"

    def __init__(self, system) -> None:
        Lyp.__init__(self, system)

    def plot_trajectory(self, trajectory=None, variables=None):
        if trajectory is None:
            trajectory = self.trajectory()
        
        if self.dim[1] == 2:
            # Plot a 2-D trajectory
            plot._two_dim_traj(trajectory, variables=self.system.variables)
        elif self.dim[1] == 3:
            plot._three_dim_traj(trajectory, variables=self.system.variables)
        elif self.dim[1] > 3:
            if variables is not None and len(variables) <=3:
                # Find the columns of the trajectory to plot
                vars = np.array(self.system.variables)
                idx = list()
                for v in variables:
                    idx.append(np.where(vars == v)[0][0]) # We are assuming that each variable only appears once
                
                self.plot_trajectory(trajectory=trajectory[:, idx], variables=variables[idx])
            else:
                print("Expecting user to input 2-3 variables to project high dimensional trajectory to lower dimension.")
                print("Use plot_trajectory(trajectory, variables='list of symbols of length < 4')")


    def plot_exponents(self, y_lims=False):
        print("Set up a Covariant or NonCovariant class")

    @property
    def forwards(self):
        return Fowards(self.system)

    @property
    def backwards(self):
        return Backwards(self.system)


# TODO: Read papers about calculating the covarient vectors
# TODO: Make a visualisation class


if __name__ == "__main__":
    from sympy import symbols

    rho = symbols('rho')

    system = lorenz()
    diag = Lyapunov(system)
    # print(diag.forwards.max_lyapunov_exp())
    print(diag.forwards.exponent_alter_param(rho, 0, 10, 10))
    # print(diag.forwards.exponents_average())