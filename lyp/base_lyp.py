import numpy as np
from abc import ABC, abstractmethod

class Lyp(ABC):
    def __init__(self) -> None:
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

    @abstractmethod
    def exponents(self):
        pass
    
    @abstractmethod
    def vectors(self):
        pass