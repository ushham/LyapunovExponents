"""
    Definition of system (base class)

    Abstract base classes defining default dynamical systems and allowing users to input their own system.

    Description of the classes:

"""
from abc import ABC
import sys
from sympy import symbols, lambdify, diff, Matrix
import numpy as np


# TODO: I am converting from sympy.Matrix objects to lists and vice versa, is this ok?

class System(ABC):
    """ General base class for the dynamical systems.

    Attributes
    ----------
    functions: list
        List of dynamical equations
    """

    _name = ""

    def __init__(self) -> None:
        self.time = None
        self.variables = list()
        self.parameters = list()
        self.parm_vals = list()
        self.equations = list()

        #Model input / output dimensions
        self.model_dims = (None, None)

    def __str__(self) -> str:
        return self.variables.__str__(), self.equations.__str__()

    def __len__(self):
        var_len = self.variables.__len__()
        eq_len = self.equations.__len__()
        if var_len != eq_len:
            print("Variable length does not match equation length")

        return var_len, eq_len

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

    def print_system(self):
        """Print the parameters contained in the container."""
        print(self._list_params())

    def dimension(self):
        return Matrix(self.equations).shape[0]

    def append(self, time, var, parms, item):
        self.time = time
        self.variables.append(var)
        self.parameters.append(parms)
        self.equations.append(item)

    def model_definition(self, time, var, parms, item):
        if (time is not None):
            self.time = time

        if (var is not None):
            for v in var:
                self.variables.append(v)

        if (parms is not None):
            for p in parms:
                self.parameters.append(p)
        
        if (item is not None):
            for it in item:
                self.equations.append(it)
        
        self.calculate_jacobian()

    def update_parameters(self, params):
        self.parm_vals = params
        self.jacobian.parm_vals = params

    def convert_location(self, locs):
        ll = list()
        for v, l in zip(self.variables, locs):
            ll.append((v, l))
        return ll

    def sub_parameters(self, new_parameters=None, matrix_fmt=False):
        """Returns system with parameters substituted for given values.
        
        Parameters
        ----------

        """
        sub_list = list()

        if new_parameters is not None:
            self.parm_vals = new_parameters

        for parms, vals in zip(self.parameters, self.parm_vals):
            sub_list.append((parms, vals))

        matrix_equations = Matrix(self.equations).subs(sub_list)
        output = matrix_equations if matrix_fmt else list(matrix_equations)

        return output


    def sub_values(self, new_parameters=None, location=None, matrix_fmt=False, numpy_fmt=False):
        """Return the equation functions with the substitutions stored in the object being applied.

        Parameters
        ----------
        new_parameters: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the equations. The 2-tuples contain first
            a `Sympy`_  expression and then the value to substitute.
            Default: None

        location: array
            Array of coordinate values to substitute with the symbolic variables.
            Default: None

        matrix_fmt: boolian
            Controls whether the output is in matrix format (True), or a list (False).
            Default: False

        numpy_fmt: boolian
            Controls whether the output is a numpy array with numpy values (True), or a symbolic array (False).
            This only has effect when the output is purely numeric (no symbols).

        Returns
        -------
        list
            List of the substituted equations
        """

        matrix_equations = self.sub_parameters(new_parameters=new_parameters, matrix_fmt=True)
        
        if location is not None:
            locations = self.convert_location(location)
            matrix_equations = matrix_equations.subs(locations)
        else:
            matrix_equations = matrix_equations.subs(self.substitutions)

        output = matrix_equations if matrix_fmt else list(matrix_equations)

        if numpy_fmt:
            output = np.array(output).astype(np.float64)      

        return output

    def derivative(self, symbol, order=1):
        """Returns the equations differentiated wrt the given `symbol`.

        Parameters
        ----------
        sumbol: Sympy symbol
            The symbol with respect to which the basis is to be differentiated.

        order: int, optional
            The order of the derivative. Default to first order.

        Returns
        -------
        DynamicalSystem:
            A new set of equations object with the differentiated equations.
        """
        dequ = list(map(lambda func: diff(func, symbol, order), self.equations))
        dsys = System()
        dsys.variables = self.variables
        dsys.equations = dequ
        dsys.substitutions = self.substitutions
        return dsys

    def calculate_jacobian(self):
        """Returns a symbolic matrix of the Jacobian of the system.

        Returns
        -------
        DynamicalSystem:
            A new 2-d set of equations of the generalised jacobian
        """
        self.jacobian = System()

        for v, p in zip(self.variables, self.parameters):
            self.jacobian.append(self.time, v, p, self.derivative(v).equations)
        
        # Transpose the jacobian as default is transposed
        self.jacobian.equations = list(map(list, zip(*self.jacobian.equations)))

        self.jacobian.parameters = self.parameters
        self.jacobian.parm_vals = self.parm_vals
        self.jacobian.substitutions = self.substitutions

    def funcs(self, new_parameters=None, location=None):
        """Return the basis functions with as python callable.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the functions before transforming them into python callable. 
            The 2-tuples contain first a `Sympy`_  expression and then the value to substitute.

        Returns
        -------
        list(callable)
            List of callable basis functions
        """

        nf = list()
        eqations = self.sub_values(new_parameters=new_parameters, location=location)

        for eq in eqations:
            try:
                vars = [self.time] + self.variables
                nf.append(lambdify(vars, eq))
            except:
                tb = sys.exc_info()[2]
                raise Exception.with_traceback(tb)
        
        return nf
    
class DynamicalSystem(System):
    """General base class for symbolic dynamic equations

    Parameters
    ----------
    """

    _name = "DynamicalSystem"

    def __init__(self, time=None, var=None, params=None, item=None) -> None:
        System.__init__(self)
        self.substitutions = list()
        self.model_definition(time, var, params, item)
        self.update_dims()

    def update_dims(self):
        if self.variables is not None:
            self.model_dims = (len(self.variables), self.dimension())

if __name__ == "__main__":
    from sympy import symbols

    #Example of implementing new dynamical system
    t = symbols('t')
    x, y, z = symbols('x, y, z')
    sigma, rho, beta = symbols('sigma rho beta')

    variables = [x, y, z]
    
    parameters = [sigma, rho, beta]
    parms=[10., 28., 8/3]

    lorenz_equations = [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

    lorenz = DynamicalSystem(time=t, var=variables, params=parameters, item=lorenz_equations)
    lorenz.update_parameters(parms)
    lorenz.print_system()