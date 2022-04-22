"""
    Definition of system (base class)

    Abstract base classes defining default dynamical systems and allowing users to input their own system.

    Description of the classes:

"""
from abc import ABC
import sys
from sympy import symbols, lambdify, diff, Matrix


# TODO: I am converting from sympy.Matrix objects to lists and vice versa, is this ok?

class System(ABC):
    """ General base class for the dynamical systems.

    Attributes
    ----------
    functions: list
        List of dynamical equations
    """

    def __init__(self) -> None:
        self.variables = list()
        self.parameters = list()
        self.parm_vals = list()
        self.equations = list()

    def __str__(self) -> str:
        return self.variables.__str__(), self.equations.__str__()

    def __len__(self):
        var_len = self.variables.__len__()
        eq_len = self.equations.__len__()
        if var_len != eq_len:
            print("Variable length does not match equation length")

        return var_len, eq_len

    def dimension(self):
        return Matrix(self.equations).shape

    def append(self, var, parms, item):
        self.variables.append(var)
        self.parameters.append(parms)
        self.equations.append(item)

    def model_definition(self, var, parms, item):
        if (var is not None) and (parms is not None) and (item is not None):
            for v, p, it in zip(var, parms, item):
                self.append(v, p, it)
        
            self.calculate_jacobian()

    def base_parameters(self, params):
        self.parm_vals = params
        self.jacobian.parm_vals = params

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


    def sub_values(self, new_parameters=None, location=None, matrix_fmt=False):
        """Return the equation functions with the substitutions stored in the object being applied.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the equations. The 2-tuples contain first
            a `Sympy`_  expression and then the value to substitute.

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
            self.jacobian.append(v, p, self.derivative(v).equations)
        
        self.jacobian.parameters = self.parameters
        self.jacobian.parm_vals = self.parm_vals
        self.jacobian.substitutions = self.substitutions

    def num_funcs(self, new_parameters=None, location=None):
        """Return the basis functions with as python callable.

        Parameters
        ----------
        extra_subs: list(tuple), optional
            List of 2-tuples containing extra substitutions to be made with the functions before transforming them into
            python callable. The 2-tuples contain first a `Sympy`_  expression and then the value to substitute.

        Returns
        -------
        list(callable)
            List of callable basis functions
        """

        nf = list()
        eqations = self.sub_values(new_parameters=new_parameters, location=location)

        for eq in eqations:
            try:
                nf.append(lambdify(self.variables, eq))
            except:
                tb = sys.exc_info()[2]
                raise Exception.with_traceback(tb)
        
        return nf

    
    

class DynamicalSystem(System):
    """General base class for symbolic dynamic equations

    Parameters
    ----------
    """

    def __init__(self, var=None, params=None, item=None) -> None:
        System.__init__(self)
        self.substitutions = list()
        self.model_definition(var, params, item)

    def convert_location(self, locs):
        ll = list()
        for v, l in zip(self.variables, locs):
            ll.append((v, l))
        return ll


    

if __name__ == "__main__":
    from sympy import symbols

    #Example of implementing new dynamical system
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

    lorenz = DynamicalSystem(variables, parameters, lorenz_equations)
    lorenz.base_parameters(parms)

    print(lorenz.sub_parameters())

    print(lorenz.jacobian.sub_parameters(matrix_fmt=True))
    funcs = lorenz.jacobian.num_funcs()
    print(len(funcs))
