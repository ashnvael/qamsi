############################################################################
### QPMwP - OPTIMIZATION
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
from abc import ABC, abstractmethod
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd

# Local modules
from qamsi.optimization.helper_functions import to_numpy
from qamsi.optimization.constraints import Constraints
from qamsi.optimization.quadratic_program import QuadraticProgram


class Objective:
    """
    A class to handle the objective function of an optimization problem.

    Parameters:
    kwargs: Keyword arguments to initialize the coefficients dictionary. E.g. P, q, constant.
    """

    def __init__(self, **kwargs):
        self.coefficients = kwargs

    @property
    def coefficients(self) -> dict:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value: dict) -> None:
        if isinstance(value, dict):
            self._coefficients = value
        else:
            raise ValueError("Input value must be a dictionary.")


class OptimizationParameter(dict):
    """
    A class to handle optimization parameters.

    Parameters:
    kwargs: Additional keyword arguments to initialize the dictionary.
    """

    def __init__(self, **kwargs):
        super().__init__(
            solver_name="cvxopt",
        )
        self.update(kwargs)


class Optimization(ABC):
    """
    Abstract base class for optimization problems.

    Parameters:
    params (OptimizationParameter): Optimization parameters.
    kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        params: Optional[OptimizationParameter] = None,
        constraints: Optional[Constraints] = None,
        **kwargs,
    ):
        self.params = OptimizationParameter() if params is None else params
        self.params.update(**kwargs)
        self.constraints = Constraints() if constraints is None else constraints
        self.objective: Objective = Objective()
        self.results = {}

    @abstractmethod
    def set_objective(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Method 'set_objective' must be implemented in derived class."
        )

    @abstractmethod
    def solve(self) -> None:
        # TODO:
        # Check consistency of constraints
        # self.check_constraints()

        # Get the coefficients of the objective function
        obj_coeff = self.objective.coefficients
        if "P" not in obj_coeff.keys() or "q" not in obj_coeff.keys():
            raise ValueError("Objective must contain 'P' and 'q'.")

        # Ensure that P and q are numpy arrays
        obj_coeff["P"] = to_numpy(obj_coeff["P"])
        obj_coeff["q"] = to_numpy(obj_coeff["q"])

        self.solve_qpsolvers()
        return None

    def solve_qpsolvers(self) -> None:
        self.model_qpsolvers()
        self.model.solve()

        solution = self.model.results["solution"]
        status = solution.found
        ids = self.constraints.ids
        weights = pd.Series(
            solution.x[: len(ids)] if status else [None] * len(ids), index=ids
        )

        self.results.update(
            {
                "weights": weights.to_dict(),
                "status": self.model.results["solution"].found,
            }
        )

        return None

    def model_qpsolvers(self) -> None:
        # constraints
        constraints = self.constraints
        GhAb = constraints.to_GhAb()
        lb = (
            constraints.box["lower"].to_numpy()
            if constraints.box["box_type"] != "NA"
            else None
        )
        ub = (
            constraints.box["upper"].to_numpy()
            if constraints.box["box_type"] != "NA"
            else None
        )

        # Create the optimization model as a QuadraticProgram
        self.model = QuadraticProgram(
            P=self.objective.coefficients["P"],
            q=self.objective.coefficients["q"],
            G=GhAb["G"],
            h=GhAb["h"],
            A=GhAb["A"],
            b=GhAb["b"],
            lb=lb,
            ub=ub,
            solver_settings=self.params,
        )

        # TODO:
        # [ ] Add turnover penalty in the objective
        # [ ] Add turnover constraint
        # [ ] Add leverage constraint

        return None


class VarianceMinimizer(Optimization):
    def __init__(
        self,
        constraints: Constraints,
        **kwargs,
    ):
        super().__init__(constraints=constraints, **kwargs)

        self.asset_names = constraints.ids

    def set_objective(self, covmat: pd.DataFrame) -> None:
        self.objective = Objective(
            P=covmat,
            q=np.zeros(covmat.shape[0]),
        )
        return None

    def solve(self) -> None:
        GhAb = self.constraints.to_GhAb()
        if GhAb["G"] is None and self.constraints.box["box_type"] == "Unbounded":
            A = GhAb["A"]
            b = GhAb["b"]
            # If b is scalar, convert it to a 1D array
            if isinstance(b, (int, float)):
                b = np.array([b])
            elif b.ndim == 0:
                b = np.array([b])

            P = self.objective.coefficients["P"]
            P_inv = np.linalg.inv(P)

            AP_invA = A @ P_inv @ A.T
            if AP_invA.shape[0] > 1:
                AP_invA_inv = np.linalg.inv(AP_invA)
            else:
                AP_invA_inv = 1 / AP_invA
            x = pd.Series(P_inv @ A.T @ AP_invA_inv @ b, index=self.constraints.ids)
            self.results.update(
                {
                    "weights": x.to_dict(),
                    "status": True,
                }
            )
            return None
        else:
            return super().solve()
