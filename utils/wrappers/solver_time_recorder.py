from typing import TypeVar, Any
from csnlp import Nlp
from csnlp.wrappers import Wrapper
import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)


class SolverTimeRecorder(Wrapper[SymType]):
    """A wrapper class of that records the time taken by the solver."""

    def __init__(self, nlp: Nlp[SymType]) -> None:
        super().__init__(nlp)
        self.solver_time: list[float] = []

    def solve(self, *args: Any, **kwds: Any) -> Any:
        sol = self.nlp.solve(*args, **kwds)
        self.solver_time.append(sol.stats["t_wall_total"])
        return sol
