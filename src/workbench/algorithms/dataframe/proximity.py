from abc import ABC, abstractmethod
import pandas as pd
from typing import Union


class Proximity(ABC):
    def __init__(self, df: pd.DataFrame, id_column: str, n_neighbors: int) -> None:
        self.df = df
        self.id_column = id_column
        self.n_neighbors = n_neighbors

    @abstractmethod
    def all_neighbors(self, include_self: bool = False) -> pd.DataFrame:
        """Compute nearest neighbors for all rows."""
        pass

    @abstractmethod
    def neighbors(
        self, query_id: Union[str, int], similarity: float = None, include_self: bool = False
    ) -> pd.DataFrame:
        """Return neighbors of the given query ID"""
        pass
