"""
Backward-compatibility shim.

All classes and constants from this module have been moved to src.datasource.
Imports of the form:
    from src.apex.modules.apex_datasource import ApexWindowDataSource
    from src.apex.modules.apex_datasource import BaseDataSource, LABEL_MAP, CACHE_VERSION
continue to work unchanged.
"""

from src.datasource.base_datasource import BaseDataSource, LABEL_MAP, CACHE_VERSION
from src.datasource.apex_window_datasource import ApexWindowDataSource
from src.datasource.apex_full_phase_datasource import ApexFullPhaseDataSource
from src.datasource.apex_hybrid_datasource import ApexHybridDataSource

__all__ = [
    "BaseDataSource",
    "ApexWindowDataSource",
    "ApexFullPhaseDataSource",
    "ApexHybridDataSource",
    "LABEL_MAP",
    "CACHE_VERSION",
]
