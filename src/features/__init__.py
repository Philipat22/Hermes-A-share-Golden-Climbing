"""
src/features - Alpha Factor Library
Extracts 190+ WorldQuant Alpha 101 + Alpha 158 factors from price/volume data.
"""
from .feature_generator import FeatureGenerator, compute_features
from .engine import DataProxy, calculate_by_expression

__all__ = [
    "FeatureGenerator",
    "compute_features",
    "DataProxy",
    "calculate_by_expression",
]
