"""UCL stations package."""

from ucl.stations.design import assume, compile_features, frame, identify, load_data
from ucl.stations.analysis import evaluate, model, test_suite

__all__ = [
    "frame",
    "assume",
    "identify",
    "load_data",
    "compile_features",
    "model",
    "evaluate",
    "test_suite",
]
