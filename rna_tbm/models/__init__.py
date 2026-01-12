"""
Deep learning models for RNA structure prediction.
"""

from .distance_predictor import (
    DistancePredictorNumpy,
    build_pairwise_features,
    DISTANCE_BINS,
    NUM_BINS,
)

from .distance_to_coords import (
    DistanceToCoords,
    HybridCoordinateGenerator,
)

from .contact_map import (
    ContactPredictorNumpy,
    predict_contacts_combined,
)

# Try to import PyTorch versions if available
try:
    from .distance_predictor import DistancePredictor, ResBlock2D
    from .contact_map import ContactPredictorNetwork
    TORCH_MODELS_AVAILABLE = True
except ImportError:
    TORCH_MODELS_AVAILABLE = False

__all__ = [
    # NumPy models (always available)
    'DistancePredictorNumpy',
    'DistanceToCoords',
    'HybridCoordinateGenerator',
    'ContactPredictorNumpy',
    'predict_contacts_combined',
    'build_pairwise_features',
    # Constants
    'DISTANCE_BINS',
    'NUM_BINS',
    'TORCH_MODELS_AVAILABLE',
]

# Add PyTorch models to exports if available
if TORCH_MODELS_AVAILABLE:
    __all__.extend([
        'DistancePredictor',
        'ResBlock2D',
        'ContactPredictorNetwork',
    ])
