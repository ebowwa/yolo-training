"""
Gaze-conditioned learning module.

Provides gaze encoding, intent decoding, and reward shaping for egocentric vision.
"""

from .intent_decoder import (
    GazePoint,
    GazeSequence,
    GazeEncoder,
    GazeIntentDecoder,
    GazeRewardShaper,
)

__all__ = [
    "GazePoint",
    "GazeSequence", 
    "GazeEncoder",
    "GazeIntentDecoder",
    "GazeRewardShaper",
]
