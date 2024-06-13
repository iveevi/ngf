import numpy as np

from typing import Tuple

def rotations(basename: str) -> Tuple[float]:
    rotations: dict = {
            'armadillo': [110, -8, -120],
            'buddha': [90, 0, 80],
            'bust': [0, 0, 0],
            'dragon': [70, 0, 50],
            'einstein': [80, 0, 30],
            'ganesha': [-90, 0, -120],
            'lucy': [90, 0, -120],
            'metatron': [90, 0, 180],
            'nefertiti': [110, -10, -120],
            'ogre': [90, 0, 60],
            'skull': [90, 0, -30],
            'xyz': [85, -11, -36],
            'indonesian': [90, 0, -30],
    }

    if basename in rotations:
        return np.radians(rotations[basename])

    return np.radians([90, 0, 0])
