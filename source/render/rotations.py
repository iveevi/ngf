import numpy as np

from typing import Tuple

def rotations(basename: str) -> Tuple[float]:
    rotations: dict = {
            'armadillo': [110, -8, -120],
            'dragon': [70, 0, 50],
            'nefertiti': [110, -10, -120],
            'skull': [90, 0, -30],
            'lucy': [90, 0, -120],
            'xyz': [80, -10, -20],
            'buddha': [90, 0, 80],
            'metatron': [90, 0, 180],
            'ogre': [90, 0, 60],
    }

    if basename in rotations:
        return np.radians(rotations[basename])

    return np.radians([90, 0, 0])
