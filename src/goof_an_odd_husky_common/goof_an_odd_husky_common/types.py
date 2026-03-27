from dataclasses import dataclass, astuple
from typing import Annotated

import numpy as np
from numpy.typing import NDArray


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)


@dataclass
class OdomDelta:
    dx: float
    dy: float
    dtheta: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class GpsCoord:
    lat: float
    lon: float

    def __iter__(self):
        yield self.lat
        yield self.lon


Trajectory = Annotated[NDArray[np.floating], (None, 4)]
