import math
import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def gps_to_vector(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float]:
    """Convert two GPS coordinates to a displacement vector (x, y) in meters.

    Uses local tangent plane approximation (flat-Earth model).
    Valid for distances < ~100 km with <0.5% error.

    Args:
        lat1, lon1: Origin point latitude and longitude (decimal degrees)
        lat2, lon2: Target point latitude and longitude (decimal degrees)

    Returns:
        tuple: (x, y) where:
            - x = East displacement (meters)
            - y = North displacement (meters)
    """
    R = 6_371_000  # mean R of Earth

    lat1_rad = math.radians(lat1)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = normalize_angle(math.radians(lon2 - lon1))

    y = R * delta_lat
    x = R * delta_lon * math.cos(lat1_rad)

    return x, y


def point_segment_distance(Px, Py, S1x, S1y, S2x, S2y):
    """Vectorized point-to-segment distance."""
    S1S2_x = S2x - S1x
    S1S2_y = S2y - S1y
    S1P_x = Px - S1x
    S1P_y = Py - S1y

    len_sq = S1S2_x**2 + S1S2_y**2
    len_sq = np.maximum(len_sq, 1e-10)

    t = (S1P_x * S1S2_x + S1P_y * S1S2_y) / len_sq
    t = np.clip(t, 0.0, 1.0)

    Proj_x = S1x + t * S1S2_x
    Proj_y = S1y + t * S1S2_y

    u_x = Px - Proj_x
    u_y = Py - Proj_y

    d = np.sqrt(u_x**2 + u_y**2 + 1e-10)
    return d, u_x, u_y, t
