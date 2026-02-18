import math


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
    R = 6_371_000  # mean R

    lat1_rad = math.radians(lat1)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    if delta_lon > math.pi:
        delta_lon -= 2 * math.pi
    elif delta_lon < -math.pi:
        delta_lon += 2 * math.pi

    y = R * delta_lat

    x = R * delta_lon * math.cos(lat1_rad)

    return x, y
