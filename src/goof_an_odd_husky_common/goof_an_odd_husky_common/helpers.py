from goof_an_odd_husky_common.types import GpsCoord
import math
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def coerce_str(value: str | list[str], allowed: set[str] | None = None) -> str:
    """Safely extracts a string or filters from a list of strings based on allowed elements.

    Args:
        value: The value to extract from (string or list of strings).
        allowed: A set of allowed string values.

    Returns:
        str: The extracted string, or an empty string if nothing matches.
    """
    if isinstance(value, list):
        for v in value:
            if allowed is None or v in allowed:
                return v
        return ""
    return value if (allowed is None or value in allowed) else ""


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi].

    Args:
        angle: The angle in radians.

    Returns:
        float: The normalized angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def gps_to_vector(coord1: GpsCoord, coord2: GpsCoord) -> tuple[float, float]:
    """Convert two GPS coordinates to a displacement vector (x, y) in meters.

    Uses local tangent plane approximation (flat-Earth model).
    Valid for distances < ~100 km with <0.5% error.

    Args:
        coord1: Origin coordinates.
        coord2: Target coordinates.

    Returns:
        tuple[float, float]: (x, y) where:
            - x = East displacement (meters)
            - y = North displacement (meters)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6_371_000  # mean R of Earth

    lat1_rad = math.radians(lat1)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = normalize_angle(math.radians(lon2 - lon1))

    y = R * delta_lat
    x = R * delta_lon * math.cos(lat1_rad)

    return x, y


def coords_distance(coord1: GpsCoord, coord2: GpsCoord) -> float:
    """Finds a distance between two GPS coordinates in meters.

    Args:
        coord1: Origin coordinates.
        coord2: Target coordinates.

    Returns:
        float: The distance in meters.
    """
    return math.hypot(*gps_to_vector(coord1, coord2))


T = TypeVar("T", float, NDArray[np.float64])


def point_segment_distance(
    Px: T, Py: T, S1x: T, S1y: T, S2x: T, S2y: T
) -> tuple[T, T, T, T]:
    """Vectorized point-to-segment distance algorithm.

    Args:
        Px: Point X coordinate.
        Py: Point Y coordinate.
        S1x: Segment Start X.
        S1y: Segment Start Y.
        S2x: Segment End X.
        S2y: Segment End Y.

    Returns:
        tuple[T, T, T, T]: (distance, vector_u_x, vector_u_y, t_projection_scalar).
    """
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


def segments_intersect(
    A_x: T,
    A_y: T,
    B_x: T,
    B_y: T,
    C_x: T,
    C_y: T,
    D_x: T,
    D_y: T,
) -> np.bool_ | NDArray[np.bool_]:
    """Check if line segments AB and CD intersect.

    Vectorized operation supporting both scalar and array inputs.
    Uses the cross-product straddle test: segments intersect if they
    straddle each other's lines (endpoints on opposite sides).

    Args:
        A_x: Segment AB start X coordinate.
        A_y: Segment AB start Y coordinate.
        B_x: Segment AB end X coordinate.
        B_y: Segment AB end Y coordinate.
        C_x: Segment CD start X coordinate.
        C_y: Segment CD start Y coordinate.
        D_x: Segment CD end X coordinate.
        D_y: Segment CD end Y coordinate.

    Returns:
        np.bool_ | NDArray[np.bool_]: True where segments AB and CD intersect.
            Returns numpy bool scalar for scalar inputs, array for array inputs.
    """
    CD_x, CD_y = D_x - C_x, D_y - C_y
    cp1 = CD_x * (A_y - C_y) - CD_y * (A_x - C_x)
    cp2 = CD_x * (B_y - C_y) - CD_y * (B_x - C_x)
    diff_side_CD = (cp1 * cp2) <= 0.0

    AB_x, AB_y = B_x - A_x, B_y - A_y
    cp3 = AB_x * (C_y - A_y) - AB_y * (C_x - A_x)
    cp4 = AB_x * (D_y - A_y) - AB_y * (D_x - A_x)
    diff_side_AB = (cp3 * cp4) <= 0.0

    return np.logical_and(diff_side_CD, diff_side_AB)


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion to a yaw angle (Z-axis rotation).

    Args:
        qx: Quaternion X component.
        qy: Quaternion Y component.
        qz: Quaternion Z component.
        qw: Quaternion W component (scalar part).

    Returns:
        float: Yaw angle in radians (rotation about Z-axis, intrinsic rotation
            sequence ZYX).
    """
    return Rotation.from_quat([qx, qy, qz, qw]).as_euler("zyx")[0]


def yaw_to_quat(yaw: float) -> NDArray[np.float64]:
    """Convert a yaw angle (Z-axis rotation) to a quaternion.

    Args:
        yaw: Yaw angle in radians.

    Returns:
        NDArray[np.float64]: Unit quaternion [qx, qy, qz, qw] representing
            the rotation.
    """
    return Rotation.from_euler("z", yaw).as_quat()
