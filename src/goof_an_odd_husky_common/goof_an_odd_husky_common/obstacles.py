from dataclasses import dataclass
from abc import ABC

from goof_an_odd_husky_msgs.msg import ObstacleArray
from goof_an_odd_husky_msgs.msg import ObstacleCircle as ObstacleCircleMsg
from goof_an_odd_husky_msgs.msg import ObstacleLine as ObstacleLineMsg


class Obstacle(ABC):
    pass


@dataclass
class CircleObstacle(Obstacle):
    x: float
    y: float
    radius: float


@dataclass
class LineObstacle(Obstacle):
    x1: float
    y1: float
    x2: float
    y2: float


def obstacles_to_msg(obstacles: list[Obstacle]) -> ObstacleArray:
    msg = ObstacleArray()
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            msg.circles.append(ObstacleCircleMsg(x=obs.x, y=obs.y, radius=obs.radius))
        elif isinstance(obs, LineObstacle):
            msg.lines.append(
                ObstacleLineMsg(x1=obs.x1, y1=obs.y1, x2=obs.x2, y2=obs.y2)
            )
    return msg


def msg_to_obstacles(msg: ObstacleArray) -> list[Obstacle]:
    return [CircleObstacle(x=c.x, y=c.y, radius=c.radius) for c in msg.circles] + [
        LineObstacle(x1=l.x1, y1=l.y1, x2=l.x2, y2=l.y2) for l in msg.lines
    ]
