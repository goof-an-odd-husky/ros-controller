from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)
