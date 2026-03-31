import time
import json
from typing import Any


class PerformanceTracker:
    """Context manager for automatically tracking and logging performance metrics."""

    def __init__(
        self,
        logger: Any = None,
        log_level: str = "debug",
        throttle_sec: float | None = None,
        log_prefix: str = "Performance",
    ) -> None:
        """
        Initialize performance tracker.

        Args:
            logger: Logger instance with debug/info/warn methods
            log_level: Log level to use ('debug', 'info', 'warn')
            throttle_sec: Throttle duration in seconds (for throttled logging)
            log_prefix: Prefix for the log message
        """
        self.performance: dict[str, float] = {}
        self.last_time: float | None = None
        self.start_time: float | None = None
        self.logger = logger
        self.log_level = log_level
        self.throttle_sec = throttle_sec
        self.log_prefix = log_prefix

    def __enter__(self) -> "PerformanceTracker":
        """Enter context and initialize timing."""
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Exit context and automatically log performance."""
        if "Total" not in self.performance and self.start_time is not None:
            self.performance["Total"] = round(
                (time.perf_counter() - self.start_time) * 1000, 2
            )

        self._log_performance()

        return False

    def update(self, label: str) -> None:
        """
        Record a checkpoint with automatic time calculation from last checkpoint.

        Args:
            label: Name for this performance checkpoint
        """
        current_time = time.perf_counter()
        if self.last_time is not None:
            self.performance[label] = round((current_time - self.last_time) * 1000, 2)
        self.last_time = current_time

    def set_total(self) -> None:
        """Explicitly set the total time (useful if you want total before end)."""
        if self.start_time is not None:
            self.performance["Total"] = round(
                (time.perf_counter() - self.start_time) * 1000, 2
            )

    def _log_performance(self) -> None:
        """Log the collected performance metrics."""
        message = f"{self.log_prefix}:\n{json.dumps(self.performance, indent=4)}"

        if self.logger:
            log_func = getattr(self.logger, self.log_level, self.logger.debug)

            if self.throttle_sec is not None:
                log_func(message, throttle_duration_sec=self.throttle_sec)
            else:
                log_func(message)
        else:
            print(message)
