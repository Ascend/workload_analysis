from enum import Enum


class AICoreMetric(Enum):
    # 对照应用开发指南
    ARITHMETIC_UTILIZATION = 0
    PIPE_UTILIZATION = 1
    MEMORY_BANDWIDTH = 2
    L0B_AND_WIDTH = 3
    RESOURCE_CONFLICT_RATIO = 4
    MEMORY_UB = 5
    COMBINE = 6
