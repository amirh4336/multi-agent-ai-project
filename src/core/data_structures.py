from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any

class Action(Enum):
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    PICKUP = "pickup"
    DROP = "drop"
    WAIT = "wait"
    COMMUNICATE = "communicate"

class CellType(Enum):
    EMPTY = "empty"
    WALL = "wall"
    GOAL = "goal"
    RESOURCE = "resource"
    HAZARD = "hazard"

@dataclass
class Position:
    x: int
    y: int
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Position(x={self.x}, y={self.y})"
    
    def distance_to(self, other: 'Position') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def move(self, dx: int, dy: int) -> 'Position':
        return Position(self.x + dx, self.y + dy)

@dataclass
class Perception:
    current_position: Position
    visible_cells: Dict[Position, CellType]  
    visible_agents: List[Position]
    energy_level: int
    carrying_resource: bool
    messages: List[str]
    
    def get_adjacent_positions(self) -> List[Position]:
        """Get all adjacent positions to current position."""
        adjacent = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adjacent.append(self.current_position.move(dx, dy))
        return adjacent
    
    def get_cells_of_type(self, cell_type: CellType) -> List[Position]:
        """Get all visible positions containing specified cell type."""
        return [pos for pos, ctype in self.visible_cells.items() 
                if ctype == cell_type]

@dataclass
class PlanStep:
    """Represents a single step in an agent's plan."""
    action: Action
    target_position: Position
    purpose: str
    estimated_cost: float
    
    def __str__(self):
        return f"{self.action.value} -> {self.target_position} ({self.purpose})"

@dataclass
class AgentState:
    """Complete state information for an agent."""
    agent_id: str
    position: Position
    energy: int
    carrying_resource: bool
    last_action: Optional[Action] = None
    
    def is_active(self) -> bool:
        """Check if agent has energy to perform actions."""
        return self.energy > 0

class MessageType(Enum):
    """Types of inter-agent communication messages."""
    RESOURCE_FOUND = "resource_found"
    GOAL_LOCATION = "goal_location"
    HAZARD_WARNING = "hazard_warning"
    REQUEST_HELP = "request_help"
    COORDINATION = "coordination"

@dataclass
class Message:
    """Inter-agent communication message."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: int
    
    def is_broadcast(self) -> bool:
        """Check if message is broadcast to all agents."""
        return self.receiver_id is None

# Environment configuration data structures
@dataclass
class EnvironmentConfig:
    """Configuration parameters for environment creation."""
    width: int
    height: int
    num_resources: int
    num_goals: int
    num_hazards: int
    max_steps: int
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.width > 0 and self.height > 0 and 
                self.num_resources >= 0 and self.num_goals >= 0 and
                self.num_hazards >= 0 and self.max_steps > 0)

# Performance metrics data structures
@dataclass
class PerformanceMetrics:
    """Performance metrics for agent evaluation."""
    success_rate: float
    efficiency_score: float
    task_completion_time: float
    energy_utilization: float
    collision_frequency: int
    exploration_coverage: float
    
    def __str__(self):
        return (f"Performance Metrics:\n"
                f"  Success Rate: {self.success_rate:.2%}\n"
                f"  Efficiency Score: {self.efficiency_score:.2f}\n"
                f"  Completion Time: {self.task_completion_time:.1f}\n"
                f"  Energy Utilization: {self.energy_utilization:.1f}\n"
                f"  Collisions: {self.collision_frequency}\n"
                f"  Exploration Coverage: {self.exploration_coverage:.2%}")

# Constants
PERCEPTION_RANGE = 2  # 5x5 grid (2 cells in each direction)
INITIAL_ENERGY = 100
ENERGY_COST_PER_ACTION = 1
ENERGY_COST_PER_MOVE = 1
ENERGY_COST_HAZARD = 5

# Direction mappings for movement actions
DIRECTION_MAPPINGS = {
    Action.MOVE_NORTH: (0, -1),
    Action.MOVE_SOUTH: (0, 1),
    Action.MOVE_EAST: (1, 0),
    Action.MOVE_WEST: (-1, 0)
}

# Utility values for goal-based agent
UTILITY_VALUES = {
    'resource_collection': 10.0,
    'resource_delivery': 20.0,
    'exploration': 1.0,
    'hazard_avoidance': 50.0
}