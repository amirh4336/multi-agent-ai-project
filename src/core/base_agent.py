"""
Base Agent class for the Multi-Agent AI System.

This module provides the abstract base class that all agent implementations
must inherit from, ensuring consistent interface and behavior.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from .data_structures import Position, Action, Perception, AgentState, PerformanceMetrics

class BaseAgent(ABC):
    """
    Abstract base class for all agent implementations.
    
    This class defines the common interface and shared functionality
    that all agent types must implement.
    """
    
    def __init__(self, agent_id: str, position: Position , initial_energy :int):
        """
        Initialize base agent with essential properties.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial position in the environment
        """
        self.agent_id = agent_id
        self.position = position
        self.energy = initial_energy
        self.carrying_resource = False
        self.last_action: Optional[Action] = None
        
        # Performance tracking
        self.actions_taken = 0
        self.resources_collected = 0
        self.goals_reached = 0
        self.energy_consumed = 0
        self.collisions = 0
        
        # Statistics for analysis
        self.decision_times = []
        self.action_history = []
    
    @abstractmethod
    def perceive(self, environment) -> Perception:
        """
        Extract perception data from the environment.
        
        Args:
            environment: The environment object
            
        Returns:
            Perception object containing relevant environmental information
        """
        pass
    
    @abstractmethod
    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """
        Decide what action to take based on perception.
        
        Args:
            perception: Current perception of the environment
            
        Returns:
            Tuple of (action_to_take, reason_for_action)
        """
        pass
    
    def execute_action(self, action: Action) -> bool:
        """
        Execute the given action and update agent state.
        
        Args:
            action: The action to execute
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        # Update statistics
        self.actions_taken += 1
        self.last_action = action
        self.action_history.append(action)
        
        # Consume energy
        energy_cost = self._get_action_energy_cost(action)
        self.energy = max(0, self.energy - energy_cost)
        self.energy_consumed += energy_cost
        
        return self.energy > 0
    
    def _get_action_energy_cost(self, action: Action) -> int:
        """
        Calculate energy cost for performing an action.
        
        Args:
            action: The action to calculate cost for
            
        Returns:
            Energy cost as integer
        """
        if action in [Action.MOVE_NORTH, Action.MOVE_SOUTH, 
                    Action.MOVE_EAST, Action.MOVE_WEST]:
            return 1
        elif action in [Action.PICKUP, Action.DROP]:
            return 1
        elif action == Action.WAIT:
            return 0
        elif action == Action.COMMUNICATE:
            return 1
        else:
            return 1
    
    def update_position(self, new_position: Position):
        """Update agent's position after successful movement."""
        self.position = new_position
    
    def pickup_resource(self) -> bool:
        """
        Pick up a resource if possible.
        
        Returns:
            True if resource was picked up successfully
        """
        if not self.carrying_resource:
            self.carrying_resource = True
            self.resources_collected += 1
            return True
        return False
    
    def drop_resource(self) -> bool:
        """
        Drop the carried resource if any.
        
        Returns:
            True if resource was dropped successfully
        """
        if self.carrying_resource:
            self.carrying_resource = False
            return True
        return False
    
    def reach_goal(self):
        """Mark that agent has reached a goal."""
        self.goals_reached += 1
        if self.carrying_resource:
            self.carrying_resource = False
            return True
        return False
    
    def record_collision(self):
        """Record a collision event for statistics."""
        self.collisions += 1
    
    def get_state(self) -> AgentState:
        """
        Get current agent state.
        
        Returns:
            AgentState object with current status
        """
        return AgentState(
            agent_id=self.agent_id,
            position=self.position,
            energy=self.energy,
            carrying_resource=self.carrying_resource,
            last_action=self.last_action
        )
    
    def is_active(self) -> bool:
        """Check if agent has enough energy to continue acting."""
        return self.energy > 0
    
    def get_performance_metrics(self, total_steps: int, 
                            explored_cells: int, 
                            total_cells: int) -> PerformanceMetrics:
        """
        Calculate performance metrics for this agent.
        
        Args:
            total_steps: Total steps in the simulation
            explored_cells: Number of cells explored by agent
            total_cells: Total cells in environment
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate success rate
        success_rate = self.goals_reached / max(1, self.resources_collected)
        
        # Calculate efficiency score (composite metric)
        efficiency_score = self._calculate_efficiency_score(total_steps)
        
        # Calculate task completion time
        completion_time = self.actions_taken / max(1, self.goals_reached)
        
        # Calculate energy utilization
        energy_utilization = 100 - self.energy
        
        # Calculate exploration coverage
        exploration_coverage = explored_cells / total_cells
        
        return PerformanceMetrics(
            success_rate=success_rate,
            efficiency_score=efficiency_score,
            task_completion_time=completion_time,
            energy_utilization=energy_utilization,
            collision_frequency=self.collisions,
            exploration_coverage=exploration_coverage
        )
    
    def _calculate_efficiency_score(self, total_steps: int) -> float:
        """
        Calculate composite efficiency score.
        
        Args:
            total_steps: Total simulation steps
            
        Returns:
            Efficiency score as float
        """
        # Efficiency components
        resource_efficiency = self.resources_collected / max(1, self.actions_taken)
        goal_efficiency = self.goals_reached / max(1, self.actions_taken)
        energy_efficiency = (100 - self.energy_consumed) / 100
        collision_penalty = max(0, 1 - (self.collisions * 0.1))
        
        # Weighted combination
        efficiency = (
            resource_efficiency * 0.3 +
            goal_efficiency * 0.4 +
            energy_efficiency * 0.2 +
            collision_penalty * 0.1
        )
        
        return efficiency * 100  # Scale to 0-100
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        self.actions_taken = 0
        self.resources_collected = 0
        self.goals_reached = 0
        self.energy_consumed = 0
        self.collisions = 0
        self.decision_times.clear()
        self.action_history.clear()
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get summary of agent statistics.
        
        Returns:
            Dictionary containing key statistics
        """
        return {
            'agent_id': self.agent_id,
            'actions_taken': self.actions_taken,
            'resources_collected': self.resources_collected,
            'goals_reached': self.goals_reached,
            'energy_remaining': self.energy,
            'energy_consumed': self.energy_consumed,
            'collisions': self.collisions,
            'carrying_resource': self.carrying_resource,
            'last_action': self.last_action.value if self.last_action else None
        }
    
    def __str__(self) -> str:
        """String representation of agent."""
        return (f"Agent {self.agent_id} at {self.position} "
                f"(Energy: {self.energy}, "
                f"Carrying: {self.carrying_resource})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}("
                f"agent_id='{self.agent_id}', "
                f"position={self.position}, "
                f"energy={self.energy}, "
                f"carrying_resource={self.carrying_resource})")