"""
Simple Reflex Agent Implementation.

This module implements the SimpleReflexAgent class following Russell & Norvig's
framework with condition-action rules and priority-based decision making.
"""

import random
from typing import Tuple, Optional, List
from ..core.base_agent import BaseAgent
from ..core.data_structures import (
    Position, Action, Perception, CellType, 
    DIRECTION_MAPPINGS
)

class SimpleReflexAgent(BaseAgent):
    """
    Simple Reflex Agent implementation following Russell & Norvig's framework.
    
    Uses condition-action rules with priority system:
    1. Hazard Avoidance (Priority 1)
    2. Resource Collection (Priority 2) 
    3. Goal Seeking (Priority 3)
    4. Resource Pursuit (Priority 4)
    5. Random Exploration (Priority 5)
    
    This agent responds immediately to perceptual inputs using predefined
    behavioral rules without maintaining internal state.
    """
    
    def __init__(self, agent_id: str, position: Position):
        """
        Initialize Simple Reflex Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial position in the environment
        """
        super().__init__(agent_id, position)
        
        # Statistics for rule activation analysis
        self.rule_activation_count = {
            'hazard_avoidance': 0,
            'resource_collection': 0,
            'goal_seeking': 0,
            'resource_pursuit': 0,
            'random_exploration': 0
        }
    
    def perceive(self, environment) -> Perception:
        """
        Extract perception data from environment.
        
        Args:
            environment: The environment object
            
        Returns:
            Perception object with current environmental state
        """
        perception = environment.get_agent_perception(self.agent_id)
        # Ensure perception reflects current agent state
        perception.carrying_resource = self.carrying_resource
        return perception
    
    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """
        Main decision-making method using priority-based rule system.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (Action, reason_string)
        """
        
        # Priority 1: Hazard Avoidance
        if self._is_in_hazard(perception):
            action, reason = self._avoid_hazard(perception)
            if action:
                self.rule_activation_count['hazard_avoidance'] += 1
                return action, f"Hazard Avoidance: {reason}"
        
        # Priority 2: Resource Collection
        if self._can_collect_resource(perception):
            self.rule_activation_count['resource_collection'] += 1
            return Action.PICKUP, "Resource Collection: Picking up adjacent resource"
        
        # Priority 3: Goal Seeking
        if perception.carrying_resource:
            if self._can_drop_resource(perception):
                self.rule_activation_count['goal_seeking'] += 1
                return Action.DROP, "Goal Seeking: Dropping resource at goal"
            else:
                action, reason = self._seek_goal(perception)
                if action:
                    self.rule_activation_count['goal_seeking'] += 1
                    return action, f"Goal Seeking: {reason}"
        
        # Priority 4: Resource Pursuit
        if not perception.carrying_resource:
            action, reason = self._pursue_resource(perception)
            if action:
                self.rule_activation_count['resource_pursuit'] += 1
                return action, f"Resource Pursuit: {reason}"
        
        # Priority 5: Random Exploration
        action, reason = self._random_exploration(perception)
        self.rule_activation_count['random_exploration'] += 1
        return action, f"Random Exploration: {reason}"
    
    def _is_in_hazard(self, perception: Perception) -> bool:
        """
        Check if current position contains a hazard.
        
        Args:
            perception: Current perception data
            
        Returns:
            True if current position is hazardous
        """
        current_cell = perception.visible_cells.get(perception.current_position)
        return current_cell == CellType.HAZARD
    
    def _avoid_hazard(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """
        Find safe adjacent cell to move to.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (safe_move_action, reason_string)
        """
        safe_moves = []
        
        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(
                perception.current_position.x + dx,
                perception.current_position.y + dy
            )
            
            cell_type = perception.visible_cells.get(new_pos)
            if cell_type and cell_type not in [CellType.WALL, CellType.HAZARD]:
                # Check if position is not occupied by another agent
                if new_pos not in perception.visible_agents:
                    safe_moves.append(action)
        
        if safe_moves:
            return random.choice(safe_moves), "Moving to safe adjacent cell"
        else:
            return Action.WAIT, "No safe moves available, waiting"
    
    def _can_collect_resource(self, perception: Perception) -> bool:
        """
        Check if there's a resource in adjacent cell and agent isn't carrying one.
        
        Args:
            perception: Current perception data
            
        Returns:
            True if can collect an adjacent resource
        """
        print("perception.carrying_resource: ", perception.carrying_resource)
        if perception.carrying_resource:
            return False

        current_cell = perception.visible_cells.get(perception.current_position)
        return current_cell == CellType.RESOURCE
    
    def _can_drop_resource(self, perception: Perception) -> bool:
        """
        Check if there's a goal in adjacent cell and agent is carrying resource.
        
        Args:
            perception: Current perception data
            
        Returns:
            True if can drop an adjacent resource
        """
        current_cell = perception.visible_cells.get(perception.current_position)
        return current_cell == CellType.GOAL
    
    def _seek_goal(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """
        Move toward visible goal when carrying resource.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (move_action, reason_string)
        """
        goal_positions = perception.get_cells_of_type(CellType.GOAL)
        
        if not goal_positions:
            return None, "No goals visible"
        
        # Find closest goal
        closest_goal = min(goal_positions, 
                        key=lambda pos: perception.current_position.distance_to(pos))
        
        # Move toward closest goal
        move_action = self._get_move_toward(perception.current_position, closest_goal, perception)
        if move_action:
            return move_action, f"Moving toward goal at {closest_goal}"
        else:
            return Action.WAIT, "Cannot move toward goal, path blocked"
    
    def _pursue_resource(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """
        Move toward visible resource when not carrying one.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (move_action, reason_string)
        """
        resource_positions = perception.get_cells_of_type(CellType.RESOURCE)
        
        if not resource_positions:
            return None, "No resources visible"
        
        # Find closest resource
        closest_resource = min(resource_positions,
                            key=lambda pos: perception.current_position.distance_to(pos))
        
        # Move toward closest resource
        move_action = self._get_move_toward(perception.current_position, closest_resource, perception)
        if move_action:
            return move_action, f"Moving toward resource at {closest_resource}"
        else:
            return Action.WAIT, "Cannot move toward resource, path blocked"
    
    def _random_exploration(self, perception: Perception) -> Tuple[Action, str]:
        """
        Move randomly to valid adjacent cell.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (move_action, reason_string)
        """
        valid_moves = []
        
        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(
                perception.current_position.x + dx,
                perception.current_position.y + dy
            )
            
            cell_type = perception.visible_cells.get(new_pos)
            if cell_type and cell_type != CellType.WALL and cell_type != CellType.HAZARD:
                # Check if position is not occupied by another agent
                if new_pos not in perception.visible_agents:
                    valid_moves.append(action)
        
        if valid_moves:
            return random.choice(valid_moves), "Moving randomly to explore"
        else:
            return Action.WAIT, "No valid moves available, waiting"
    
    def _get_move_toward(self, from_pos: Position, to_pos: Position, 
                    perception: Perception) -> Optional[Action]:
        """
        Get the best move action to approach target position.

        Args:
            from_pos: Current position
            to_pos: Target position
            perception: Current perception data

        Returns:
            Best movement action or None if no valid moves
        """
        best_action = None
        min_distance = float('inf')

        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(from_pos.x + dx, from_pos.y + dy)
            if new_pos not in perception.visible_cells:
                continue

            cell_type = perception.visible_cells[new_pos]
            if cell_type in [CellType.WALL, CellType.HAZARD]:
                continue
            if new_pos in perception.visible_agents:
                continue

            distance = new_pos.distance_to(to_pos)
            if distance < min_distance:
                min_distance = distance
                best_action = action

        return best_action