"""
Model-Based Reflex Agent Implementation.

This module implements the ModelBasedReflexAgent class that maintains internal
representations of environment state for more sophisticated decision-making.
"""

import random
from typing import Tuple, Optional, Set, Dict
from ..core.base_agent import BaseAgent
from ..core.data_structures import (
    Position, Action, Perception, CellType, 
    DIRECTION_MAPPINGS
)

class ModelBasedReflexAgent(BaseAgent):
    """
    Model-Based Reflex Agent implementation following Russell & Norvig's framework.
    
    Maintains internal models for:
    - Spatial Memory: Set of visited positions
    - Resource Tracking: Known resource locations
    - Goal Awareness: Known goal locations
    - Hazard Mapping: Known hazard locations
    - Environment Layout: Wall positions and navigable spaces
    
    Decision hierarchy:
    1. Emergency Response (Hazard avoidance)
    2. Opportunistic Collection (Adjacent resource pickup)
    3. Strategic Goal Completion (Navigate to known goals)
    4. Informed Resource Acquisition (Move toward known resources)
    5. Intelligent Exploration (Systematic exploration of unvisited areas)
    """
    
    def __init__(self, agent_id: str, position: Position):
        """
        Initialize Model-Based Reflex Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial position in the environment
        """
        super().__init__(agent_id, position)
        
        # Internal world model
        self.visited_positions: Set[Position] = set()
        self.known_resources: Set[Position] = set()
        self.known_goals: Set[Position] = set()
        self.known_hazards: Set[Position] = set()
        self.known_walls: Set[Position] = set()
        self.known_empty: Set[Position] = set()
        
        # Decision-making statistics
        self.decision_reasons = {
            'emergency_response': 0,
            'opportunistic_collection': 0,
            'strategic_goal_completion': 0,
            'informed_resource_acquisition': 0,
            'intelligent_exploration': 0
        }
        
        # Add initial position to visited
        self.visited_positions.add(position)
    
    def perceive(self, environment) -> Perception:
        """
        Extract perception data and update internal world model.
        
        Args:
            environment: The environment object
            
        Returns:
            Perception object with current environmental state
        """
        perception = environment.get_agent_perception(self.agent_id)
        perception.carrying_resource = self.carrying_resource
        
        # Update world model with new perception data
        self.update_world_model(perception)
        
        return perception
    
    def update_world_model(self, perception: Perception):
        """
        Update internal world model with perception data.
        
        Args:
            perception: Current perception data
        """
        # Add current position to visited
        self.visited_positions.add(perception.current_position)
        # Update knowledge about visible cells
        for position, cell_type in perception.visible_cells.items():
            if cell_type == CellType.WALL:
                self.known_walls.add(position)
            elif cell_type == CellType.RESOURCE:
                self.known_resources.add(position)
                # Remove from other sets if it was there
                self.known_empty.discard(position)
            elif cell_type == CellType.GOAL:
                self.known_goals.add(position)
                self.known_empty.discard(position)
            elif cell_type == CellType.HAZARD:
                self.known_hazards.add(position)
                self.known_empty.discard(position)
            elif cell_type == CellType.EMPTY:
                self.known_empty.add(position)
                # Remove from resource/goal sets if resource was collected/goal was reached
                self.known_resources.discard(position)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """
        Main decision-making method using internal model and priority system.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (Action, reason_string)
        """
        
        # Priority 1: Emergency Response (Hazard avoidance)
        if self._is_in_hazard(perception):
            self.energy = max(0, self.energy - 4)  # Hazard energy penalty
            action, reason = self._emergency_hazard_response(perception)
            if action:
                self.decision_reasons['emergency_response'] += 1
                return action, f"Emergency Response: {reason}"
        
        # Priority 2: Opportunistic Collection (Adjacent resource pickup)
        if self._can_collect_resource(perception):
            self.decision_reasons['opportunistic_collection'] += 1
            return Action.PICKUP, "Opportunistic Collection: Picking up adjacent resource"
        
        # Priority 3: Strategic Goal Completion (Navigate to known goals when carrying)
        if perception.carrying_resource:
            if self._can_drop_resource(perception):
                self.decision_reasons['strategic_goal_completion'] += 1
                return Action.DROP, "Strategic Goal Completion: Dropping resource at goal"
            else:
                action, reason = self._strategic_goal_navigation(perception)
                if action:
                    self.decision_reasons['strategic_goal_completion'] += 1
                    return action, f"Strategic Goal Completion: {reason}"
        
        # Priority 4: Informed Resource Acquisition (Move toward known resources)
        if not perception.carrying_resource:
            action, reason = self._informed_resource_acquisition(perception)
            if action:
                self.decision_reasons['informed_resource_acquisition'] += 1
                return action, f"Informed Resource Acquisition: {reason}"
        
        # Priority 5: Intelligent Exploration (Systematic exploration)
        action, reason = self._intelligent_exploration(perception)
        self.decision_reasons['intelligent_exploration'] += 1
        return action, f"Intelligent Exploration: {reason}"
    
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
    
    def _emergency_hazard_response(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """
        Use spatial memory to find safest escape route from hazard.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (safe_move_action, reason_string)
        """
        safe_moves = []
        preferred_moves = []
        
        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(
                perception.current_position.x + dx,
                perception.current_position.y + dy
            )
            
            # Check if position is valid and safe
            cell_type = perception.visible_cells.get(new_pos)
            if cell_type and cell_type not in [CellType.WALL, CellType.HAZARD]:
                if new_pos not in perception.visible_agents:
                    safe_moves.append(action)
                    
                    # Prefer previously visited safe positions
                    if not new_pos in self.visited_positions and new_pos not in self.known_hazards:
                        if not perception.carrying_resource:
                            action, reason = self._informed_resource_acquisition(perception)
                            preferred_moves.append(action)
                        else :
                            action, reason = self._strategic_goal_navigation(perception)
                            preferred_moves.append(action)
        print(preferred_moves)
        if preferred_moves:
            print(preferred_moves)
            return random.choice(preferred_moves), "Moving to base on goal or resource safe position"
        elif safe_moves:
            return random.choice(safe_moves), "Moving to safe adjacent cell"
        else:
            return Action.WAIT, "No safe escape route available"
    
    def _can_collect_resource(self, perception: Perception) -> bool:
        """
        Check if there's a resource at current position and agent isn't carrying one.
        
        Args:
            perception: Current perception data
            
        Returns:
            True if can collect resource at current position
        """
        if perception.carrying_resource:
            return False
        
        current_cell = perception.visible_cells.get(perception.current_position)
        return current_cell == CellType.RESOURCE
    
    def _can_drop_resource(self, perception: Perception) -> bool:
        """
        Check if there's a goal at current position and agent is carrying resource.
        
        Args:
            perception: Current perception data
            
        Returns:
            True if can drop resource at current position
        """
        current_cell = perception.visible_cells.get(perception.current_position)
        return current_cell == CellType.GOAL
    
    def _strategic_goal_navigation(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """
        Navigate to known goal locations using internal model.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (move_action, reason_string)
        """
        # First check visible goals
        visible_goals = perception.get_cells_of_type(CellType.GOAL)
        if visible_goals:
            closest_goal = min(visible_goals,
                             key=lambda pos: perception.current_position.distance_to(pos))
            move_action = self._get_smart_move_toward(perception.current_position, 
                                                   closest_goal, perception)
            if move_action:
                return move_action, f"Moving toward visible goal at {closest_goal}"
        
        # Use known goals from internal model
        if self.known_goals:
            closest_known_goal = min(self.known_goals,
                                   key=lambda pos: perception.current_position.distance_to(pos))
            move_action = self._get_smart_move_toward(perception.current_position,
                                                   closest_known_goal, perception)
            if move_action:
                return move_action, f"Moving toward known goal at {closest_known_goal}"
        
        return None, "No goals known or reachable"
    
    def _informed_resource_acquisition(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """
        Move toward known resource locations using internal model.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (move_action, reason_string)
        """
        # First check visible resources
        visible_resources = perception.get_cells_of_type(CellType.RESOURCE)
        if visible_resources:
            closest_resource = min(visible_resources,
                                    key=lambda pos: perception.current_position.distance_to(pos))
            print("closeet", closest_resource)
            move_action = self._get_smart_move_toward(perception.current_position,
                                                    closest_resource, perception)
            if move_action:
                return move_action, f"Moving toward visible resource at {closest_resource}"
        
        # Use known resources from internal model
        if self.known_resources:
            closest_known_resource = min(self.known_resources,
                                        key=lambda pos: perception.current_position.distance_to(pos))
            move_action = self._get_smart_move_toward(perception.current_position,
                                                    closest_known_resource, perception)
            if move_action:
                return move_action, f"Moving toward known resource at {closest_known_resource}"
        
        return None, "No resources known or reachable"
    
    def _intelligent_exploration(self, perception: Perception) -> Tuple[Action, str]:
        """
        Systematic exploration prioritizing unvisited areas.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (move_action, reason_string)
        """
        exploration_moves = []
        unvisited_moves = []
        
        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(
                perception.current_position.x + dx,
                perception.current_position.y + dy
            )
            
            # Check if position is valid
            cell_type = perception.visible_cells.get(new_pos)
            if cell_type and cell_type != CellType.WALL:
                if new_pos not in perception.visible_agents:
                    # Avoid known hazards unless no other choice
                    if new_pos not in self.known_hazards:
                        exploration_moves.append(action)
                        
                        # Prioritize unvisited positions
                        if new_pos not in self.visited_positions:
                            unvisited_moves.append(action)
        
        # Prefer unvisited positions for exploration
        if unvisited_moves:
            chosen_action = random.choice(unvisited_moves)
            return chosen_action, "Exploring unvisited area"
        elif exploration_moves:
            chosen_action = random.choice(exploration_moves)
            return chosen_action, "Exploring safe area"
        else:
            return Action.WAIT, "No safe exploration options available"
    
    def _get_smart_move_toward(self, from_pos: Position, to_pos: Position,
                            perception: Perception) -> Optional[Action]:
        """
        Get intelligent move toward target using internal model knowledge.
        
        Args:
            from_pos: Current position
            to_pos: Target position
            perception: Current perception data

        Returns:
            Best movement action or None if no valid moves
        """

        def is_valid_position(pos: Position) -> bool:
            if pos.x < 0 or pos.y < 0:
                return False
            if pos in self.known_walls:
                return False
            if perception.visible_cells.get(pos) == CellType.WALL:
                return False
            if pos in perception.visible_agents:
                return False
            return True

        def has_safe_alternative(excluded_pos: Position) -> bool:
            for dx, dy in DIRECTION_MAPPINGS.values():
                alt_pos = Position(from_pos.x + dx, from_pos.y + dy)
                if alt_pos == excluded_pos:
                    continue
                if alt_pos in self.known_hazards:
                    continue
                if is_valid_position(alt_pos):
                    return True
            return False

        def move_cost(pos: Position) -> int:
            """
            Estimate cost of moving into this position based on what agent knows.
            Lower is better.
            """
            if pos in self.known_walls:
                return float('inf')  # Can't go
            if pos in self.known_hazards:
                return 10  # Risky
            if pos in self.known_resources:
                return 1  # Might be good
            if pos in self.known_empty or pos in self.visited_positions:
                return 2
            if pos in self.known_goals:
                return 1  # Desirable
            return 3  # Unknown or neutral

        best_action = None
        lowest_total_cost = float('inf')

        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(from_pos.x + dx, from_pos.y + dy)

            if not is_valid_position(new_pos):
                continue

            if new_pos in self.known_hazards and has_safe_alternative(new_pos):
                continue

            # Total cost = move cost + remaining estimated distance
            total_cost = move_cost(new_pos) + new_pos.distance_to(to_pos)

            if total_cost < lowest_total_cost:
                lowest_total_cost = total_cost
                best_action = action

        return best_action


    
    def get_model_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the internal world model.
        
        Returns:
            Dictionary with model statistics
        """
        return {
            'visited_positions': len(self.visited_positions),
            'known_resources': len(self.known_resources),
            'known_goals': len(self.known_goals),
            'known_hazards': len(self.known_hazards),
            'known_walls': len(self.known_walls),
            'known_empty': len(self.known_empty),
            'total_known_cells': len(self.visited_positions) + len(self.known_walls) + 
                                len(self.known_resources) + len(self.known_goals) + 
                                len(self.known_hazards) + len(self.known_empty)
        }
    
    def get_decision_statistics(self) -> Dict[str, int]:
        """
        Get statistics about decision-making patterns.
        
        Returns:
            Dictionary with decision statistics
        """
        return self.decision_reasons.copy()
    
    def reset_model(self):
        """Reset the internal world model."""
        self.visited_positions.clear()
        self.known_resources.clear()
        self.known_goals.clear()
        self.known_hazards.clear()
        self.known_walls.clear()
        self.known_empty.clear()
        self.visited_positions.add(self.position)
        
        # Reset decision statistics
        for key in self.decision_reasons:
            self.decision_reasons[key] = 0