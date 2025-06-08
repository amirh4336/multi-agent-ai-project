"""
Goal-Based Agent Implementation.

This module implements the GoalBasedAgent class that incorporates deliberative
planning capabilities with A* path finding and utility-based goal selection.
"""

import heapq
import random
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass
from ..core.base_agent import BaseAgent
from ..core.data_structures import (
    Position, Action, Perception, CellType, PlanStep,
    DIRECTION_MAPPINGS, UTILITY_VALUES
)

@dataclass
class Goal:
    """Represents a goal with its utility and target position."""
    goal_type: str
    target_position: Position
    base_utility: float
    distance_adjusted_utility: float
    description: str

class GoalBasedAgent(BaseAgent):
    """
    Goal-Based Agent implementation with deliberative planning capabilities.
    
    Features:
    - Utility-based goal selection
    - A* path finding algorithm
    - Plan generation and execution
    - Dynamic replanning when plans become invalid
    - Adaptive response to changing conditions
    
    Goal Types and Base Utilities:
    - Resource Collection: 10.0
    - Resource Delivery: 20.0
    - Exploration: 1.0
    - Hazard Avoidance: 50.0
    
    Utility adjustment: final_utility = base_utility / (distance + 1)
    """
    
    def __init__(self, agent_id: str, position: Position):
        """
        Initialize Goal-Based Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            position: Initial position in the environment
        """
        super().__init__(agent_id, position)
        
        # Planning system
        self.current_plan: List[PlanStep] = []
        self.current_goal: Optional[Goal] = None
        self.plan_step_index: int = 0
        
        # World knowledge (similar to model-based agent)
        self.visited_positions: Set[Position] = set()
        self.known_resources: Set[Position] = set()
        self.known_goals: Set[Position] = set()
        self.known_hazards: Set[Position] = set()
        self.known_walls: Set[Position] = set()
        self.known_empty: Set[Position] = set()
        
        # Planning statistics
        self.planning_stats = {
            'plans_generated': 0,
            'plans_completed': 0,
            'plans_abandoned': 0,
            'replanning_events': 0,
            'path_finding_calls': 0,
            'path_finding_failures': 0
        }
        
        # Goal selection statistics
        self.goal_selection_stats = {
            'resource_collection_selected': 0,
            'resource_delivery_selected': 0,
            'exploration_selected': 0,
            'hazard_avoidance_selected': 0
        }
        
        # Add initial position to visited
        self.visited_positions.add(position)
    
    def perceive(self, environment) -> Perception:
        """
        Extract perception data and update world knowledge.
        
        Args:
            environment: The environment object
            
        Returns:
            Perception object with current environmental state
        """
        perception = environment.get_agent_perception(self.agent_id)
        perception.carrying_resource = self.carrying_resource
        
        # Update world knowledge
        self.update_world_knowledge(perception)
        
        return perception
    
    def update_world_knowledge(self, perception: Perception):
        """
        Update internal world knowledge with perception data.
        
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
                self.known_empty.discard(position)
            elif cell_type == CellType.GOAL:
                self.known_goals.add(position)
                self.known_empty.discard(position)
            elif cell_type == CellType.HAZARD:
                self.known_hazards.add(position)
                self.known_empty.discard(position)
            elif cell_type == CellType.EMPTY:
                self.known_empty.add(position)
                # Remove from resource/goal sets if they're no longer there
                self.known_resources.discard(position)
    
    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """
        Main decision-making method using planning and utility-based goal selection.
        
        Args:
            perception: Current perception data
            
        Returns:
            Tuple of (Action, reason_string)
        """
        # Check if current plan is still valid and executable
        if self._is_plan_valid(perception) and self.current_plan:
            action, reason = self._execute_current_plan(perception)
            if action:
                return action, f"Plan Execution: {reason}"
        
        # Generate new plan if current one is invalid or completed
        new_goal = self._select_best_goal(perception)
        if new_goal:
            new_plan = self._generate_plan(new_goal, perception)
            if new_plan:
                self.current_goal = new_goal
                self.current_plan = new_plan
                self.plan_step_index = 0
                self.planning_stats['plans_generated'] += 1
                
                # Execute first step of new plan
                action, reason = self._execute_current_plan(perception)
                if action:
                    return action, f"New Plan: {reason}"
        
        # Fallback to reactive behavior if planning fails
        return self._reactive_fallback(perception)
    
    def _select_best_goal(self, perception: Perception) -> Optional[Goal]:
        """
        Select the goal with highest utility value.
        
        Args:
            perception: Current perception data
            
        Returns:
            Goal with highest utility or None if no goals available
        """
        candidate_goals = []
        
        # Emergency hazard avoidance
        if self._is_in_hazard(perception):
            dict_positions = self._find_safe_positions(perception)
            
            safe_positions = dict_positions["safe_positions"]
            preferred_safe_moves = dict_positions["preferred_safe_moves"]

            # print(f"safe_positions: {safe_positions} , preferred_safe_moves: {preferred_safe_moves}")

            if preferred_safe_moves :
                closest_safe = min(preferred_safe_moves, 
                                key=lambda pos: perception.current_position.distance_to(pos))
                distance = perception.current_position.distance_to(closest_safe)
                utility = UTILITY_VALUES['hazard_avoidance'] / (distance + 1)
                candidate_goals.append(Goal(
                    goal_type='hazard_avoidance',
                    target_position=closest_safe,
                    base_utility=UTILITY_VALUES['hazard_avoidance'],
                    distance_adjusted_utility=utility,
                    description=f"Escape hazard to {closest_safe}"
                ))
            if safe_positions :
                closest_safe = min(safe_positions, 
                                key=lambda pos: perception.current_position.distance_to(pos))
                # closest_safe = random.choice(safe_positions)
                distance = perception.current_position.distance_to(closest_safe)
                utility = UTILITY_VALUES['hazard_avoidance'] / (distance + 1)
                candidate_goals.append(Goal(
                    goal_type='hazard_avoidance',
                    target_position=closest_safe,
                    base_utility=UTILITY_VALUES['hazard_avoidance'],
                    distance_adjusted_utility=utility,
                    description=f"Escape hazard to {closest_safe}"
                ))
        
        # Resource delivery (highest priority when carrying)
        if perception.carrying_resource:
            goal_positions = list(self.known_goals) + perception.get_cells_of_type(CellType.GOAL)
            for goal_pos in set(goal_positions):  # Remove duplicates
                distance = perception.current_position.distance_to(goal_pos)
                utility = UTILITY_VALUES['resource_delivery'] / (distance + 1)
                candidate_goals.append(Goal(
                    goal_type='resource_delivery',
                    target_position=goal_pos,
                    base_utility=UTILITY_VALUES['resource_delivery'],
                    distance_adjusted_utility=utility,
                    description=f"Deliver resource to {goal_pos}"
                ))
        
        # Resource collection (when not carrying)
        if not perception.carrying_resource:
            resource_positions = list(self.known_resources) + perception.get_cells_of_type(CellType.RESOURCE)
            for resource_pos in set(resource_positions):  # Remove duplicates
                distance = perception.current_position.distance_to(resource_pos)
                utility = UTILITY_VALUES['resource_collection'] / (distance + 1)
                candidate_goals.append(Goal(
                    goal_type='resource_collection',
                    target_position=resource_pos,
                    base_utility=UTILITY_VALUES['resource_collection'],
                    distance_adjusted_utility=utility,
                    description=f"Collect resource at {resource_pos}"
                ))
        
        # Exploration goals
        exploration_targets = self._find_exploration_targets(perception)
        for explore_pos in exploration_targets:
            distance = perception.current_position.distance_to(explore_pos)
            utility = UTILITY_VALUES['exploration'] / (distance + 1)
            candidate_goals.append(Goal(
                goal_type='exploration',
                target_position=explore_pos,
                base_utility=UTILITY_VALUES['exploration'],
                distance_adjusted_utility=utility,
                description=f"Explore area at {explore_pos}"
            ))
        
        # Select goal with highest utility
        if candidate_goals:
            best_goal = max(candidate_goals, key=lambda g: g.distance_adjusted_utility)
            
            # Update statistics
            goal_type_key = f"{best_goal.goal_type}_selected"
            if goal_type_key in self.goal_selection_stats:
                self.goal_selection_stats[goal_type_key] += 1
            
            return best_goal
        
        return None
    
    def _generate_plan(self, goal: Goal, perception: Perception) -> List[PlanStep]:
        """
        Generate action plan to achieve the given goal.
        
        Args:
            goal: The goal to achieve
            perception: Current perception data
            
        Returns:
            List of PlanStep objects representing the plan
        """
        if goal.goal_type == 'hazard_avoidance':
            return self._plan_hazard_escape(goal, perception)
        elif goal.goal_type == 'resource_collection':
            return self._plan_resource_collection(goal, perception)
        elif goal.goal_type == 'resource_delivery':
            return self._plan_resource_delivery(goal, perception)
        elif goal.goal_type == 'exploration':
            return self._plan_exploration(goal, perception)
        
        return []
    
    def _plan_hazard_escape(self, goal: Goal, perception: Perception) -> List[PlanStep]:
        """Generate plan to escape from hazard."""
        path = self.find_path(perception.current_position, goal.target_position, perception)
        if path:
            return self._convert_path_to_plan(path, "Escape hazard")
        return []
    
    def _plan_resource_collection(self, goal: Goal, perception: Perception) -> List[PlanStep]:
        """Generate plan to collect a resource."""
        path = self.find_path(perception.current_position, goal.target_position, perception)
        if path:
            plan = self._convert_path_to_plan(path, "Move to resource")
            # Add pickup action
            plan.append(PlanStep(
                action=Action.PICKUP,
                target_position=goal.target_position,
                purpose="Pick up resource",
                estimated_cost=1.0
            ))
            return plan
        return []
    
    def _plan_resource_delivery(self, goal: Goal, perception: Perception) -> List[PlanStep]:
        """Generate plan to deliver a resource."""
        path = self.find_path(perception.current_position, goal.target_position, perception)
        if path:
            plan = self._convert_path_to_plan(path, "Move to goal")
            # Add drop action
            plan.append(PlanStep(
                action=Action.DROP,
                target_position=goal.target_position,
                purpose="Drop resource at goal",
                estimated_cost=1.0
            ))
            return plan
        return []
    
    def _plan_exploration(self, goal: Goal, perception: Perception) -> List[PlanStep]:
        """Generate plan to explore an area."""
        path = self.find_path(perception.current_position, goal.target_position, perception)
        if path:
            return self._convert_path_to_plan(path, "Explore area")
        return []
    
    def find_path(self, start: Position, goal: Position, perception: Perception) -> List[Position]:
        """
        A* path finding algorithm implementation.
        
        Args:
            start: Starting position
            goal: Goal position
            perception: Current perception data
            
        Returns:
            List of positions representing the path, or empty list if no path found
        """
        self.planning_stats['path_finding_calls'] += 1
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, float] = {start: 0}
        f_score: Dict[Position, float] = {start: self._manhattan_distance(start, goal)}
        
        visited = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
			
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                print("path", path)
                print("came_from" , came_from)
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Position(current.x + dx, current.y + dy)
                
                # Skip if neighbor is obstacle or out of bounds
                if self._is_obstacle(neighbor, perception):
                    continue
                
                # Calculate tentative g_score
                move_cost = self._get_move_cost(current, neighbor, perception)
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        # No path found
        self.planning_stats['path_finding_failures'] += 1
        return []
    
    def _manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate Manhattan distance heuristic."""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
    
    def _is_obstacle(self, position: Position, perception: Perception) -> bool:
        """Check if position is an obstacle."""
        # Check known walls
        if position in self.known_walls:
            return True
        
        # Check out of bounds
        if position not in perception.visible_cells:
            return True
        
        # Check visible cells
        cell_type = perception.visible_cells.get(position)
        if cell_type == CellType.WALL:
            return True
        
        # Check for other agents
        if position in perception.visible_agents:
            return True
        
        return False
    
    def _get_move_cost(self, from_pos: Position, to_pos: Position, perception: Perception) -> float:
        """Calculate cost of moving between two positions."""
        base_cost = 1.0
        
        # Higher cost for hazards
        if to_pos in self.known_hazards:
            return base_cost + 5.0
        
        cell_type = perception.visible_cells.get(to_pos)
        if cell_type == CellType.HAZARD:
            return base_cost + 5.0
        
        return base_cost
    
    def _convert_path_to_plan(self, path: List[Position], purpose: str) -> List[PlanStep]:
        """Convert path to list of plan steps."""
        plan = []
        current_pos = self.position
        
        for next_pos in path:
            # Find the action to move from current_pos to next_pos
            dx = next_pos.x - current_pos.x
            dy = next_pos.y - current_pos.y
            
            action = None
            for move_action, (move_dx, move_dy) in DIRECTION_MAPPINGS.items():
                if dx == move_dx and dy == move_dy:
                    action = move_action
                    break
            
            if action:
                plan.append(PlanStep(
                    action=action,
                    target_position=next_pos,
                    purpose=purpose,
                    estimated_cost=1.0
                ))
            
            current_pos = next_pos
        
        return plan
    
    def _is_plan_valid(self, perception: Perception) -> bool:
        """Check if current plan is still valid."""
        if not self.current_plan or self.plan_step_index >= len(self.current_plan):
            return False
        
        # Check if goal still exists and is reachable
        if self.current_goal:
            if self.current_goal.goal_type == 'resource_collection':
                # Check if resource still exists
                target_cell = perception.visible_cells.get(self.current_goal.target_position)
                if target_cell is not None and target_cell != CellType.RESOURCE:
                    if self.current_goal.target_position not in self.known_resources:
                        return False
            elif self.current_goal.goal_type == 'resource_delivery':
                # Check if we're still carrying a resource
                if not perception.carrying_resource:
                    return False
        
        return True
    
    def _execute_current_plan(self, perception: Perception) -> Tuple[Optional[Action], str]:
        """Execute the current step of the plan."""
        if not self.current_plan or self.plan_step_index >= len(self.current_plan):
            return None, "No plan or plan completed"
        
        current_step = self.current_plan[self.plan_step_index]
        
        # Validate the action is still feasible
        if self._is_action_feasible(current_step.action, perception):
            self.plan_step_index += 1
            
            # Check if plan is completed
            if self.plan_step_index >= len(self.current_plan):
                self.planning_stats['plans_completed'] += 1
                self.current_plan = []
                self.current_goal = None
                self.plan_step_index = 0
            
            return current_step.action, current_step.purpose
        else:
            # Plan step not feasible, abandon plan
            self.planning_stats['plans_abandoned'] += 1
            self.planning_stats['replanning_events'] += 1
            self.current_plan = []
            self.current_goal = None
            self.plan_step_index = 0
            return None, "Plan step not feasible, replanning needed"
    
    def _is_action_feasible(self, action: Action, perception: Perception) -> bool:
        """Check if an action is feasible given current perception."""
        if action in DIRECTION_MAPPINGS:
            dx, dy = DIRECTION_MAPPINGS[action]
            new_pos = Position(perception.current_position.x + dx, perception.current_position.y + dy)
            
            # Check if position is valid
            cell_type = perception.visible_cells.get(new_pos)
            if cell_type == CellType.WALL or new_pos in perception.visible_agents:
                return False
        
        elif action == Action.PICKUP:
            current_cell = perception.visible_cells.get(perception.current_position)
            if current_cell != CellType.RESOURCE or perception.carrying_resource:
                return False
        
        elif action == Action.DROP:
            current_cell = perception.visible_cells.get(perception.current_position)
            if current_cell != CellType.GOAL or not perception.carrying_resource:
                return False
        
        return True
    
    def _reactive_fallback(self, perception: Perception) -> Tuple[Action, str]:
        """Reactive fallback behavior when planning fails."""
        # Simple reactive behavior similar to reflex agent
        
        # Immediate hazard avoidance
        if self._is_in_hazard(perception):
            self.energy = max(0, self.energy - 4)
            safe_moves = []
            for action, (dx, dy) in DIRECTION_MAPPINGS.items():
                new_pos = Position(perception.current_position.x + dx, perception.current_position.y + dy)
                cell_type = perception.visible_cells.get(new_pos)
                if cell_type and cell_type not in [CellType.WALL, CellType.HAZARD]:
                    if new_pos not in perception.visible_agents:
                        safe_moves.append(action)
            if safe_moves:
                return random.choice(safe_moves), "Reactive: Escaping hazard"
        
        # Opportunistic actions
        current_cell = perception.visible_cells.get(perception.current_position)
        if current_cell == CellType.RESOURCE and not perception.carrying_resource:
            return Action.PICKUP, "Reactive: Picking up resource"
        if current_cell == CellType.GOAL and perception.carrying_resource:
            return Action.DROP, "Reactive: Dropping at goal"
        
        # Random movement
        valid_moves = []
        for action, (dx, dy) in DIRECTION_MAPPINGS.items():
            new_pos = Position(perception.current_position.x + dx, perception.current_position.y + dy)
            cell_type = perception.visible_cells.get(new_pos)
            if cell_type and cell_type != CellType.WALL:
                if new_pos not in perception.visible_agents:
                    valid_moves.append(action)
        
        if valid_moves:
            return random.choice(valid_moves), "Reactive: Random exploration"
        else:
            return Action.WAIT, "Reactive: No valid moves"
    
    def _is_in_hazard(self, perception: Perception) -> bool:
        """Check if current position contains a hazard."""
        current_cell = perception.visible_cells.get(perception.current_position)
        return current_cell == CellType.HAZARD
    
    def _find_safe_positions(self, perception: Perception) -> dict[str : List[Position]]:
        """Find safe positions to escape to."""
        all_available_positions: Set[Tuple[Position, CellType]] = (
            {(pos, CellType.EMPTY) for pos in self.known_empty} |
            {(pos, CellType.GOAL) for pos in self.known_goals} |
            {(pos, CellType.RESOURCE) for pos in self.known_resources}
        )
        safe_positions = []
        preferred_positions = []
        print(f"test" , all_available_positions)
        for position, cell_type in all_available_positions:
            if cell_type in [CellType.EMPTY, CellType.RESOURCE, CellType.GOAL]:
                if position not in perception.visible_agents:
                    safe_positions.append(position)
                    if not self.carrying_resource and cell_type == CellType.RESOURCE:
                        preferred_positions.append(position)
                    elif self.carrying_resource and cell_type == CellType.GOAL:
                        preferred_positions.append(position)
                    elif position not in self.visited_positions:
                        preferred_positions.append(position) 

        return {
            "safe_positions" : safe_positions,
            "preferred_safe_moves" : preferred_positions
        }
    
    def _find_exploration_targets(self, perception: Perception) -> List[Position]:
        """Find good exploration targets."""
        targets = []
        
        # Look for edges of visible area
        for position in perception.visible_cells.keys():
            # Check if position is at edge of perception
            if self._is_edge_position(position, perception):
                if position not in self.visited_positions:
                    targets.append(position)
        
        # If no edge targets, look for any unvisited visible positions
        if not targets:
            for position, cell_type in perception.visible_cells.items():
                if cell_type != CellType.WALL and position not in self.visited_positions:
                    targets.append(position)
        
        return targets[:5]  # Limit to 5 targets to avoid excessive computation
    
    def _is_edge_position(self, position: Position, perception: Perception) -> bool:
        """Check if position is at the edge of visible area."""
        # Simple heuristic: position is at edge if it has fewer than 4 visible neighbors
        neighbor_count = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = Position(position.x + dx, position.y + dy)
            if neighbor in perception.visible_cells:
                neighbor_count += 1
        return neighbor_count < 4
    
    def get_planning_statistics(self) -> Dict[str, int]:
        """Get planning system statistics."""
        return self.planning_stats.copy()
    
    def get_goal_statistics(self) -> Dict[str, int]:
        """Get goal selection statistics."""
        return self.goal_selection_stats.copy()
    
    def get_current_plan_info(self) -> Dict[str, any]:
        """Get information about current plan."""
        return {
            'has_plan': len(self.current_plan) > 0,
            'plan_length': len(self.current_plan),
            'current_step': self.plan_step_index,
            'remaining_steps': len(self.current_plan) - self.plan_step_index,
            'current_goal': self.current_goal.description if self.current_goal else None,
            'current_goal_type': self.current_goal.goal_type if self.current_goal else None
        }