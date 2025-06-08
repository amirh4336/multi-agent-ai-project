import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from .base_agent import BaseAgent
from .data_structures import Action, Perception, Position, PERCEPTION_RANGE, DIRECTION_MAPPINGS, CellType

class GridWorld:
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.step_count = 0  # Track simulation steps
        self.agent_instances = {}  # Store agent instances for metrics calculation

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)
        self.grid_size = tuple(config["grid_size"])
        self.walls = set(Position(*pos) for pos in config["walls"])
        self.goals = set(Position(*pos) for pos in config["goals"])
        self.resources = set(Position(*pos) for pos in config["resources"])
        self.hazards = set(Position(*pos) for pos in config["hazards"])
        self.agents = {agent["id"]: {
            "position": Position(*agent["position"]),
            "energy": agent["energy"],
            "carrying": False,
            "type" : agent["type"]
        } for agent in config["agents"]}

    def register_agent_instance(self, agent: BaseAgent):
        """Register agent instance for metrics tracking."""
        self.agent_instances[agent.agent_id] = agent

    def apply_action(self, agent: BaseAgent, action: Action):
        # Register agent if not already registered
        if agent.agent_id not in self.agent_instances:
            self.register_agent_instance(agent)
            
        current_pos = agent.position
        new_pos = current_pos
        if action in DIRECTION_MAPPINGS:
            dx, dy = DIRECTION_MAPPINGS[action]
            candidate_pos = current_pos.move(dx, dy)

            # Block if wall or out of bounds
            if (candidate_pos not in self._valid_positions()
                or candidate_pos in self.walls):
                agent.record_collision()
                return

            # Move agent
            agent.update_position(candidate_pos)
            self.agents[agent.agent_id]["position"] = candidate_pos

        elif action == Action.PICKUP:
            if current_pos in self.resources and not agent.carrying_resource:
                agent.pickup_resource()
                self.resources.remove(current_pos)

        elif action == Action.DROP:
            if agent.carrying_resource and current_pos in self.goals:
                agent.reach_goal()

        elif action == Action.WAIT:
            pass  # do nothing

        elif action == Action.COMMUNICATE:
            pass  # add message system later

        # Update energy after all actions
        agent.execute_action(action)
        self.step_count += 1

    def _valid_positions(self):
        """Returns all valid (non-wall, in-bounds) positions on grid."""
        width, height = self.grid_size
        return {
            Position(x, y)
            for x in range(width)
            for y in range(height)
        }

    def get_agent_perception(self, agent_id: str) -> Perception:
        """Returns the perception of the agent at its current position."""
        agent_info = self.agents[agent_id]
        current_pos = agent_info["position"]
        
        # Calculate visible cells within perception range
        visible_cells = {}
        min_x = max(0, current_pos.x - PERCEPTION_RANGE)
        max_x = min(self.grid_size[0] - 1, current_pos.x + PERCEPTION_RANGE)
        min_y = max(0, current_pos.y - PERCEPTION_RANGE)
        max_y = min(self.grid_size[1] - 1, current_pos.y + PERCEPTION_RANGE)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pos = Position(x, y)
                
                # Skip walls (they block perception)
                
                    
                # Determine cell type
                if pos in self.goals:
                    cell_type = CellType.GOAL
                elif pos in self.resources:
                    cell_type = CellType.RESOURCE
                elif pos in self.hazards:
                    cell_type = CellType.HAZARD
                elif pos in self.walls:
                    cell_type = CellType.WALL
                else:
                    cell_type = CellType.EMPTY
                    
                visible_cells[pos] = cell_type
        
        # Find visible agents (excluding self)
        visible_agents = []
        for other_id, other_info in self.agents.items():
            if other_id == agent_id:
                continue
                
            other_pos = other_info["position"]  # Already a Position object
            if (abs(current_pos.x - other_pos.x) <= PERCEPTION_RANGE and 
                abs(current_pos.y - other_pos.y) <= PERCEPTION_RANGE):
                visible_agents.append(other_pos)
        
        return Perception(
            current_position=current_pos,
            visible_cells=visible_cells,
            visible_agents=visible_agents,
            energy_level=agent_info["energy"],
            carrying_resource=agent_info["carrying"],
            messages=[]  # Will be populated by message system
        )

    def _calculate_explored_cells(self, agent: BaseAgent) -> int:
        """Calculate number of cells explored by agent (simplified estimation)."""
        # This is a simplified version - in practice you'd track visited positions
        return min(agent.actions_taken * 2, self.grid_size[0] * self.grid_size[1])

    def render(self):
        # Calculate figure height to accommodate statistics
        num_agents = len(self.agent_instances)
        stats_height = max(3, num_agents * 0.8)  # Height for statistics display
        
        fig = plt.figure(figsize=(max(12, self.grid_size[0] * 1.2), 
                                self.grid_size[1] + stats_height))
        
        # Create main grid subplot
        ax_grid = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
        ax_grid.set_xlim(0, self.grid_size[0])
        ax_grid.set_ylim(0, self.grid_size[1])
        ax_grid.set_xticks(range(self.grid_size[0] + 1))
        ax_grid.set_yticks(range(self.grid_size[1] + 1))
        ax_grid.set_xticklabels([])
        ax_grid.set_yticklabels([])
        ax_grid.grid(True)

        # Cell drawing helper
        def draw_cell(pos, color, label=None):
            rect = patches.Rectangle(pos, 1, 1, facecolor=color, edgecolor='black')
            ax_grid.add_patch(rect)
            if label:
                ax_grid.text(pos[0]+0.5, pos[1]+0.5, label, ha='center', va='center', fontsize=10)

        # Draw walls
        for wall_pos in self.walls:
            draw_cell((wall_pos.x, wall_pos.y), "black", label="Wall")

        # Draw goals
        for goal_pos in self.goals:
            draw_cell((goal_pos.x, goal_pos.y), "green", label="Goal")

        # Draw resources
        for resource_pos in self.resources:
            draw_cell((resource_pos.x, resource_pos.y), "blue", label="Resource")

        # Draw hazards
        for hazard_pos in self.hazards:
            draw_cell((hazard_pos.x, hazard_pos.y), "orange", label="Hazard")

        # Draw agents
        for agent_id, info in self.agents.items():
            x = info["position"].x
            y = info["position"].y
            triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, orientation=0,
                                            color='red')
            ax_grid.add_patch(triangle)
            ax_grid.text(x + 0.5, y + 0.2, agent_id, ha='center', va='center', color='black', fontsize=8)

        ax_grid.set_aspect('equal')
        ax_grid.invert_yaxis()
        ax_grid.set_title(f"GridWorld Environment - Step {self.step_count}")

        # Create statistics subplot
        ax_stats = plt.subplot2grid((2, 1), (1, 0), rowspan=1)
        ax_stats.axis('off')
        
        # Display agent statistics
        self._display_agent_statistics(ax_stats)
        
        plt.tight_layout()
        plt.show()

    def _display_agent_statistics(self, ax):
        """Display performance metrics and statistics for all agents."""
        if not self.agent_instances:
            ax.text(0.5, 0.5, "No agent statistics available", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return

        # Prepare statistics text
        stats_text = "AGENT PERFORMANCE METRICS & STATISTICS\n"
        stats_text += "=" * 80 + "\n\n"
        
        total_cells = self.grid_size[0] * self.grid_size[1]
        
        for agent_id, agent in self.agent_instances.items():
            # Get basic statistics
            basic_stats = agent.get_statistics_summary()
            
            # Calculate explored cells (simplified)
            explored_cells = self._calculate_explored_cells(agent)
            print(basic_stats)
            # Get performance metrics
            perf_metrics = agent.get_performance_metrics(
                total_steps=self.step_count,
                explored_cells=explored_cells,
                total_cells=total_cells
            )
            
            # Format agent statistics
            stats_text += f"AGENT: {agent_id}\n"
            stats_text += "-" * 40 + "\n"
            
            # Basic Statistics
            # stats_text += f"Position: {basic_stats['position']}\n"
            stats_text += f"Energy: {basic_stats['energy_remaining']}/{100} "
            stats_text += f"(Consumed: {basic_stats['energy_consumed']})\n"
            stats_text += f"Actions Taken: {basic_stats['actions_taken']}\n"
            stats_text += f"Resources Collected: {basic_stats['resources_collected']}\n"
            stats_text += f"Goals Reached: {basic_stats['goals_reached']}\n"
            stats_text += f"Collisions: {basic_stats['collisions']}\n"
            stats_text += f"Carrying Resource: {basic_stats['carrying_resource']}\n"
            stats_text += f"Last Action: {basic_stats['last_action']}\n"
            
            # Performance Metrics
            stats_text += f"\nPerformance Metrics:\n"
            stats_text += f"  Success Rate: {perf_metrics.success_rate:.2f}\n"
            stats_text += f"  Efficiency Score: {perf_metrics.efficiency_score:.2f}%\n"
            stats_text += f"  Task Completion Time: {perf_metrics.task_completion_time:.2f}\n"
            stats_text += f"  Energy Utilization: {perf_metrics.energy_utilization:.2f}%\n"
            stats_text += f"  Collision Frequency: {perf_metrics.collision_frequency}\n"
            stats_text += f"  Exploration Coverage: {perf_metrics.exploration_coverage:.2%}\n"
            
            stats_text += "\n" + "=" * 80 + "\n"
        
        # Display the text
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    def get_simulation_summary(self):
        """Get overall simulation summary with all agent statistics."""
        summary = {
            'total_steps': self.step_count,
            'total_agents': len(self.agent_instances),
            'environment_size': self.grid_size,
            'agents': {}
        }
        
        total_cells = self.grid_size[0] * self.grid_size[1]
        
        for agent_id, agent in self.agent_instances.items():
            explored_cells = self._calculate_explored_cells(agent)
            
            summary['agents'][agent_id] = {
                'basic_stats': agent.get_statistics_summary(),
                'performance_metrics': agent.get_performance_metrics(
                    total_steps=self.step_count,
                    explored_cells=explored_cells,
                    total_cells=total_cells
                )
            }
        
        return summary