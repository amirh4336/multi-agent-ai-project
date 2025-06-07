import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .base_agent import BaseAgent
from .data_structures import Action , Perception , Position , PERCEPTION_RANGE , DIRECTION_MAPPINGS , CellType

class GridWorld:
    def __init__(self, config_path: str):
        self.load_config(config_path)

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)
        self.grid_size = tuple(config["grid_size"])
        self.walls = set(tuple(pos) for pos in config["walls"])
        self.goals = set(tuple(pos) for pos in config["goals"])
        self.resources = set(tuple(pos) for pos in config["resources"])
        self.hazards = set(tuple(pos) for pos in config["hazards"])
        self.agents = {agent["id"]: {
            "position": Position(x=agent["position"][0], y=agent["position"][1]),
            "energy": agent["energy"],
            "carrying": False
        } for agent in config["agents"]}

    def apply_action(self, agent: BaseAgent, action: Action):
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
                if (x, y) in self.walls:
                    continue
                    
                # Determine cell type
                if (x, y) in self.goals:
                    cell_type = CellType.GOAL
                elif (x, y) in self.resources:
                    cell_type = CellType.RESOURCE
                elif (x, y) in self.hazards:
                    cell_type = CellType.HAZARD
                else:
                    cell_type = CellType.EMPTY
                    
                visible_cells[pos] = cell_type
        
        # Find visible agents (excluding self)
        visible_agents = []
        for other_id, other_info in self.agents.items():
            if other_id == agent_id:
                continue
                
            other_pos = Position(*other_info["position"])
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

    def render(self):
        fig, ax = plt.subplots(figsize=(self.grid_size[0], self.grid_size[1]))
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_xticks(range(self.grid_size[0] + 1))
        ax.set_yticks(range(self.grid_size[1] + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        # Cell drawing helper
        def draw_cell(pos, color, label=None):
            rect = patches.Rectangle(pos, 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            if label:
                ax.text(pos[0]+0.5, pos[1]+0.5, label, ha='center', va='center', fontsize=10)

        # Draw walls
        for x, y in self.walls:
            draw_cell((x, y), "black", label="Wall")

        # Draw goals
        for x, y in self.goals:
            draw_cell((x, y), "green" , label="Goal")

        # Draw resources
        for x, y in self.resources:
            draw_cell((x, y), "blue", label="Resource")

        # Draw hazards
        for x, y in self.hazards:
            draw_cell((x, y), "orange", label="Hazard")

        # Draw agents
        for agent_id, info in self.agents.items():
            x = info["position"].x
            y = info["position"].y
            triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, orientation=0,
                                            color='purple')
            ax.add_patch(triangle)
            ax.text(x + 0.5, y + 0.2, agent_id, ha='center', va='center', color='white', fontsize=8)

        ax.set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.title("GridWorld Environment")
        plt.show()
