import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
            "position": tuple(agent["position"]),
            "energy": agent["energy"],
            "carrying": False
        } for agent in config["agents"]}

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
            x, y = info["position"]
            triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, orientation=0,
                                              color='purple')
            ax.add_patch(triangle)
            ax.text(x + 0.5, y + 0.2, agent_id, ha='center', va='center', color='white', fontsize=8)

        ax.set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.title("GridWorld Environment")
        plt.show()
