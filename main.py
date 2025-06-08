import json
import copy
from turtle import bgcolor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from src.agents.goal_based_agent import GoalBasedAgent
from src.agents.simple_reflex_agent import SimpleReflexAgent
from src.core.environment import GridWorld
from src.agents.model_based_reflex_agent import ModelBasedReflexAgent

class InteractiveSimulation:
    def __init__(self, config_path: str):
        # Load environment
        self.env = GridWorld(config_path)
        
        # Get agent info from config
        agent_configs = list(self.env.agents.items())
        
        # Create agent instances
        self.agent_instances = []
        for agent_id, agent_data in agent_configs:
            match agent_data["type"]:
                case "goal":
                    agent = GoalBasedAgent(agent_id, agent_data['position'])
                case "module":
                    agent = ModelBasedReflexAgent(agent_id, agent_data['position'])
                case "simple":
                    agent = SimpleReflexAgent(agent_id, agent_data['position'])

            self.agent_instances.append(agent)
            self.env.register_agent_instance(agent)
        
        # Simulation state
        self.current_step = 0
        self.max_steps = 100
        self.simulation_ended = False
        self.auto_running = False
        self.history = []  # Store simulation states for undo
        self.save_state()  # Save initial state
        
        # Animation
        self.anim = None
        
        # Setup matplotlib figure
        self.setup_figure()
        
        print("Interactive simulation initialized!")
        print(f"Agents: {[agent.agent_id for agent in self.agent_instances]}")
        print("Use 'Next', 'Previous', 'Auto Run', or 'Pause/Resume' buttons.")
    
    def save_state(self):
        """Save the current simulation state to history"""
        state = {
            'step': self.current_step,
            'env': {
                'agents': copy.deepcopy(self.env.agents),
                'resources': copy.deepcopy(self.env.resources),
                'step_count': self.env.step_count
            },
            'agent_instances': [
                {
                    'position': copy.deepcopy(agent.position),
                    'energy': agent.energy,
                    'carrying_resource': agent.carrying_resource,
                    'actions_taken': agent.actions_taken,
                    'resources_collected': agent.resources_collected,
                    'goals_reached': agent.goals_reached,
                    'collisions': agent.collisions,
                    'last_action': agent.last_action
                } for agent in self.agent_instances
            ]
        }
        self.history.append(state)
        # Limit history size to prevent memory issues
        if len(self.history) > self.max_steps + 1:
            self.history.pop(0)
    
    def restore_state(self):
        """Restore the previous state from history"""
        if len(self.history) <= 1:
            return False  # No previous state
        self.history.pop()  # Remove current state
        prev_state = self.history[-1]
        
        # Restore environment
        self.current_step = prev_state['step']
        self.env.agents = copy.deepcopy(prev_state['env']['agents'])
        self.env.resources = copy.deepcopy(prev_state['env']['resources'])
        self.env.step_count = prev_state['env']['step_count']
        
        # Restore agent instances
        for i, agent in enumerate(self.agent_instances):
            agent_state = prev_state['agent_instances'][i]
            agent.position = copy.deepcopy(agent_state['position'])
            agent.energy = agent_state['energy']
            agent.carrying_resource = agent_state['carrying_resource']
            agent.actions_taken = agent_state['actions_taken']
            agent.resources_collected = agent_state['resources_collected']
            agent.goals_reached = agent_state['goals_reached']
            agent.collisions = agent_state['collisions']
            agent.last_action = agent_state['last_action']
        
        self.simulation_ended = False
        return True
    
    def setup_figure(self):
        """Setup the matplotlib figure with grid, statistics, and buttons"""
        num_agents = len(self.agent_instances)
        grid_width = max(12, self.env.grid_size[0] * 1.2)
        grid_height = self.env.grid_size[1] * 1.2
        stats_height = 8
        button_height = 0.8
        
        self.fig = plt.figure(figsize=(grid_width, grid_height + stats_height + button_height))
        
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 1, height_ratios=[grid_height, stats_height, button_height], hspace=0.3)
        
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_stats = self.fig.add_subplot(gs[1, 0])
        
        # Create button axes (horizontal layout)
        button_width = 0.12
        button_height = 0.04
        margin_x = 0.02
        margin_y = 0.02
        total_button_width = button_width * 4 + margin_x * 3
        start_x = 1 - total_button_width - margin_x
        
        # Next Step button
        ax_next = self.fig.add_axes([start_x + button_width + margin_x, margin_y, button_width, button_height])
        self.button_next = Button(ax_next, 'Next Step', color='lightblue', hovercolor='blue')
        self.button_next.on_clicked(self.next_step)
        
        # Previous Step button
        ax_prev = self.fig.add_axes([start_x, margin_y, button_width, button_height])
        self.button_prev = Button(ax_prev, 'Previous', color='lightblue', hovercolor='blue')
        self.button_prev.on_clicked(self.prev_step)
        
        # Auto Run button
        ax_auto = self.fig.add_axes([start_x + (button_width + margin_x) * 2, margin_y, button_width, button_height])
        self.button_auto = Button(ax_auto, 'Auto Run', color='lightgreen', hovercolor='green')
        self.button_auto.on_clicked(self.start_auto)
        
        # Pause/Resume button
        ax_pause = self.fig.add_axes([start_x + (button_width + margin_x) * 3, margin_y, button_width, button_height])
        self.button_pause = Button(ax_pause, 'Pause', color='lightcoral', hovercolor='red')
        self.button_pause.on_clicked(self.toggle_pause)
        
        self.render_current_state()
        plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    
    def next_step(self, event=None):
        """Execute one step of the simulation"""
        if self.simulation_ended:
            print("Simulation has ended!")
            return
        
        if self.current_step >= self.max_steps:
            print("Maximum steps reached!")
            self.simulation_ended = True
            self.stop_auto()
            return
        
        print(f"\n--- Step {self.current_step} ---")
        
        active_agents = 0
        for agent in self.agent_instances:
            if not agent.is_active():
                print(f"{agent.agent_id} is inactive (no energy)")
                continue
            active_agents += 1
            perception = agent.perceive(self.env)
            action, reason = agent.decide_action(perception)
            print(f"{agent.agent_id} at {agent.position} -> {action.name}: {reason}")
            self.env.apply_action(agent, action)
        
        if active_agents == 0:
            print("All agents are inactive. Ending simulation.")
            self.simulation_ended = True
            self.stop_auto()
            self.button_next.label.set_text('Simulation Ended')
            self.show_final_summary()
            return
        
        self.current_step += 1
        self.env.step_count = self.current_step  # Sync step count
        self.save_state()
        
        self.render_current_state()
        self.fig.canvas.draw()
    
    def prev_step(self, event):
        """Revert to the previous simulation state"""
        if self.auto_running:
            print("Cannot go back during auto-run. Pause first.")
            return
        
        if self.restore_state():
            print(f"\n--- Reverted to Step {self.current_step} ---")
            self.button_next.label.set_text('Next Step')  # Reset button text
            self.render_current_state()
            self.fig.canvas.draw()
        else:
            print("No previous state available.")
    
    def start_auto(self, event):
        """Start auto-iteration of simulation steps"""
        if self.simulation_ended:
            print("Simulation has ended!")
            return
        if not self.auto_running:
            self.auto_running = True
            self.button_pause.label.set_text('Pause')
            self.anim = FuncAnimation(self.fig, self.next_step, interval=500, cache_frame_data=False)
            self.fig.canvas.draw()
    
    def toggle_pause(self, event):
        """Pause or resume auto-iteration"""
        if self.auto_running:
            self.stop_auto()
            self.button_pause.label.set_text('Resume')
            print("Auto-run paused. Use 'Next Step' or 'Previous' for manual control.")
        else:
            if self.simulation_ended:
                print("Simulation has ended!")
                return
            self.auto_running = True
            self.button_pause.label.set_text('Pause')
            self.anim = FuncAnimation(self.fig, self.next_step, interval=500, cache_frame_data=False)
            self.fig.canvas.draw()
    
    def stop_auto(self):
        """Stop auto-iteration"""
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None
        self.auto_running = False
    
    def render_current_state(self):
        """Render the current state of the grid and statistics"""
        self.ax_grid.clear()
        
        self.ax_grid.set_xlim(0, self.env.grid_size[0])
        self.ax_grid.set_ylim(0, self.env.grid_size[1])
        self.ax_grid.set_xticks(range(self.env.grid_size[0] + 1))
        self.ax_grid.set_yticks(range(self.env.grid_size[1] + 1))
        self.ax_grid.set_xticklabels([])
        self.ax_grid.set_yticklabels([])
        self.ax_grid.grid(True)
        
        def draw_cell(pos, color, label=None , label_color = "black"):
            rect = patches.Rectangle(pos, 1, 1, facecolor=color, edgecolor='black')
            self.ax_grid.add_patch(rect)
            if label:
                self.ax_grid.text(pos[0]+0.5, pos[1]+0.5, label, ha='center', va='center' , color=label_color, fontsize=10)
        
        for wall_pos in self.env.walls:
            draw_cell((wall_pos.x, wall_pos.y), "black", label="W" , label_color="white")
        for goal_pos in self.env.goals:
            draw_cell((goal_pos.x, goal_pos.y), "green", label="G")
        for resource_pos in self.env.resources:
            draw_cell((resource_pos.x, resource_pos.y), "blue", label="R")
        for hazard_pos in self.env.hazards:
            draw_cell((hazard_pos.x, hazard_pos.y), "orange", label="H")
        
        for agent_id, info in self.env.agents.items():
            x = info["position"].x
            y = info["position"].y

            current_energy = 0
            for agent in self.agent_instances:
                if agent.agent_id == agent_id:
                    current_energy = agent.energy
                    break
            
            triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, 
                                                orientation=0, color=info["bg_color"])
            self.ax_grid.add_patch(triangle)
            self.ax_grid.text(x + 0.5, y + 0.2, current_energy, ha='center', va='center', 
                                color='black', fontsize=8)
        
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title(f"GridWorld Environment - Step {self.current_step}")
        
        self.display_agent_statistics()
    
    def display_agent_statistics(self):
        """Display performance metrics and statistics for all agents in a flex-wrap layout"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if not self.env.agent_instances:
            self.ax_stats.text(0.5, 0.5, "No agent statistics available", 
                                ha='center', va='center', transform=self.ax_stats.transAxes, fontsize=12)
            return
        
        self.ax_stats.text(0.5, 0.95, "AGENT PERFORMANCE METRICS", 
                            transform=self.ax_stats.transAxes, 
                            fontsize=12, fontweight='bold', ha='center', va='top')
        
        total_cells = self.env.grid_size[0] * self.env.grid_size[1]
        agents = list(self.env.agent_instances.items())
        num_agents = len(agents)
        
        box_width = 0.30
        box_height = 0.80
        margin_x = 0.02
        margin_y = 0.08
        start_y = 0.85
        
        boxes_per_row = int((1.0 - margin_x) / (box_width + margin_x)) or 1
        
        for i, (agent_id, agent) in enumerate(agents):
            row = i // boxes_per_row
            col = i % boxes_per_row
            x_pos = margin_x + col * (box_width + margin_x)
            y_pos = start_y - row * (box_height + margin_y)
            
            basic_stats = agent.get_statistics_summary()
            explored_cells = self.env._calculate_explored_cells(agent)
            perf_metrics = agent.get_performance_metrics(
                total_steps=self.env.step_count,
                explored_cells=explored_cells,
                total_cells=total_cells
            )

            agent_color = self.env.agents[agent_id]["bg_color"]
            
            stats_text = f"{agent_id}\n"
            stats_text += f"Color: {agent_color}\n" 
            stats_text += f"Energy: {basic_stats['energy_remaining']}/100\n"
            stats_text += f"Actions: {basic_stats['actions_taken']}\n"
            stats_text += f"Resources: {basic_stats['resources_collected']}\n"
            stats_text += f"Goals: {basic_stats['goals_reached']}\n"
            stats_text += f"Collisions: {basic_stats['collisions']}\n"
            stats_text += f"Carrying: {basic_stats['carrying_resource']}\n"
            stats_text += f"Last: {basic_stats['last_action']}\n"
            stats_text += f"Success: {perf_metrics.success_rate:.2f}\n"
            stats_text += f"Efficiency: {perf_metrics.efficiency_score:.1f}%\n"
            stats_text += f"Energy Use: {perf_metrics.energy_utilization:.1f}%\n"
            stats_text += f"Coverage: {perf_metrics.exploration_coverage:.1%}"
            
            box_rect = patches.Rectangle((x_pos, y_pos - box_height), box_width, box_height,
                                        facecolor='lightblue', edgecolor='navy', 
                                        linewidth=1, alpha=0.7, 
                                        transform=self.ax_stats.transAxes)
            self.ax_stats.add_patch(box_rect)
            
            self.ax_stats.text(x_pos + box_width/2, y_pos - box_height/2, stats_text,
                                transform=self.ax_stats.transAxes,
                                fontfamily='monospace', fontsize=8,
                                ha='center', va='center', color='darkblue')
        
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
    
    def show_final_summary(self):
        """Display final simulation summary"""
        print("\n" + "="*80)
        print("FINAL SIMULATION SUMMARY")
        print("="*80)
        
        summary = self.env.get_simulation_summary()
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Total Agents: {summary['total_agents']}")
        print(f"Environment Size: {summary['environment_size']}")
        
        for agent_id, agent_data in summary['agents'].items():
            print(f"\nFinal stats for {agent_id}:")
            basic_stats = agent_data['basic_stats']
            perf_metrics = agent_data['performance_metrics']
            print(f"  Goals Reached: {basic_stats['goals_reached']}")
            print(f"  Resources Collected: {basic_stats['resources_collected']}")
            print(f"  Energy Remaining: {basic_stats['energy_remaining']}")
            print(f"  Efficiency Score: {perf_metrics.efficiency_score:.2f}%")
            print(f"  Success Rate: {perf_metrics.success_rate:.2f}")
    
    def run(self):
        """Start the interactive simulation"""
        plt.show()

def run_interactive_simulation():
    """Main function to run the interactive simulation"""
    sim = InteractiveSimulation("data/environments/simple_collection.json")
    sim.run()

if __name__ == "__main__":
    run_interactive_simulation()