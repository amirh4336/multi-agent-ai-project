from src.core.environment import GridWorld
from src.agents.simple_reflex_agent import SimpleReflexAgent
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import json

class InteractiveSimulation:
    def __init__(self, config_path: str):
        # Load environment
        self.env = GridWorld(config_path)
        
        # Get agent info from config
        agent_configs = list(self.env.agents.items())
        
        # Create agent instances
        self.agent_instances = []
        for agent_id, agent_data in agent_configs:
            agent = SimpleReflexAgent(agent_id, agent_data['position'])
            self.agent_instances.append(agent)
            # Register agent instance with environment for statistics tracking
            self.env.register_agent_instance(agent)
        
        # Simulation state
        self.current_step = 0
        self.max_steps = 100
        self.simulation_ended = False
        
        # Setup matplotlib figure
        self.setup_figure()
        
        print("Interactive simulation initialized!")
        print(f"Agents: {[agent.agent_id for agent in self.agent_instances]}")
        print("Click 'Next Step' to advance the simulation")
    
    def setup_figure(self):
        """Setup the matplotlib figure with grid and button"""
        # Calculate figure dimensions
        num_agents = len(self.agent_instances)
        stats_height = max(3, num_agents * 0.8)
        
        self.fig = plt.figure(figsize=(max(12, self.env.grid_size[0] * 1.2), 
                                     self.env.grid_size[1] + stats_height + 1))
        
        # Create main grid subplot
        self.ax_grid = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
        
        # Create statistics subplot
        self.ax_stats = plt.subplot2grid((3, 1), (1, 0), rowspan=1)
        
        # Create button subplot - positioned in bottom right
        ax_button = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
        # Calculate button dimensions: 200px width, 40px height
        # Convert pixels to figure coordinates (approximate)
        fig_width_inches = self.fig.get_figwidth()
        fig_height_inches = self.fig.get_figheight()
        dpi = self.fig.dpi
        
        button_width_norm = 10 / (fig_width_inches * dpi)  # 200px to normalized units
        button_height_norm = 50 / (fig_height_inches * dpi)  # 40px to normalized units
        
        # Position in bottom right with some margin
        margin = 0.02
        left = 1 - button_width_norm - margin
        bottom = margin
        
        ax_button.set_position([left, bottom, button_width_norm, button_height_norm])
        
        # Create the Next Step button
        self.button = Button(ax_button, 'Next Step', color='lightblue', hovercolor='blue')
        self.button.on_clicked(self.next_step)
        
        # Initial render
        self.render_current_state()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for button
    
    def next_step(self, event):
        """Execute one step of the simulation"""
        if self.simulation_ended:
            print("Simulation has ended!")
            return
        
        if self.current_step >= self.max_steps:
            print("Maximum steps reached!")
            self.simulation_ended = True
            return
        
        print(f"\n--- Step {self.current_step} ---")
        
        # Each agent acts
        active_agents = 0
        for agent in self.agent_instances:
            if not agent.is_active():
                print(f"{agent.agent_id} is inactive (no energy)")
                continue
            
            active_agents += 1
            perception = agent.perceive(self.env)
            action, reason = agent.decide_action(perception)
            print(f"{agent.agent_id} at {agent.position} -> {action.name}: {reason}")
            
            # Let the environment handle action application and updates
            self.env.apply_action(agent, action)
        
        # Check if simulation should continue
        if active_agents == 0:
            print("All agents are inactive. Ending simulation.")
            self.simulation_ended = True
            self.show_final_summary()
            # Change button text to indicate simulation ended
            self.button.label.set_text('Simulation Ended')
            return
        
        self.current_step += 1
        
        # Update the display
        self.render_current_state()
        self.fig.canvas.draw()
    
    def render_current_state(self):
        """Render the current state of the grid and statistics"""
        # Clear previous content
        self.ax_grid.clear()
        self.ax_stats.clear()
        
        # Setup grid
        self.ax_grid.set_xlim(0, self.env.grid_size[0])
        self.ax_grid.set_ylim(0, self.env.grid_size[1])
        self.ax_grid.set_xticks(range(self.env.grid_size[0] + 1))
        self.ax_grid.set_yticks(range(self.env.grid_size[1] + 1))
        self.ax_grid.set_xticklabels([])
        self.ax_grid.set_yticklabels([])
        self.ax_grid.grid(True)
        
        # Cell drawing helper
        def draw_cell(pos, color, label=None):
            rect = patches.Rectangle(pos, 1, 1, facecolor=color, edgecolor='black')
            self.ax_grid.add_patch(rect)
            if label:
                self.ax_grid.text(pos[0]+0.5, pos[1]+0.5, label, ha='center', va='center', fontsize=10)
        
        # Draw walls
        for wall_pos in self.env.walls:
            draw_cell((wall_pos.x, wall_pos.y), "black", label="Wall")
        
        # Draw goals
        for goal_pos in self.env.goals:
            draw_cell((goal_pos.x, goal_pos.y), "green", label="Goal")
        
        # Draw resources
        for resource_pos in self.env.resources:
            draw_cell((resource_pos.x, resource_pos.y), "blue", label="Resource")
        
        # Draw hazards
        for hazard_pos in self.env.hazards:
            draw_cell((hazard_pos.x, hazard_pos.y), "orange", label="Hazard")
        
        # Draw agents
        for agent_id, info in self.env.agents.items():
            x = info["position"].x
            y = info["position"].y
            triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, orientation=0,
                                            color='purple')
            self.ax_grid.add_patch(triangle)
            self.ax_grid.text(x + 0.5, y + 0.2, agent_id, ha='center', va='center', color='white', fontsize=8)
        
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title(f"GridWorld Environment - Step {self.current_step}")
        
        # Display agent statistics
        self.display_agent_statistics()
    
    def display_agent_statistics(self):
        """Display performance metrics and statistics for all agents."""
        self.ax_stats.axis('off')
        
        if not self.env.agent_instances:
            self.ax_stats.text(0.5, 0.5, "No agent statistics available", 
                                ha='center', va='center', transform=self.ax_stats.transAxes, fontsize=12)
            return
        
        # Prepare statistics text
        stats_text = "AGENT PERFORMANCE METRICS & STATISTICS\n"
        stats_text += "=" * 80 + "\n\n"
        
        total_cells = self.env.grid_size[0] * self.env.grid_size[1]
        
        for agent_id, agent in self.env.agent_instances.items():
            # Get basic statistics
            basic_stats = agent.get_statistics_summary()
            
            # Calculate explored cells (simplified)
            explored_cells = self.env._calculate_explored_cells(agent)
            
            # Get performance metrics
            perf_metrics = agent.get_performance_metrics(
                total_steps=self.env.step_count,
                explored_cells=explored_cells,
                total_cells=total_cells
            )
            
            # Format agent statistics
            stats_text += f"AGENT: {agent_id}\n"
            stats_text += "-" * 40 + "\n"
            
            # Basic Statistics
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
        self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes, 
                            fontfamily='monospace', fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
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
    # Initialize and run the interactive simulation
    sim = InteractiveSimulation("data/environments/simple_collection.json")
    sim.run()

if __name__ == "__main__":
    run_interactive_simulation()