import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from src.core.environment import GridWorld
from src.agents.simple_reflex_agent import SimpleReflexAgent

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
        """Setup the matplotlib figure with grid, statistics, and button"""
        # Calculate figure dimensions
        num_agents = len(self.agent_instances)
        grid_width = max(12, self.env.grid_size[0] * 1.2)
        grid_height = self.env.grid_size[1] * 1.2  # Increased grid height
        stats_height = 8  # Increased height for statistics display to accommodate larger boxes
        button_height = 0.8  # Height for button area
        
        # Create figure with adjusted height
        self.fig = plt.figure(figsize=(grid_width, grid_height + stats_height + button_height))
        
        # Use GridSpec for layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 1, height_ratios=[grid_height, stats_height, button_height], 
                      hspace=0.3)
        
        # Create main grid subplot
        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        
        # Create statistics subplot
        self.ax_stats = self.fig.add_subplot(gs[1, 0])
        
        # Create button axes in figure-normalized coordinates
        button_width = 0.15
        button_height = 0.04
        margin_x = 0.02
        margin_y = 0.02
        left = 1 - button_width - margin_x
        bottom = margin_y
        ax_button = self.fig.add_axes([left, bottom, button_width, button_height])
        
        # Create the Next Step button
        self.button = Button(ax_button, 'Next Step', color='lightblue', hovercolor='blue')
        self.button.on_clicked(self.next_step)
        
        # Initial render
        self.render_current_state()
        
        plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    
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
            triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, 
                                             orientation=0, color='purple')
            self.ax_grid.add_patch(triangle)
            self.ax_grid.text(x + 0.5, y + 0.2, agent_id, ha='center', va='center', 
                              color='white', fontsize=8)
        
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_title(f"GridWorld Environment - Step {self.current_step}")
        
        # Display agent statistics
        self.display_agent_statistics()
    
    def display_agent_statistics(self):
        """Display performance metrics and statistics for all agents in a flex-wrap layout"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if not self.env.agent_instances:
            self.ax_stats.text(0.5, 0.5, "No agent statistics available", 
                                ha='center', va='center', transform=self.ax_stats.transAxes, fontsize=12)
            return
        
        # Add title
        self.ax_stats.text(0.5, 0.95, "AGENT PERFORMANCE METRICS", 
                            transform=self.ax_stats.transAxes, 
                            fontsize=12, fontweight='bold',
                            ha='center', va='top')
        
        # Calculate layout parameters
        total_cells = self.env.grid_size[0] * self.env.grid_size[1]
        agents = list(self.env.agent_instances.items())
        num_agents = len(agents)
        
        # Define box dimensions and spacing
        box_width = 0.30  # Width of each agent box (30% of total width)
        box_height = 0.70  # Height of each agent box (increased to fit content)
        margin_x = 0.02   # Horizontal margin between boxes
        margin_y = 0.08   # Vertical margin between rows
        start_y = 0.85    # Starting Y position (below title)
        
        # Calculate how many boxes fit per row
        boxes_per_row = int((1.0 - margin_x) / (box_width + margin_x))
        if boxes_per_row == 0:
            boxes_per_row = 1
        
        # Create agent statistics boxes
        for i, (agent_id, agent) in enumerate(agents):
            # Calculate position for this box (row-based with wrapping)
            row = i // boxes_per_row
            col = i % boxes_per_row
            
            # Calculate box position
            x_pos = margin_x + col * (box_width + margin_x)
            y_pos = start_y - row * (box_height + margin_y)
            
            # Get agent statistics
            basic_stats = agent.get_statistics_summary()
            explored_cells = self.env._calculate_explored_cells(agent)
            perf_metrics = agent.get_performance_metrics(
                total_steps=self.env.step_count,
                explored_cells=explored_cells,
                total_cells=total_cells
            )
            
            # Create compact statistics text for this agent
            stats_text = f"{agent_id}\n"
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
            
            # Draw box background
            box_rect = patches.Rectangle((x_pos, y_pos - box_height), box_width, box_height,
                                        facecolor='lightblue', edgecolor='navy', 
                                        linewidth=1, alpha=0.7, 
                                        transform=self.ax_stats.transAxes)
            self.ax_stats.add_patch(box_rect)
            
            # Add agent statistics text
            self.ax_stats.text(x_pos + box_width/2, y_pos - box_height/2, stats_text,
                                transform=self.ax_stats.transAxes,
                                fontfamily='monospace',
                                fontsize=8,
                                ha='center', va='center',
                                color='darkblue')
        
        # Set limits to prevent clipping
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