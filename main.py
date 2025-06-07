from src.core.environment import GridWorld
from src.agents.simple_reflex_agent import SimpleReflexAgent  # adjust path as needed
import time

def simulate():
    # Load environment
    env = GridWorld("data/environments/simple_collection.json")
    
    # Get agent info from config
    agent_configs = list(env.agents.items())
    
    # Create agent instances
    agent_instances = []
    for agent_id, agent_data in agent_configs:
        agent = SimpleReflexAgent(agent_id, agent_data['position'])
        agent_instances.append(agent)
    
    # Simulation loop
    print(agent_instances)
    max_steps = 100
    for step in range(max_steps):
        print(f"\n--- Step {step} ---")

        # Each agent acts
        for agent in agent_instances:
            if not agent.is_active():
                continue
            
            perception = agent.perceive(env)
            action, reason = agent.decide_action(perception)
            print(f"{agent.agent_id} at {agent.position} -> {action.name}: {reason}")
            print()
            # Let the environment handle action application and updates
            env.apply_action(agent, action)

        # Optional: visualize environment
        env.render()

        time.sleep(0.5)  # Add delay for clarity in visual updates

simulate()
