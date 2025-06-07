from src.core.environment import GridWorld

if __name__ == "__main__":
    env = GridWorld("data/environments/simple_collection.json")
    env.render()