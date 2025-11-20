import pickle
import neat
from AI_Flappy_Bird import watch_winner

# Load NEAT config
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "Config_Feedforward.txt"
)

# Load saved best genome
with open("champion_bird.pkl", "rb") as f:
    best_genome = pickle.load(f)

print("[INFO] Loaded best genome!")

# Play champion bird
watch_winner(config, best_genome)
