#  AI Flappy Bird — Neuroevolution with NEAT  
*A machine-learning powered Flappy Bird agent using evolutionary neural networks.*

![Fitness Curve](fitness_curve.png)

## Project Overview

This project implements **Flappy Bird** with an **AI agent trained using NEAT**  
(NeuroEvolution of Augmenting Topologies).  

The AI **evolves its neural network topology and weights** over generations to learn a near-optimal strategy for surviving between pipes.

This project demonstrates:

- ✔ Neuroevolution (NEAT-Python)  
- ✔ Reward shaping & fitness engineering  
- ✔ Interpretable ML (network visualization + vision lines)  
- ✔ Fast training mode (no rendering)  
- ✔ Saved best genome (`champion_bird.pkl`)  
- ✔ ML-style training graph (`fitness_curve.png`)  
- ✔ Champion playback mode (`play_champion.py`)  

This is **not** a basic Flappy Bird clone — it’s a full **AI/ML systems project** showcasing model evolution, environment design, and visualization.

---

## Installation

Install dependencies:

```bash
pip install pygame neat-python matplotlib
```
## Run training
```bash
python AI_Flappy_Bird.py
```
## Play Back the Trained Champion

After training, the best genome is saved as:
```bash
champion_bird.pkl
```
To watch your evolved AI play:
```bash
python play_champion.py
```
The playback script loads the pickled genome and runs it through the game environment, just like an evaluation phase in RL.

## How the AI Learns (NEAT Overview)

This project uses **NEAT** (NeuroEvolution of Augmenting Topologies), an algorithm that evolves:

- Network weights  
- Activation functions  
- Hidden nodes  
- Network topology  

Over generations, NEAT selects better-performing genomes and mutates them to discover more optimal policies.

### References
- NEAT Python docs: https://neat-python.readthedocs.io  
- Original NEAT paper (Stanley & Miikkulainen):  
  https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf

---

## State Representation

The bird receives **four normalized inputs**:

| Input      | Description                          |
|------------|--------------------------------------|
| `y_norm`   | Bird's vertical position (0–1)        |
| `top_diff` | Distance to top pipe gap              |
| `bot_diff` | Distance to bottom pipe gap           |
| `vel_norm` | Bird's vertical velocity              |

These features create a compact **Markovian** state representation for stable learning.

---

## Reward Shaping (Fitness Function)

The fitness signal is carefully engineered to promote smooth, stable flight:

| Action / Behavior            | Reward / Penalty |
|------------------------------|------------------|
| Frame survived               | **+0.3**         |
| Pass a pipe                  | **+20**          |
| Hit ground/pipe              | **–1**           |
| Jump spamming                | **–0.03**        |
| Hugging screen top/bottom    | **–0.01**        |

This reward system prevents degenerate behaviors and accelerates convergence.

---

## Training Visualization

The script automatically generates:

### **`fitness_curve.png`**
Includes:
- Best fitness per generation  
- Average fitness  
- ±1 standard deviation band  
- Clean ML-style formatting  

Example title:  
**“NEAT Flappy Bird – Fitness Over Generations”**

### **`champion_bird.pkl`**
Serialized best genome after training  
(ignored by Git via `.gitignore`).

---

## Controls

### During Training
| Key | Action                          |
|-----|----------------------------------|
| **F** | Toggle fast mode (no rendering) |
| **D** | Toggle “vision” lines           |
| **N** | Toggle neural net panel         |
| **I** | Inspect inputs/outputs          |

### During Champion Playback
| Key | Action |
|-----|--------|
| **Q** | Quit  |

---

## Project Structure

```bash
AI-Flappy-Bird-GUH/
│── Images/imgs/              # Game sprites
│── AI_Flappy_Bird.py         # Main training + game logic
│── play_champion.py          # Playback script for champion genome
│── Config_Feedforward.txt    # NEAT configuration
│── champion_bird.pkl         # Saved best genome (ignored in Git)
│── fitness_curve.png         # Auto-generated training plot
│── README.md                 # Project documentation
│── .gitignore
```

## NEAT Configuration Highlights

### **Population & Evolution**
- `pop_size = 80`
- `fitness_criterion = max` (selects highest performers)
- `fitness_threshold` (early stopping when solved)

### **Mutation Behavior**
- `activation_mutate_rate`
- `activation_options = tanh, sigmoid, relu`
- `bias_mutate_rate`
- `weight_mutate_rate`
- `node_add_prob`, `conn_add_prob`

### **Speciation**
Preserves innovation by grouping similar genomes:

- `compatibility_threshold`
- `reset_on_extinction`
- `max_stagnation`

This prevents premature convergence and maintains diversity.

---

## Acknowledgements

- **NEAT-Python** by Matt Cooper & contributors  
- **NEAT algorithm** by Kenneth O. Stanley & Risto Miikkulainen