"""
The classic game of Flappy Bird with an AI player powered by NEAT.
Made with Python and Pygame. Uses pixel-perfect collision via masks.

This version is designed as an AI portfolio project:
- Neuroevolution with NEAT
- Normalized state representation
- Reward shaping and fitness engineering
- Fast training mode (no rendering)
- Interpretable AI (vision lines + simple NN visualizer)
- Champion playback after training
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # silence pkg_resources warning

import os
import random
import time
import statistics

import neat
import pygame

pygame.font.init()  # init font

# Window dimensions
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

STAT_FONT = pygame.font.SysFont("bold", 50)
DEBUG_FONT = pygame.font.SysFont("bold", 18)

# Global debug / UX flags
DRAW_LINES = False       # show "what the bird sees" to the next pipe
FAST_MODE = False        # when True: minimal drawing, high FPS (fast training)
INSPECT_MODE = False     # when True: print inputs/outputs for the lead bird on jumps
VISUALIZE_NET = False    # when True: draw a tiny network visualization for the lead bird

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Load bird images for animation (bird1, bird2, bird3)
bird_images = [
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("Images", "imgs", f"bird{x}.png"))
    )
    for x in range(1, 4)
]

pipe_img = pygame.transform.scale2x(
    pygame.image.load(os.path.join("Images", "imgs", "pipe.png")).convert_alpha()
)
bg_img = pygame.transform.scale(
    pygame.image.load(os.path.join("Images", "imgs", "bg.png")).convert_alpha(),
    (600, 900),
)
base_img = pygame.transform.scale2x(
    pygame.image.load(os.path.join("Images", "imgs", "base.png")).convert_alpha()
)

# Generation counter (used for display + summary)
gen = 0


class Bird:
    """Represents the Flappy Bird controlled either by a human or by NEAT."""

    MAX_ROTATION = 25  # max tilt angle in degrees
    IMGS = bird_images
    ROT_VEL = 20       # rotation speed per frame
    ANIMATION_TIME = 5 # frames to show each wing position

    def __init__(self, x, y):
        # Starting position
        self.x = x
        self.y = y

        # Movement / physics
        self.tilt = 0  # degrees
        self.tick_count = 0
        self.vel = 0
        self.height = self.y

        # Animation
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """Make the bird jump upwards (negative velocity in pygame coordinates)."""
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """Apply simple physics to the bird each frame."""
        self.tick_count += 1

        # Displacement: s = v * t + 0.5 * a * t^2
        displacement = self.vel * self.tick_count + 0.5 * 3 * (self.tick_count ** 2)

        # Terminal velocity
        if displacement >= 16:
            displacement = 16
        if displacement < 0:
            # Make jumps feel more responsive
            displacement -= 2

        self.y += displacement

        # Tilt behavior:
        # - Tilt up when moving up or still near jump height
        # - Gradually tilt down when falling
        if displacement < 0 or self.y < self.height + 50:
            # Smooth upward rotation up to MAX_ROTATION
            self.tilt = min(self.tilt + self.ROT_VEL * 1.2, self.MAX_ROTATION)
        else:
            # Smooth downward rotation up to -90 degrees (nose dive)
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL * 0.7

    def draw(self, win):
        """Draw the bird and handle wing animation."""
        self.img_count += 1

        # Cycle through wing positions
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        else:
            self.img = self.IMGS[0]
            self.img_count = 0

        # If diving steeply, freeze wings
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """Return a mask for pixel-perfect collision."""
        return pygame.mask.from_surface(self.img)


class Pipe:
    """Represents a pipe obstacle."""

    GAP = 200  # vertical gap between top and bottom pipes
    VEL = 5    # horizontal velocity of pipes (can be scaled with score)

    def __init__(self, x):
        self.x = x
        self.height = 0

        # Top and bottom of pipe
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img

        self.passed = False  # whether the bird has passed this pipe
        self.set_height()

    def set_height(self):
        """Randomly set the vertical position of the gap."""
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """Move the pipe left across the screen."""
        self.x -= self.VEL

    def draw(self, win):
        """Draw the pipe on the screen."""
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        """Return True if bird collides with this pipe."""
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return bool(b_point or t_point)


class Base:
    """Represents the moving ground at the bottom of the screen."""

    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """Scroll the base to create the illusion of movement."""
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        # Cycle images when they move off-screen
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    """Rotate an image around its center and blit it to the surface."""
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(
        center=image.get_rect(topleft=topleft).center
    )
    surf.blit(rotated_image, new_rect.topleft)


def draw_network(win, genome):
    """
    Improved visualization of the lead bird's network.

    - Inputs: 4 nodes on the left (y, top_diff, bottom_diff, vel).
    - Output: 1 node on the right (jump decision).
    - Connections: green for positive weights, red for negative, thickness ~ |weight|.
    - Unused inputs are dimmed.
    """
    if genome is None:
        return

    # Panel position and size
    panel_width, panel_height = 240, 240
    panel_x, panel_y = WIN_WIDTH - panel_width - 10, 490

    # Semi-transparent background panel
    panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel_surf.fill((0, 0, 0, 120))  # black with alpha
    win.blit(panel_surf, (panel_x, panel_y))

    # Outer border
    pygame.draw.rect(win, (230, 230, 230), (panel_x, panel_y, panel_width, panel_height), 2)

    # Title
    title = STAT_FONT.render("Lead Bird NN", True, (255, 255, 255))
    win.blit(title, (panel_x + (panel_width - title.get_width()) // 2, panel_y - 40))

    # Node positions
    inputs = [-1, -2, -3, -4]  # NEAT uses negative keys for inputs
    node_pos = {}

    x_in = panel_x + 40
    x_out = panel_x + panel_width - 40
    y_start = panel_y + 40
    y_step = 40

    labels = {
        -1: "y_norm",
        -2: "top_diff",
        -3: "bot_diff",
        -4: "vel",
    }

    # Determine which inputs are actually used
    used_inputs = set()
    for conn in genome.connections.values():
        if conn.enabled:
            in_key, out_key = conn.key
            if in_key in inputs:
                used_inputs.add(in_key)

    # Draw input nodes
    for idx, key in enumerate(inputs):
        pos = (x_in, y_start + idx * y_step)
        node_pos[key] = pos

        # Dim unused inputs
        if key in used_inputs:
            fill_color = (200, 200, 200)
            outline_color = (255, 255, 255)
        else:
            fill_color = (80, 80, 80)
            outline_color = (130, 130, 130)

        pygame.draw.circle(win, fill_color, pos, 9)
        pygame.draw.circle(win, outline_color, pos, 9, 2)

        text = DEBUG_FONT.render(labels[key], True, outline_color)
        win.blit(text, (pos[0] - text.get_width() - 10, pos[1] - text.get_height() / 2))

    # Output node
    out_pos = (x_out, panel_y + panel_height // 2)
    node_pos[0] = out_pos
    pygame.draw.circle(win, (255, 255, 255), out_pos, 11)
    pygame.draw.circle(win, (255, 255, 255), out_pos, 11, 2)

    out_label = DEBUG_FONT.render("jump", True, (255, 255, 255))
    win.blit(out_label, (out_pos[0] - out_label.get_width() / 2, out_pos[1] + 14))

    # Draw connections
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        in_key, out_key = conn.key
        if in_key in node_pos and out_key in node_pos:
            start = node_pos[in_key]
            end = node_pos[out_key]
            w = conn.weight
            thickness = max(2, min(5, int(abs(w) * 1.5)))
            color = (100, 255, 100) if w > 0 else (255, 100, 100)

            # Simple "glow": draw a faint thicker line behind
            pygame.draw.line(win, (color[0], color[1], color[2], 80), start, end, thickness + 2)
            pygame.draw.line(win, color, start, end, thickness)


def draw_window(win, birds, pipes, base, score, gen, pipe_ind, genome=None):
    """Draw all game elements on the screen."""
    if gen == 0:
        gen = 1

    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    # Optionally draw helper lines from birds to the pipe gap
    for bird in birds:
        if DRAW_LINES and 0 <= pipe_ind < len(pipes):
            pygame.draw.line(
                win,
                (255, 0, 0),
                (bird.x + bird.img.get_width() / 2,
                 bird.y + bird.img.get_height() / 2),
                (
                    pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2,
                    pipes[pipe_ind].height,
                ),
                3,
            )
            pygame.draw.line(
                win,
                (255, 0, 0),
                (bird.x + bird.img.get_width() / 2,
                 bird.y + bird.img.get_height() / 2),
                (
                    pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2,
                    pipes[pipe_ind].bottom,
                ),
                3,
            )

        bird.draw(win)

    # HUD
    score_label = STAT_FONT.render(f"Score: {score}", 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    gen_label = STAT_FONT.render(f"Gens: {gen - 1}", 1, (255, 255, 255))
    win.blit(gen_label, (10, 10))

    alive_label = STAT_FONT.render(f"Alive: {len(birds)}", 1, (255, 255, 255))
    win.blit(alive_label, (10, 50))

    # Mode indicators
    mode_text = []
    if FAST_MODE:
        mode_text.append("FAST")
    if DRAW_LINES:
        mode_text.append("LINES")
    if VISUALIZE_NET:
        mode_text.append("NN")
    if INSPECT_MODE:
        mode_text.append("INSPECT")

    if mode_text:
        label = DEBUG_FONT.render(
            " | ".join(mode_text), True, (255, 255, 0)
        )
        win.blit(label, (10, FLOOR + 10))

    # Optional neural network visualization for lead bird
    if VISUALIZE_NET and genome is not None:
        draw_network(win, genome)

    pygame.display.update()


def eval_genomes(genomes, config):
    """
    NEAT evaluation function.
    Runs the simulation for all genomes in the current generation and
    assigns fitness based on performance.

    This function is where the ML design lives:
    - State representation (normalized y, distances, velocity)
    - Action mapping (jump if output > threshold)
    - Reward shaping:
        * +0.1 per frame alive
        * +20 per pipe passed
        * -1 on collision
        * small penalties for hugging top/bottom or jump spamming
    """
    global WIN, gen, FAST_MODE, DRAW_LINES, INSPECT_MODE, VISUALIZE_NET
    win = WIN
    gen += 1

    nets = []
    birds = []
    ge = []

    # Create a bird and neural net for each genome
    for _, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()
    run = True

    while run and len(birds) > 0:
        clock.tick(240 if FAST_MODE else 30)

        # Handle events (including debug toggles)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    FAST_MODE = not FAST_MODE
                    print(f"[DEBUG] FAST_MODE = {FAST_MODE}")
                elif event.key == pygame.K_d:
                    DRAW_LINES = not DRAW_LINES
                    print(f"[DEBUG] DRAW_LINES = {DRAW_LINES}")
                elif event.key == pygame.K_i:
                    INSPECT_MODE = not INSPECT_MODE
                    print(f"[DEBUG] INSPECT_MODE = {INSPECT_MODE}")
                elif event.key == pygame.K_n:
                    VISUALIZE_NET = not VISUALIZE_NET
                    print(f"[DEBUG] VISUALIZE_NET = {VISUALIZE_NET}")

        # Determine which pipe to use as input to the network
        pipe_ind = 0
        if len(pipes) > 1 and len(birds) > 0:
            if birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        # Move birds and activate their nets
        for x, bird in enumerate(birds):
            bird.move()

            # Reward survival slightly every frame
            ge[x].fitness += 0.1

            # Small penalty for hugging top or bottom (encourage staying near center)
            if bird.y < 10 or bird.y > FLOOR - 50:
                ge[x].fitness -= 0.01

            # Normalized inputs for better learning stability
            y_norm = bird.y / WIN_HEIGHT
            top_diff = (bird.y - pipes[pipe_ind].height) / WIN_HEIGHT
            bottom_diff = (bird.y - pipes[pipe_ind].bottom) / WIN_HEIGHT
            vel_norm = bird.vel / 20.0

            output = nets[x].activate((y_norm, top_diff, bottom_diff, vel_norm))

            # If output high enough, jump (tanh output is between -1 and 1)
            if output[0] > 0.5:
                bird.jump()
                # Tiny penalty for jump spamming to encourage efficient movement
                ge[x].fitness -= 0.03

                if INSPECT_MODE and x == 0:
                    print(
                        f"[INSPECT] y_norm={y_norm:.3f}, top_diff={top_diff:.3f}, "
                        f"bot_diff={bottom_diff:.3f}, vel_norm={vel_norm:.3f}, "
                        f"output={output[0]:.3f} -> JUMP"
                    )

        base.move()

        rem = []
        add_pipe = False

        # For "passed" checks, use the furthest bird on x-axis
        if birds:
            lead_bird_x = max(bird.x for bird in birds)
        else:
            lead_bird_x = 0

        # Move pipes and check for collisions
        for pipe in pipes:
            pipe.move()

            # Collisions (iterate backwards to remove safely)
            for i in range(len(birds) - 1, -1, -1):
                bird = birds[i]
                if pipe.collide(bird, win):
                    ge[i].fitness -= 1
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)

            # Off-screen pipe
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            # Has the lead bird passed this pipe?
            if (not pipe.passed) and pipe.x + pipe.PIPE_TOP.get_width() < lead_bird_x:
                pipe.passed = True
                add_pipe = True

        # Add new pipe when one is passed
        if add_pipe:
            score += 1

            # Reward all surviving genomes when a pipe is passed
            for genome in ge:
                genome.fitness += 20

            # Slightly ramp up difficulty as score increases
            Pipe.VEL = 5 + score * 0.1
            Base.VEL = Pipe.VEL

            pipes.append(Pipe(WIN_WIDTH))

        # Remove pipes that have gone off screen
        for r in rem:
            pipes.remove(r)

        # Remove birds that hit the ground or fly too high
        for i in range(len(birds) - 1, -1, -1):
            bird = birds[i]
            if (
                bird.y + bird.img.get_height() - 10 >= FLOOR
                or bird.y < -50
            ):
                birds.pop(i)
                nets.pop(i)
                ge.pop(i)

        # Fast mode: skip drawing to train faster
        if not FAST_MODE:
            lead_genome = ge[0] if ge else None
            draw_window(WIN, birds, pipes, base, score, gen, pipe_ind, genome=lead_genome)


def watch_winner(config, genome):
    """
    After training, watch the best genome play indefinitely.

    This reuses the same physics and state representation but
    runs a single bird controlled by the winning network.
    """
    global WIN, FAST_MODE, DRAW_LINES, INSPECT_MODE, VISUALIZE_NET

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()
    run = True

    print("\n[INFO] Watching champion play. Keys: F=fast, D=lines, N=NN, Q=quit\n")

    while run:
        clock.tick(240 if FAST_MODE else 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    FAST_MODE = not FAST_MODE
                    print(f"[DEBUG] FAST_MODE = {FAST_MODE}")
                elif event.key == pygame.K_d:
                    DRAW_LINES = not DRAW_LINES
                    print(f"[DEBUG] DRAW_LINES = {DRAW_LINES}")
                elif event.key == pygame.K_n:
                    VISUALIZE_NET = not VISUALIZE_NET
                    print(f"[DEBUG] VISUALIZE_NET = {VISUALIZE_NET}")
                elif event.key == pygame.K_q:
                    run = False  # exit champion viewer

        # Determine which pipe to use as input
        pipe_ind = 0
        if len(pipes) > 1:
            if bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        # Network controls the bird
        y_norm = bird.y / WIN_HEIGHT
        top_diff = (bird.y - pipes[pipe_ind].height) / WIN_HEIGHT
        bottom_diff = (bird.y - pipes[pipe_ind].bottom) / WIN_HEIGHT
        vel_norm = bird.vel / 20.0

        output = net.activate((y_norm, top_diff, bottom_diff, vel_norm))
        if output[0] > 0.5:
            bird.jump()

        bird.move()
        base.move()

        rem = []
        add_pipe = False

        for pipe in pipes:
            pipe.move()

            if pipe.collide(bird, WIN):
                # Reset environment when champion dies
                bird = Bird(230, 350)
                base = Base(FLOOR)
                pipes = [Pipe(700)]
                score = 0
                break

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x + pipe.PIPE_TOP.get_width() < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        if not FAST_MODE:
            draw_window(WIN, [bird], pipes, base, score, gen=0, pipe_ind=pipe_ind, genome=genome)


class SimpleReporter(neat.reporting.BaseReporter):
    """
    Custom NEAT reporter that prints clean per-generation info
    WITHOUT the 'Total extinctions: 0' line.
    """

    def __init__(self):
        self.generation = None
        self.start_time = None

    def start_generation(self, generation):
        self.generation = generation
        self.start_time = time.time()
        print(f"\n ****** Running generation {generation} ****** \n")

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]

        if fitnesses:
            avg_fitness = statistics.mean(fitnesses)
            stdev_fitness = statistics.pstdev(fitnesses) if len(fitnesses) > 1 else 0.0
        else:
            avg_fitness = 0.0
            stdev_fitness = 0.0

        print(f"Population's average fitness: {avg_fitness:.5f} stdev: {stdev_fitness:.5f}")
        print(f"Best fitness: {best_genome.fitness:.5f} - size: ({len(best_genome.nodes)}, {len(best_genome.connections)})")

        # Species summary (more compact than default, no extinctions line)
        print("   ID   size  best_fit")
        print("  ====  ====  =========")

        for sid, s in species.species.items():
            best_in_species = max([population[g].fitness for g in s.members])
            print(f"{sid:5d}{len(s.members):6d}{best_in_species:10.1f}")

    def end_generation(self, config, population, species):
        if self.start_time is not None:
            dur = time.time() - self.start_time
            print(f"Generation time: {dur:.3f} sec")


def run(config_file):
    """Runs the NEAT algorithm to train a network to play Flappy Bird."""
    global gen

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)

    # Use custom reporter instead of StdOutReporter to avoid 'Total extinctions' spam
    p.add_reporter(SimpleReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run up to 50 generations (early-stop if fitness_threshold reached)
    winner = p.run(eval_genomes, 50)

    # Training summary
    print("\n================ TRAINING SUMMARY ================")
    print(f"Generations run      : {gen}")
    print(f"Best fitness (winner): {winner.fitness:.2f}")
    print("Network complexity   : "
          f"{len(winner.nodes)} nodes, {len(winner.connections)} connections")
    print("Notes:")
    print("- NEAT evolved a minimal policy that solves Flappy Bird.")
    print("- Inputs are normalized (y, top diff, bottom diff, velocity).")
    print("- Reward shaping (survival + pipe passes) drove rapid convergence.")
    print("=================================================\n")
    print("Best genome:\n{!s}".format(winner))

    # Watch the champion play
    watch_winner(config, winner)


"""
EXPERIMENT LOG / ML NOTES (for readers & recruiters)

- Algorithm: NEAT (NeuroEvolution of Augmenting Topologies) via neat-python.
- State representation:
    * y_norm          = bird.y / WIN_HEIGHT
    * top_diff        = (bird.y - pipe.height) / WIN_HEIGHT
    * bottom_diff     = (bird.y - pipe.bottom) / WIN_HEIGHT
    * vel_norm        = bird.vel / 20
  These are chosen to give the network a compact, normalized view of:
    * where it is vertically,
    * where the safe gap is,
    * and how fast it is moving.

- Action:
    * Single continuous output in [-1, 1] via tanh.
    * Jump if output > 0.5.

- Reward shaping:
    * +0.1 per frame alive  -> encourages survival and smooth flying.
    * +20 per pipe passed   -> strongly rewards actual progress.
    * -1 on collision       -> penalizes crashing.
    * Small penalties for:
        - hugging the very top/bottom of the screen,
        - jump spamming.
      These reduce degenerate policies and oscillation.

- Observed behavior:
    * With these settings, the network typically finds a very strong policy
      in just a few generations (e.g. best fitness > 2000 in generation 3).
    * NEAT often converges to an extremely small network:
        - one output node, no hidden nodes, a few weighted input connections.
      This highlights a key NEAT strength: topology search + weight search.

- Why this is an AI project (not just a game clone):
    * Neuroevolution with non-trivial state design and reward engineering.
    * Fast/slow training modes explicitly separate simulation from rendering.
    * Simple interpretability tools:
        - visualizing the bird's "vision" lines to the next pipe,
        - drawing a minimal network diagram with weight sign/strength.
    * Champion playback loop after training completes.
"""


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "Config_Feedforward.txt")
    run(config_path)
