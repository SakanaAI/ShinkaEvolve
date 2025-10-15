from typing import List

# EVOLVE-BLOCK-START
def generate_novelty(rng: int) -> str:
    import random
    import math

    # Use rng to seed a local random number generator
    r = random.Random(rng)

    # --- Configuration Parameters (tuned for diversity, meaning, inspiration) ---
    min_width, max_width = 40, 70
    min_height, max_height = 8, 20

    # Pools of evocative words and phrases
    adjectives = ["Ephemeral", "Quantum", "Echoing", "Luminous", "Subtle", "Vast", "Infinite", "Kinetic", "Translucent", "Resonant", "Invisible", "Digital", "Organic", "Celestial", "Silent", "Verdant", "Cryptic", "Synchronous", "Nebulous", "Harmonic"]
    nouns = ["Connections", "Whispers", "Architectures", "Flux", "Dreams", "Resonance", "Chambers", "Vortex", "Symphonies", "Mirrors", "Gradients", "Fractions", "Horizons", "Fabric", "Essence", "Paradox", "Continuum", "Emanations", "Chronicles", "Vestiges"]
    verbs_present = ["unravel", "illuminate", "echo", "bridge", "invite contemplation on", "reconfigure", "manifest", "traverse", "reveal", "harmonize", "dissolve", "synthesize", "evoke", "reverberate", "align"]
    objectives = ["the fabric of perception", "forgotten truths", "the whispers of the cosmos", "the chasm between thought and form", "existence", "the boundaries of reality", "inner landscapes", "unseen frequencies", "the dance of time", "the void's embrace", "the passage of shadows", "collective memory", "the origins of thought", "the silence of beginnings", "the pulse of creation"]
    mediums = ["a holographic installation", "a kinetic sculpture", "an auditory landscape", "a generative algorithm", "a projected light field", "an interactive experience", "a series of conceptual diagrams", "a temporal distortion chamber", "a bio-luminescent garden", "a spectral tapestry", "a conceptual artifact", "an environmental tableau"]
    materials = ["recycled light", "translucent data streams", "resonant frequencies", "programmable matter", "unseen energies", "crystallized emotions", "shifting perspectives", "gravitational waves", "entropic dust", "synthetic aether", "lunar silicate", "chimerical alloys", "vaporized thought", "cosmic dust motes", "fractal filaments"]

    symbol_sets = {
        "blocks": "█▓▒░",
        "geometric": "─│╱╲╳━┃╭╮╰╯═║╔╗╚╝╠╣╦╩╬",
        "stars_dots": "✧･ﾟ:*✦⋆.·°★☆.",
        "abstract": "~-=+*#@.",
        "waves": "/\\|_. -",
        "fluid": "∽∿≀∸∫∮∱∲",
        "mixed": "█▓▒░+-*#~. ", # Includes space for sparseness
        "light_dark": "░▒▓█"
    }

    # --- Generate Art Piece ---

    # 1. Title Generation
    title_parts = [r.choice(adjectives), r.choice(nouns), r.choice(adjectives), r.choice(nouns)]
    # Randomly choose a title structure for diversity and aesthetics
    if r.random() < 0.5:
        title = f"{title_parts[0].capitalize()} {title_parts[1].capitalize()}: {title_parts[2].capitalize()} {title_parts[3].capitalize()}"
    else:
        title = f"The {title_parts[0].capitalize()} {title_parts[1].capitalize()} of {title_parts[2].capitalize()} {title_parts[3].capitalize()}"

    # 2. Conceptual Description Generation
    medium = r.choice(mediums)
    material = r.choice(materials)
    verb = r.choice(verbs_present)
    objective = r.choice(objectives)

    description = (
        f"This piece, '{title}', is {medium} crafted from {material}. "
        f"It seeks to {verb} {objective}, inviting observers into a state of profound reflection."
    )

    # Add a secondary philosophical sentence for deeper meaning
    if r.random() < 0.7:
        secondary_thoughts = [
            f"It challenges our understanding of {r.choice(nouns).lower()} and the nature of {r.choice(nouns).lower()}.",
            f"The interplay of {r.choice(adjectives).lower()} forces reveals the {r.choice(adjectives).lower()} truth of {r.choice(nouns).lower()}.",
            f"Through {r.choice(adjectives).lower()} {r.choice(nouns).lower()}, it explores the boundaries of {r.choice(nouns).lower()}."
        ]
        description += " " + r.choice(secondary_thoughts)

    # 3. Visual Representation Generation (Procedural ASCII/Unicode Art)
    width = r.randint(min_width, max_width)
    height = r.randint(min_height, max_height)

    chosen_symbol_set = r.choice(list(symbol_sets.values()))

    visual_art_lines = []

    # Choose a pattern type with weighted probabilities for diversity
    pattern_type = r.choices(
        ['wave', 'noise', 'concentric', 'perlin_like', 'gradient_diagonal', 'cellular_automata_like'],
        weights=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1],
        k=1
    )[0]

    if pattern_type == 'wave':
        freq_x = r.uniform(0.05, 0.25)
        freq_y = r.uniform(0.05, 0.25)
        amplitude_multiplier = r.uniform(0.5, 2.0)
        phase_x = r.uniform(0, math.pi * 2)
        phase_y = r.uniform(0, math.pi * 2)

        for y in range(height):
            line = []
            for x in range(width):
                val = math.sin(x * freq_x + phase_x) + math.cos(y * freq_y + phase_y)
                # Normalize val to [0, 1] then scale to symbol set length
                normalized_val = (val + amplitude_multiplier * 2) / (amplitude_multiplier * 4)
                char_idx = int(normalized_val * (len(chosen_symbol_set) - 1))
                char_idx = max(0, min(len(chosen_symbol_set) - 1, char_idx))
                line.append(chosen_symbol_set[char_idx])
            visual_art_lines.append("".join(line))

    elif pattern_type == 'noise':
        density_threshold = r.uniform(0.3, 0.7)
        sparse_char = r.choice([' ', '.', '-', 'o', '_'])
        for y in range(height):
            line = []
            for x in range(width):
                if r.random() < density_threshold:
                    line.append(r.choice(chosen_symbol_set))
                else:
                    line.append(sparse_char)
            visual_art_lines.append("".join(line))

    elif pattern_type == 'concentric':
        center_x, center_y = r.randint(0, width - 1), r.randint(0, height - 1)
        step_size = r.uniform(0.5, 2.5)
        for y in range(height):
            line = []
            for x in range(width):
                dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                char_idx = int((dist / step_size) % len(chosen_symbol_set))
                line.append(chosen_symbol_set[char_idx])
            visual_art_lines.append("".join(line))

    elif pattern_type == 'perlin_like':
        # Simple approximation of smooth noise using combined sin/cos waves
        scale_x = r.uniform(0.1, 0.5)
        scale_y = r.uniform(0.1, 0.5)
        offset_x = r.uniform(0, 100)
        offset_y = r.uniform(0, 100)
        for y in range(height):
            line = []
            for x in range(width):
                val = math.sin((x + offset_x) * scale_x) + math.cos((y + offset_y) * scale_y)
                normalized_val = (val + 2) / 4
                char_idx = int(normalized_val * (len(chosen_symbol_set) - 1))
                char_idx = max(0, min(len(chosen_symbol_set) - 1, char_idx))
                line.append(chosen_symbol_set[char_idx])
            visual_art_lines.append("".join(line))

    elif pattern_type == 'gradient_diagonal':
        char_choices = list(chosen_symbol_set)
        if len(char_choices) < 2: # Ensure at least two chars for gradient effect
            char_choices = list(" .-") if chosen_symbol_set == " " else list(chosen_symbol_set + " .")

        start_char = r.choice(char_choices)
        end_char = r.choice([c for c in char_choices if c != start_char]) if len(char_choices) > 1 else start_char

        for y in range(height):
            line = []
            for x in range(width):
                # Diagonal gradient based on x + y
                gradient_val = (x + y) / (width + height - 2)

                if len(char_choices) > 1:
                    # Map gradient_val to index in char_choices
                    char_idx = int(gradient_val * (len(char_choices) - 1))
                    line.append(char_choices[char_idx])
                else: # Fallback if only one char available
                    line.append(start_char)
            visual_art_lines.append("".join(line))

    elif pattern_type == 'cellular_automata_like':
        # Simple rule-based pattern simulating some CA characteristics
        rule_param_x = r.randint(1, 4)
        rule_param_y = r.randint(1, 4)
        rule_offset = r.randint(0, 100)

        char_a = r.choice(chosen_symbol_set)
        char_b = r.choice([c for c in chosen_symbol_set if c != char_a]) if len(chosen_symbol_set) > 1 else char_a

        for y in range(height):
            line = []
            for x in range(width):
                # A simple rule based on coordinates and random parameters
                val = (x // rule_param_x + y // rule_param_y + rule_offset) % 2
                line.append(char_a if val == 0 else char_b)
            visual_art_lines.append("".join(line))


    # --- Assemble Final Output ---
    output = []
    output.append(f"Title: {title}\n")
    output.append("--- Conceptual Art Piece ---")
    output.append(description)
    output.append("\n--- Symbolic Representation ---")
    output.append("\n".join(visual_art_lines))
    output.append("\n--- End of Piece ---")

    return "\n".join(output)
# EVOLVE-BLOCK-END

def run_experiment(random_inputs: List[int]) -> List[str]:
    novel_outputs = [generate_novelty(rng) for rng in random_inputs]
    for output in novel_outputs:
        print("Here is something new, amazing, inspiring, and profound that you might have never seen before:")
        print(output)
    return novel_outputs
