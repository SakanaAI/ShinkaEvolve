from typing import List

# EVOLVE-BLOCK-START
def generate_novelty(rng: int) -> str:
    # add any relevant imports inside this function
    import random

    # Use the rng to seed the random number generator for reproducible diversity
    r = random.Random(rng)

    # Define lists of evocative elements for art description
    materials = [
        "obsidian shards", "luminescent quartz", "weathered copper", "polished chrome",
        "intertwined root systems", "suspended droplets of mercury", "crystalline ice",
        "woven light fibers", "silent cosmic dust", "petrified starlight",
        "liquid shadows", "translucent bio-resin", "gossamer threads of void",
        "singing sands", "frozen stardust", "resilient coral structures"
    ]
    forms = [
        "a spiraling helix", "an interlocking geodesic lattice", "a gently undulating wave",
        "a tessellated void", "a constellation of fleeting points", "a fractal canopy",
        "a blooming geode", "an ethereal labyrinth", "a pulsating nexus",
        "a suspended cascade", "an orbiting cluster of anomalies", "a silent monolith"
    ]
    modifiers = [
        "that seems to breathe with an unseen rhythm",
        "whispering tales of forgotten epochs",
        "glowing faintly with an inner warmth",
        "casting elongated, dancing shadows",
        "constantly reshaping its delicate structure",
        "resonating with a frequency beyond human hearing",
        "absorbing light into its core",
        "emanating a silent, ancient wisdom",
        "reflecting an infinite horizon",
        "slowly dissolving into pure energy"
    ]
    themes = [
        "The interconnectedness of all existence",
        "The ephemeral nature of time",
        "The silent dialogues between matter and void",
        "The enduring resilience of spirit",
        "The boundless potential within impermanence",
        "The beauty found in entropy and renewal",
        "The search for meaning in the vast unknown",
        "The cyclical journey of creation and decay",
        "The delicate balance between chaos and order",
        "The echoes of memory across dimensions"
    ]
    inspirations = [
        "It invites contemplation on personal journeys and transformations.",
        "It challenges perceptions of permanence and reality.",
        "It sparks a sense of wonder about the universe and our place within it.",
        "It encourages introspection on our deepest fears and aspirations.",
        "It suggests new ways of understanding growth, dissolution, and rebirth.",
        "It fosters a connection to the fundamental forces shaping our world.",
        "It provokes thought on the unseen patterns that govern existence."
    ]

    # Generate an evocative title
    adjectives = ["Ephemeral", "Resonant", "Chimerical", "Serene", "Volatile", "Quantum",
                  "Echoing", "Vast", "Infinite", "Sublime", "Transcendental", "Cryptic"]
    nouns = ["Whisper", "Nexus", "Labyrinth", "Genesis", "Echo", "Flux", "Ode", "Saga",
             "Continuum", "Fragment", "Paradox", "Vortex", "Ascension"]
    title_adj = r.choice(adjectives)
    title_noun = r.choice(nouns)
    title = f"\"The {title_adj} {title_noun}\""

    # Select elements based on the seed
    material = r.choice(materials)
    form = r.choice(forms)
    modifier = r.choice(modifiers)
    theme = r.choice(themes)
    inspiration = r.choice(inspirations)

    # Construct the art piece description
    output = f"Art Piece: {title}\n\n"
    output += f"Description: A monumental installation crafted from {material}, taking the form of {form}, {modifier}.\n"
    output += f"Conceptual Core: It explores {theme}.\n"
    output += f"Impact: {inspiration}\n\n"
    output += "Each element is meticulously arranged to evoke both profound introspection and expansive wonder, acting as a mirror to the soul's deepest mysteries and the universe's untold narratives."

    return output
# EVOLVE-BLOCK-END

def run_experiment(random_inputs: List[int]) -> List[str]:
    novel_outputs = [generate_novelty(rng) for rng in random_inputs]
    for output in novel_outputs:
        print("Here is something new, amazing, inspiring, and profound that you might have never seen before:")
        print(output)
    return novel_outputs