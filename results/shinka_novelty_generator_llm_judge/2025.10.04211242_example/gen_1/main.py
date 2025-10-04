from typing import List

# EVOLVE-BLOCK-START
def generate_novelty(rng: int) -> str:
    # Add relevant imports inside this function
    import random

    # Use rng to seed the random generator for deterministic diversity
    rand = random.Random(rng)

    art_forms = [
        "A hyper-realistic painting",
        "A kinetic sculpture",
        "A generative digital art piece",
        "An architectural marvel",
        "A sonic installation",
        "A living botanical sculpture",
        "A holographic projection",
        "A performance art piece",
        "A poetic narrative woven into a landscape",
        "An ephemeral light sculpture",
        "A conceptual art piece",
        "A virtual reality experience",
        "A bio-luminescent garden",
    ]

    primary_subjects = [
        "the cosmic dance of nebulae",
        "the quiet resilience of ancient trees",
        "the fleeting nature of human memory",
        "the intricate beauty of a microscopic world",
        "the silent language of glaciers",
        "the boundless potential of future cities",
        "the paradoxical unity of chaos and order",
        "the journey of a single drop of water",
        "the echoes of forgotten civilizations",
        "the dreams of dormant volcanoes",
        "the interconnectedness of all living things",
        "the infinite possibilities of an unwritten future",
    ]

    styles_materials = [
        "rendered in luminescent bioluminescent pigments that shift with the observer's gaze.",
        "crafted from translucent, self-repairing polymers and suspended by magnetic fields.",
        "comprising interwoven threads of starlight and solidified whispers.",
        "formed by the interplay of shadow and light, constantly reconfiguring itself.",
        "composed of reclaimed urban materials, vibrating with an unseen energy.",
        "depicting intricate fractal patterns that slowly evolve over millennia.",
        "using sound waves sculpted into tangible forms, audible only to the attuned.",
        "built with organic, self-growing structures that pulse with a gentle rhythm.",
        "a minimalist arrangement of mirrors reflecting infinite possibilities.",
        "a complex tapestry woven from the data streams of a global consciousness.",
        "a dynamic system reacting to ambient environmental data.",
        "a shimmering vortex of recontextualized historical artifacts.",
    ]

    emotional_tones = [
        "It evokes a sense of profound serenity and timelessness.",
        "It inspires awe at the vastness of existence and our place within it.",
        "It provokes introspection on the cycles of creation and decay.",
        "It radiates a hopeful energy, celebrating resilience and renewal.",
        "It challenges perceptions, inviting the viewer to find meaning in ambiguity.",
        "It stirs a melancholic beauty, reminiscent of moments lost and found.",
        "It pulses with a vibrant chaos, a testament to unbridled energy.",
        "It whispers secrets of deep interconnectedness, fostering empathy.",
        "It offers a glimpse into an imagined future, both utopian and cautionary.",
        "It grounds the observer, connecting them to the primal forces of the earth.",
        "It encourages a re-evaluation of human impact on nature.",
        "It celebrates the silent grandeur of the cosmos.",
    ]

    # Select components using the seeded random generator
    form = rand.choice(art_forms)
    subject = rand.choice(primary_subjects)
    style_material = rand.choice(styles_materials)
    tone = rand.choice(emotional_tones)

    # Generate a unique title based on RNG for additional diversity
    adjectives = ["Ephemeral", "Quantum", "Whispering", "Infinite", "Chrono", "Luminous", "Verdant", "Silent", "Echoing", "Fractal", "Harmonic", "Celestial"]
    nouns = ["Symphony", "Gateway", "Reflections", "Nexus", "Dreamscape", "Harbinger", "Chrysalis", "Flux", "Overture", "Confluence", "Resonance", "Enigma"]

    # Extract the core subject from the 'primary_subjects' list for the title
    # e.g., "the cosmic dance of nebulae" becomes "cosmic dance of nebulae"
    subject_for_title = subject.replace("the ", "", 1) if subject.startswith("the ") else subject
    title = f"{rand.choice(adjectives)} {rand.choice(nouns)} of {subject_for_title}"

    # Assemble the artwork description
    description = (
        f"Title: {title}\n\n"
        f"{form}: Depicting {subject}, this piece is {style_material} "
        f"{tone}"
    )

    return description
# EVOLVE-BLOCK-END

def run_experiment(random_inputs: List[int]) -> List[str]:
    novel_outputs = [generate_novelty(rng) for rng in random_inputs]
    for output in novel_outputs:
        print("Here is something new, amazing, inspiring, and profound that you might have never seen before:")
        print(output)
    return novel_outputs