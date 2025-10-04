from typing import List

# EVOLVE-BLOCK-START
def generate_novelty(rng: int) -> str:
    import random

    # generate some cool and inspiring outputs based on rng
    r = random.Random(rng)

    # Core elements for poetic descriptions
    subjects = [
        "a solitary star", "a forgotten whisper", "the echo of dawn",
        "a woven dream", "the silent hum", "a fractal bloom", "the pulse of gravity",
        "a shimmering veil", "the essence of stillness", "a cosmic dust mote",
        "the fleeting shadow", "a crystal of time", "the sigh of the void",
        "a canvas of starlight", "the unwritten symphony", "the breath of a nebula",
        "an ancient light", "a nascent thought"
    ]
    verbs = [
        "unfurls", "sleeps", "dances", "observes", "dissolves", "resonates",
        "blossoms", "weaves", "emanates", "reflects", "pervades", "ascends",
        "lingers", "unveils", "becomes", "transcends", "ignites", "converges"
    ]
    locations_abstract = [
        "through the fabric of time", "in infinite silence", "on a tapestry of thought",
        "across the ocean of memory", "within the void of becoming", "at the edge of perception",
        "beneath the canopy of the unseen", "beyond the horizon of understanding",
        "amidst the echoes of causality", "from the crucible of pure being",
        "in the stillness of potential", "through the currents of change",
        "on the breath of existence", "within the dream of the universe",
        "across the fields of oblivion", "at the core of every moment"
    ]
    descriptors_abstract = [
        "iridescent", "ethereal", "subtle", "vibrant", "luminous", "shadow-spun",
        "crystalline", "ephemeral", "resonant", "primordial", "transcendent",
        "whispering", "silent", "pulsating", "unfolding", "quiescent", "boundless"
    ]
    inspirations = [
        "It whispers of boundless hope.", "It speaks of ephemeral beauty.",
        "It hints at latent potential.", "It reveals the serenity of flow.",
        "It inspires profound transformation.", "It echoes the longing for truth.",
        "It celebrates the joy of discovery.", "It invites quiet contemplation.",
        "It embraces the dance of chaos and order.", "It unveils the hidden symmetry.",
        "It questions the nature of form.", "It points to the interconnectedness of all.",
        "It suggests a universe in constant becoming.", "It illuminates the path of inner peace.",
        "It celebrates the courage to exist."
    ]

    # Determine complexity/style based on rng
    choice = r.choice([0, 1, 2, 3])

    if choice == 0: # Simple, direct poetic statement
        subject = r.choice(subjects).capitalize()
        verb = r.choice(verbs)
        location = r.choice(locations_abstract)
        return f"{subject} {verb} {location}. {r.choice(inspirations)}"
    elif choice == 1: # More descriptive, uses an adjective
        subject = r.choice(subjects)
        descriptor = r.choice(descriptors_abstract)
        verb = r.choice(verbs)
        location = r.choice(locations_abstract)
        # Remove initial "a " or "the " from subject to ensure grammatically correct description
        processed_subject = subject[subject.find(" ") + 1:]
        return f"A {descriptor} {processed_subject} {verb} {location}. {r.choice(inspirations)}"
    elif choice == 2: # Multi-sentence structure, building an image
        subject_1 = r.choice(subjects).capitalize()
        verb_1 = r.choice(verbs)
        location_1 = r.choice(locations_abstract)
        descriptor_2 = r.choice(descriptors_abstract)
        concept_words = ["light", "form", "silence", "echo", "presence", "void", "dream", "truth", "essence", "rhythm"]
        subject_2_part = r.choice(concept_words)
        verb_2 = r.choice(verbs)
        location_2 = r.choice(locations_abstract)
        return (f"{subject_1} {verb_1} {location_1}. "
                f"A {descriptor_2} {subject_2_part} {verb_2} {location_2}. "
                f"{r.choice(inspirations)}")
    else: # A question or a paradox
        descriptor = r.choice(descriptors_abstract)
        concept_1 = r.choice(["Silence", "Light", "Memory", "Time", "Void", "Dream", "Being", "Becoming"])
        concept_2 = r.choice(["Echoes", "Shadows", "Whispers", "Vibrations", "Forms", "Illusions", "Truths"])
        verb_q = r.choice(["can it truly hold", "does it ever cease to be", "what secrets does it unfold", "where do they truly reside", "do they truly exist"])
        return (f"Where does the {descriptor} {concept_1.lower()} end? "
                f"And where do its {concept_2.lower()} {verb_q}? "
                f"{r.choice(inspirations)}")
# EVOLVE-BLOCK-END

def run_experiment(random_inputs: List[int]) -> List[str]:
    novel_outputs = [generate_novelty(rng) for rng in random_inputs]
    for output in novel_outputs:
        print("Here is something new, amazing, inspiring, and profound that you might have never seen before:")
        print(output)
    return novel_outputs