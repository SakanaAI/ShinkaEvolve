from typing import List

# EVOLVE-BLOCK-START
def generate_novelty(rng: int) -> str:
    # add any relevant imports inside this function
    import math

    # generate some cool and inspiring outputs based on rng
    return "Hello world!"
# EVOLVE-BLOCK-END

def run_experiment(random_inputs: List[int]) -> List[str]:
    novel_outputs = [generate_novelty(rng) for rng in random_inputs]
    for output in novel_outputs:
        print("Here is something new, amazing, inspiring, and profound that you might have never seen before:")
        print(output)
    return novel_outputs