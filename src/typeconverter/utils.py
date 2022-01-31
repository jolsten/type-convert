import numpy as np

def bits_to_wordsize(size: np.uint8) -> np.uint8:
    for word_size in [8, 16, 32, 64]:
        if size <= word_size:
            return word_size
    return 64
