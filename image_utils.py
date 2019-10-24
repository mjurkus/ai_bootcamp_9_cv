import math

import matplotlib.pyplot as plt
from PIL import Image


def show_distances(
        dataset,
        target_index,
        similar_indices,
        distances,
        cols: int = 4,
        debug: bool = False,
):
    if cols >= len(distances) + 1:
        cols = len(distances) + 1
        rows = 1
    else:
        rows = math.ceil((len(distances) + 1) / cols)

    figsize = (3 * cols, 4 * rows) if debug else (3 * cols, 3 * rows)
    _, ax = plt.subplots(rows, cols, figsize=figsize)

    for i, (x, y, distance) in enumerate(
            zip(
                dataset.x[[target_index] + similar_indices],
                dataset.y[[target_index] + similar_indices],
                [0] + distances,
            )
    ):
        idx = (i // cols, i % cols) if rows > 1 else i % cols
        ax[idx].axis("off")
        ax[idx].imshow(Image.open(x))
        title = f"Label: {y}\nShape: {x.shape}\n" if debug else f"{y}\n{distance:.2f}"
        ax[idx].set_title(title)
