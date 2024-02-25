import torch
import numpy as np
import matplotlib.pyplot as plt

def find_a_in_b(a, b, inv=False):
    condition = (a[0][:, None] == b[0].view(1, -1)) & (a[1][:, None] == b[1].view(1, -1))

    if inv:
        indices = torch.where(condition.sum(dim=0)==0)[0]
    else:
        indices = torch.where(condition)[1]

    # indicies are relative to b
    return indices



def plot_circle_grid(rows, cols, filled_indices, title='visualization'):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Calculate the size of each cell
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows

    # Plot filled circles
    for idx in filled_indices:
        col = idx // rows
        row = idx % rows
        center_x = (col + 0.5) * cell_width
        center_y = 1 - (row + 0.5) * cell_height
        radius = min(cell_width, cell_height) / 2.5
        circle = plt.Circle((center_x, center_y), radius, color='blue', fill=True)
        ax.add_artist(circle)

    # Plot hollow circles
    for col in range(cols):
        for row in range(rows):
            if col * rows + row not in filled_indices:
                center_x = (col + 0.5) * cell_width
                center_y = 1 - (row + 0.5) * cell_height
                radius = min(cell_width, cell_height) / 2.5
                circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False)
                ax.add_artist(circle)

    # Remove grid lines and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add column numbers
    for col in range(cols):
        col_center_x = (col + 0.5) * cell_width
        ax.text(col_center_x, -0.05, str(col), ha='center', va='center')

    # Add row numbers
    for row in range(rows):
        row_center_y = 1 - (row + 0.5) * cell_height
        ax.text(-0.05, row_center_y, str(row), ha='center', va='center')

    plt.title(title)
    plt.show()

