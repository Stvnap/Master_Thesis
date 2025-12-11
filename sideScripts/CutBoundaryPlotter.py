"""
TestBoundaryPlotter.py

Script for short test plotting domain boundary error of a small subset split by class. This were results from early development when trying the cutting boundary prediction approach.

"""

# -------------------------
# Imports & Globals
# -------------------------
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

COLORS = cm.get_cmap("tab10").colors


# -------------------------
# Function
# -------------------------


def CutBoundaryPlotter():
    """
    Plots the change in logits when cutting the N-terminal and C-terminal boundaries of a protein sequence. First visualizes the performance of the cutting boundary approach.
    Gives 2 plots for the front and back cut respectively. Data is manually inputted from previous experiments.
    """
    # Front cut data
    y1 = [
        8.421588,
        9.121255,
        9.817998,
        10.432904,
        10.648284,
        10.360881,
        9.735748,
        8.657065,
        9.204885,
        9.252943,
        8.593333,
    ]
    x1 = [
        "0-165",
        "10-165",
        "20-165",
        "30-165",
        "40-165",
        "50-165",
        "60-165",
        "70-165",
        "80-165",
        "90-165",
        "100-165",
    ]

    # Back cut data
    y2 = [
        8.421588,
        8.697448,
        7.7547417,
        7.3161817,
        7.2720814,
        6.7960944,
        6.1509957,
        5.516048,
        5.8958745,
        6.534407,
        7.080178,
    ]
    x2 = [
        "0-165",
        "0-155",
        "0-145",
        "0-135",
        "0-125",
        "0-115",
        "0-105",
        "0-95",
        "0-85",
        "0-75",
        "0-65",
    ]

    # Plotting both graphs
    for x, y in [(x1, y1), (x2, y2)]:
        plt.figure(figsize=(10, 6   ))
        plt.plot(x, y, marker="o", color=COLORS[0] if x == x1 else COLORS[1])
        plt.ylabel("Absolute Error in Residues")
        if x == x1:
            plt.legend(["HMM Boundary Start at 24"])
            plt.title("Change of Logits When Cutting N-terminal Boundary")
            plt.xlabel("Residue Range")
            plt.savefig('/home/sapelt/Documents/Master/FINAL/Change of Logits When Cutting N-terminal Boundary.png', dpi=600)

        else:
            plt.legend(["HMM Boundary End at 158"])
            plt.title("Change of Logits When Cutting C-terminal Boundary")
            plt.xlabel("Residue Range")
            plt.savefig('/home/sapelt/Documents/Master/FINAL/Change of Logits When Cutting C-terminal Boundary.png', dpi=600)

        # plt.xticks(rotation=45)
        plt.show()


def CutClassPlotter():
    # color map for classes
    class_cmap = ListedColormap(COLORS[:2])

    # Data with boundaries and boundary predictions (y_pred) and classes (0 and 1)
    boundaries = [24, 8, 25, 2, 2, 2, 8, 18, 47, 8, 2, 9, 41, 2, 9, 65, 17]
    y_pred = [40, 10, 30, 80, 80, 80, 20, 20, 70, 20, 80, 40, 60, 80, 10, 80, 20]
    x = [
        "#0",
        "#14",
        "#23",
        "#35",
        "#38",
        "#55",
        "#56",
        "#57",
        "#58",
        "#63",
        "#72",
        "#74",
        "#77",
        "#80",
        "#93",
        "#96",
        "#97",
    ]
    classes = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]

    # Calculate absolute differences between boundaries and predictions (error)
    y = [abs(boundaries[i] - y_pred[i]) for i in range(len(boundaries))]

    # Plotting scatter
    plt.figure(figsize=(8,5))
    plt.scatter(
        x, y, c=classes, cmap=class_cmap, marker="o", s=50, edgecolor="k", alpha=0.8
    )
    plt.title("Difference (boundary to prediction) by Class")
    plt.xlabel("Sequence Index")
    plt.ylabel("Residue Difference")
    # plt.grid(True)
    # Add legend for classes
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Class 0 (PF00177)",
            markerfacecolor=COLORS[0],
            markersize=10,
            markeredgecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Class 1 (PF00210)",
            markerfacecolor=COLORS[1],
            markersize=10,
            markeredgecolor="k",
        ),
    ]
    plt.legend(handles=legend_elements, title="Classes")
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Cut Boundary Prediction Error by Class.png', dpi=600)
    plt.show()


#########################################
if __name__ == "__main__":
    CutBoundaryPlotter()
    CutClassPlotter()
