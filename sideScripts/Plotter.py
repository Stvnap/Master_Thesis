import matplotlib.pyplot as plt
import numpy as np


def plotter():
    # # Front cut
    # y=[8.421588,9.121255,9.817998,10.432904,10.648284,10.360881,9.735748,8.657065,9.204885,9.252943,8.593333]
    # x=["0-165",'10-165','20-165','30-165','40-165','50-165','60-165','70-165','80-165','90-165','100-165']

    # # Back cut
    # y=[8.421588,8.697448,7.7547417,7.3161817,7.2720814,6.7960944,6.1509957,5.516048,5.8958745,6.534407,7.080178]
    # x = ['0-165','0-155','0-145','0-135','0-125','0-115','0-105','0-95','0-85','0-75','0-65']
    # # print(x)



    # how far off
    boundaries=[24,8,25,2,2,2,8,18,47,8,2,9,41,2,9,65,17]
    y_pred=[40, 10, 30, 80, 80, 80, 20, 20, 70, 20, 80, 40, 60, 80, 10, 80, 20]
    x = ["#0", "#14", "#23", "#35", "#38", "#55", "#56", "#57", "#58", "#63", "#72", "#74", "#77", "#80", "#93", "#96", "#97"]
    classes=[1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1]


    print(len(boundaries),len(y_pred),len(classes))
    y = [abs(boundaries[i] - y_pred[i]) for i in range(len(boundaries))]





    # choose a colormap (e.g. 0→blue, 1→red)
    cmap = plt.get_cmap('bwr')  # blue-white-red

    plt.figure(figsize=(10,6))
    scatter = plt.scatter(
        x, y,
        c=classes,           # class 0 or 1
        cmap=cmap,           # map 0→blue, 1→red
        marker='o',
        s=50,               # marker size
        edgecolor='k',       # black border around points
        alpha=0.8
    )

    plt.title('Difference (boundary to prediction) by Class')
    plt.xlabel('Sequence Index')
    plt.ylabel('Residue Difference')
    plt.grid(True)

    # add a legend for the classes:
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='class 0 (PF00177)',
            markerfacecolor=cmap(0.0), markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='class 1 (PF00210)',
            markerfacecolor=cmap(1.0), markersize=10, markeredgecolor='k'),
    ]
    plt.legend(handles=legend_elements, title='Classes')

    plt.show()

if __name__ == "__main__":
    plotter()