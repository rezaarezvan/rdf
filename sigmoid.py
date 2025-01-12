import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid(ax=None, color_map=None):
    """
    Plot the sigmoid function
    """
    fig, ax = plt.subplots()

    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))

    ax.plot(x, y, label='sigmoid')

    ax.set_title('Sigmoid Function')
    ax.set_xlabel('f(x)')
    ax.set_ylabel(r'$\sigma(f(x))$')
    ax.legend()

    plt.tight_layout()


if __name__ == "__main__":
    from rdp import RDP
    plotter = RDP()
    svg_content = plotter.create_themed_plot(plot_sigmoid)
    with open('sigmoid.svg', 'w') as f:
        f.write(svg_content)
