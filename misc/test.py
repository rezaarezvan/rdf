import matplotlib.pyplot as plt

# Example plot
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4], label="Example")
ax.legend()

# Save as SVG
fig.savefig("output.svg", format="svg")
