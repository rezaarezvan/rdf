# iris.py - Different figures and plots to illustrate the Iris dataset

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

# Matplot settings for beautiful and blog-friendly/paper-ready plots
plt.style.use('ggplot')

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Plot feature histogram over all classes, use petal length as example, color coded by class
# x-axis: petal length, y-axis: frequency
plt.hist(X[y == 0, 2], color='r', alpha=0.5, label='setosa')
plt.hist(X[y == 1, 2], color='g', alpha=0.5, label='versicolor')
plt.hist(X[y == 2, 2], color='b', alpha=0.5, label='virginica')

plt.title('Histogram of petal length')
plt.xlabel('Petal length (cm)')
plt.ylabel('Frequency')
plt.legend()

# Save pictures as SVG and other misc. settings to make it look good
plt.savefig('iris_histogram.svg', bbox_inches='tight')
