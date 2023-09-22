#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yoo sun song
@student number: 301091906
"""

#1.Retrieve and load the mnist_784 dataset of 70,000 instances.
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
print(mnist.data.shape)

#Split the dataset into training & testing data
X_train, y_train = mnist.data[:60000], mnist.target[:60000]
print(X_train.shape )
X_test, y_test = mnist.data[60000:], mnist.target[60000:]
print(X_test.shape)

#2.Display each digit. 
import matplotlib.pyplot as plt
import numpy as np
unique_labels = np.unique(y_train)
fig, axes = plt.subplots(1, 10, figsize=(10, 1))

for ax, label in zip(axes, unique_labels):
    img_data = X_train[y_train == label][0].reshape(28, 28)
    ax.imshow(img_data, cmap="gray")
    ax.axis("off")
    ax.set_title(label)
plt.show()

#3.Use PCA to retrieve the 1th and 2nd principal component and output their explained variance ratio.
from sklearn.decomposition import PCA
n_components_2 = 2
pca_2 = PCA(n_components= n_components_2)
X_pca_2 = pca_2.fit_transform(X_train)
explained_var_ratio_2 = pca_2.explained_variance_ratio_
explained_var_ratio_sum_2 = pca_2.explained_variance_ratio_.sum()
print(f"Explained Variance Ratio for 1st and 2nd components: {explained_var_ratio_2}")
print(f"Sum of Explained Variance Ratios for 1st and 2nd components: {explained_var_ratio_sum_2}")

#4.Plot the projections of the 1th and 2nd principal component onto a 1D hyperplane.

# First subplot for the 1st Principal Component
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(10):
    plt.scatter(X_pca_2[y_train == str(i), 0], [0]*len(X_pca_2[y_train == str(i), 0]), alpha=0.6, label=str(i))
plt.xlabel("1st Principal Component")
plt.title("Projection of 1st Principal Component onto 1D hyperplane")
plt.legend()


# Second subplot for the 2nd Principal Component
plt.subplot(1, 2, 2)
for i in range(10):
    plt.scatter([0]*len(X_pca_2[y_train == str(i), 1]), X_pca_2[y_train == str(i), 1], alpha=0.6, label=str(i))
plt.ylabel("2nd Principal Component")
plt.title("Projection of 2nd Principal Component onto 1D hyperplane")
plt.legend()

plt.tight_layout()
plt.show()

#5.Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 
#dimensions. 
from sklearn.decomposition import IncrementalPCA
n_batches = 100
n_components_154 = 154
incremental_pca = IncrementalPCA(n_components = n_components_154)
for batch_x in np.array_split(X_train, n_batches):
    incremental_pca.partial_fit(batch_x)
X_reduced_incremental = incremental_pca.transform(X_train)

#6.Display the original and compressed digits from (5).
X_recovered_incremental = incremental_pca.inverse_transform(X_reduced_incremental)

index_to_show = 0

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(X_train[index_to_show].reshape(28, 28), cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(X_recovered_incremental[index_to_show].reshape(28, 28), cmap="gray")
axes[1].set_title("Compressed")
plt.show()