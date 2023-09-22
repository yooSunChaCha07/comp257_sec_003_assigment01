#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yoo sun song
@student number: 301091906
"""

#1.Generate Swiss roll dataset. 
from sklearn.datasets import make_swiss_roll
X, y = make_swiss_roll(n_samples=1000, noise=0.1, random_state=96)

#2.Plot the resulting generated Swiss roll dataset.
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
plt.title('Swiss Roll')
plt.show()
#3.Use Kernel PCA (kPCA) with linear kernel, a RBF kernel, and a sigmoid kernel.
#4.Plot the kPCA results of applying the linear kernel, a RBF kernel, and a sigmoid kernel from (3). 
#Explain and compare the results.

from sklearn.decomposition import KernelPCA

#Linear Kernel

kpca_linear_kernel = KernelPCA(kernel="linear",n_components =2)
X_kpca_linear_kernel = kpca_linear_kernel.fit_transform(X)
plt.scatter(X_kpca_linear_kernel[:, 0], X_kpca_linear_kernel[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Linear Kernel')
plt.show()

#RBF Kernel

kpca_rbf_kernel = KernelPCA(n_components=2, kernel='rbf',gamma=0.04)
X_kpca_rbf_kernel = kpca_rbf_kernel.fit_transform(X)
plt.scatter(X_kpca_rbf_kernel[:, 0], X_kpca_rbf_kernel[:, 1], c=y)
plt.title('RBF Kernel')
plt.show()

#Sigmoid Kernel

kpca_sigmoid_kernel = KernelPCA(n_components=2, kernel='sigmoid', gamma=1e-05,coef0=0)
X_kpca_sigmoid_kernel = kpca_sigmoid_kernel.fit_transform(X)
plt.scatter(X_kpca_sigmoid_kernel[:, 0], X_kpca_sigmoid_kernel[:, 1], c=y)
plt.title('Sigmoid Kernel')
plt.show()

#5.Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
#Use GridSearchCV to find the best kernel and gamma value for kPCA 
#in order to get the best classification accuracy at the end of the pipeline. 
#Print out best parameters found by GridSearchCV. 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

y = np.where(y <= y.mean(), 0, 1)

clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)), 
    ('log_reg', LogisticRegression())
    ])

param_grid = {
    "kpca__gamma": np.linspace(0.1, 1, 10),
    "kpca__kernel": ["linear", "rbf", "sigmoid"], 
    }

grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', refit=True, cv=3)
grid_search.fit(X, y)

print("Best parameters: ", grid_search.best_params_)

#6.Plot the results from using GridSearchCV in (5).
best_pipeline = grid_search.best_estimator_

X_best_transformed = best_pipeline.named_steps['kpca'].transform(X)

plt.scatter(X_best_transformed[:, 0], X_best_transformed[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Best kPCA Transformation According to GridSearchCV")
plt.show()
