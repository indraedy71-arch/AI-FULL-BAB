# Exercise 4.2
# Modified Python program from Example 4.2 so that it has three inputs instead of two inputs.

from sklearn.neural_network import MLPClassifier

# 3-input dataset (Logical OR for 3 inputs)
X = [
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

print("Predictions for [[0, 0, 0], [1, 1, 1], [1, 0, 0]]:")
print(clf.predict([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.]]))
print("Layer weights shapes:", [coef.shape for coef in clf.coefs_])
