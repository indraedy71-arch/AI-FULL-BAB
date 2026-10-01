# Exercise 4.1
# Modified Python program from Example 4.1 to perform logical OR of inputs.

from numpy import exp, array, random, dot

# Set up parameters for Logical OR ================================
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
outputs = array([[0, 1, 1, 1]]).T  # Logical OR outputs
random.seed(1)
weights = 2 * random.random((2, 1)) - 1

# Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

# Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron with different inputs ===========================
test_inputs = [array([0, 0]), array([1, 0]), array([0, 1]), array([1, 1])]
for x in test_inputs:
    print(f"Input {x} -> Predicted Output: {neuron(x, weights)[0]:.4f}")
