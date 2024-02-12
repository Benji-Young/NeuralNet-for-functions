import np as np

import data
from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer
"""
points, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (4, 2))
w_h_o = np.random.uniform(-0.5, 0.5, (2, 4))
b_i_h = np.zeros((4, 1))
b_h_o = np.zeros((2, 1))

learn_rate = 0.05
nr_correct = 0
epochs = 10
maxim = 100
minum = 0

database = []

for epoch in range(epochs):
    for p, l in zip(points, labels):
        p = np.transpose([p])
        p = (p-0)/(maxim-0)

        # Forward propagation input -> hidden
        h_pre = b_i_h + (w_i_h @ p)
        h = 1 / (1 + np.exp(-h_pre))

        # Forward propagation hidden -> output
        o_pre = (w_h_o @ h) + b_h_o
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)

        # Backpropagation output -> hidden (cost function derivative)
        if l == 1:
            delta_o = o - np.transpose([[0,1]])
            if o[1] > o[0]:
                nr_correct += 1
        else:
            delta_o = o - np.transpose([[1,0]])
            if o[1] < o[0]:
                nr_correct += 1

        w_h_o += -learn_rate * (delta_o @ np.transpose(h))
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(p)
        b_i_h += -learn_rate * delta_h
    # Show accuracy for this epoch
    acc = round((nr_correct / len(points)) * 100,2)
    print(f"Acc: {round((nr_correct / len(points)) * 100, 2)}%")
    nr_correct = 0
    database.append(acc)

count = 0
loop = 100

figure, axis = plt.subplots(1, 2)

# Show results
for x in range(loop):
    point = np.transpose(np.random.uniform(0, 100, size=(1,2)).astype(int))
    p = (point - 0) / (maxim - 0)

    # 1 above, 0 below
    if data.function(point[0][0]) >= point[1][0]:
        target = 1
    else:
        target = 0

    # Forward propagation input -> hidden
    h_pre = b_i_h + (w_i_h @ p)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = (w_h_o @ h) + b_h_o
    o = 1 / (1 + np.exp(-o_pre))

    out = np.argmax(o)
    if out == 1:
        colour = "red"
    else:
        colour = "blue"

    if target == out:
        count += 1

    x = np.linspace(0, 100, 1000)
    axis[0].plot(x, data.function(x))
    axis[0].scatter(point[0][0], point[1][0], color=colour)
    axis[0].set_title("Function")

accuracy = (count / loop) * 100
print(f'accuracy : {accuracy}')

axis[1].plot([i+1 for i in range(epochs)], database)
axis[1].set_title("Accuracy")
axis[1].set_ylabel("Accuracy")
axis[1].set_xlabel("epoch")

plt.show()
