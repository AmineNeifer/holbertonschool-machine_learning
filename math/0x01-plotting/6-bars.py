#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

apples = fruit[0, :]
bananas = fruit[0:2, :]
oranges = fruit[0:3, :]
names = ["Farrah", "Fred", "Felicia"]
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.ylim(0, 80)
p = plt.bar(names, np.sum(fruit, axis=0),
            color="#ffe5b4", width=0.5, label="peaches")
o = plt.bar(names, np.sum(oranges, axis=0), width=0.5,
            color="#ff8000", label="oranges")
b = plt.bar(names, np.sum(bananas, axis=0), width=0.5,
            color="yellow", label="banans")
a = plt.bar(names, apples, width=0.5,
            color="red", label="apples")
handles = (a, b, o, p)
labels = ("apples", "bananas", "oranges", "peaches")
plt.legend(handles, labels, loc=1)
plt.show()
