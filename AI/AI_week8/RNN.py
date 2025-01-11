import numpy as np
import matplotlib.pyplot as plt

# Parameters
pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)  # 70% for training


# Generate data
def gen_data(x):
    return (x % pitch) / pitch

t = np.arange(1, N + 1)
y = np.array([gen_data(i) for i in t])

def predict(x):
    return (x % (pitch + 0.1)) / (pitch + 1)  # Modify this logic to simulate prediction

y_pred = np.array([predict(i) for i in t])

def plot_comparison(t, y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_true, label="Original ", linestyle="-", linewidth=2, color='blue')
    plt.plot(t, y_pred, label="Predicted ", linestyle="--", linewidth=2, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_comparison(t, y, y_pred)

