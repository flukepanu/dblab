from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# สร้างข้อมูลสำหรับสองกลุ่ม A และ B
X1, y1 = make_blobs(n_samples=100, 
                    n_features=2,
                    centers=1, 
                    center_box=(2.0, 2.0),
                    cluster_std=0.75, 
                    random_state=69)

X2, y2 = make_blobs(n_samples=100, 
                    n_features=2,
                    centers=1, 
                    center_box=(3.0, 3.0),
                    cluster_std=0.75, 
                    random_state=69)

X = np.vstack((X1, X2))  # รวมข้อมูล
y = np.hstack((np.zeros((X1.shape[0],)), np.ones((X2.shape[0],))))  # รวม labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=69)

model = Sequential()
model.add(Dense(10, input_dim=2, activation='linear'))  # Hidden Layer
model.add(Dense(1, activation='sigmoid'))  # Output Layer 
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(0.001), 
              metrics=['accuracy'])

# สอนโมเดล
model.fit(X_train, y_train, epochs=300, batch_size=16, verbose=1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# แสดงผลลัพธ์ Decision Boundary
plt.figure()
plt.contourf(xx, yy, Z, levels=[-np.inf, 0.5, np.inf], colors=['red', 'blue'], alpha=0.5)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label="Class 1", edgecolor='k', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label="Class 2", edgecolor='k', alpha=0.6)
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Boundary')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
