import numpy as np


# Fungsi sigmoid untuk regresi logistik
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Fungsi untuk melakukan training model regresi logistik
def train_logistic_regression(X, y, learning_rate=0.01, num_iterations=500):
    m, n = X.shape
    # Inisialisasi bobot dan bias dengan nilai acak
    np.random.seed(0)
    w = np.random.randn(n, 1)
    b = 0

    # Gradient descent
    for i in range(num_iterations):
        # Hitung nilai z dan sigmoid
        z = np.dot(X, w) + b
        A = sigmoid(z)

        # Hitung gradien
        dw = (1 / m) * np.dot(X.T, (A - y))
        db = (1 / m) * np.sum(A - y)

        # Update bobot dan bias
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b


# Fungsi untuk melakukan prediksi dengan model yang sudah dilatih
def predict_logistic_regression(X, w, b):
    z = np.dot(X, w) + b
    A = sigmoid(z)
    predictions = (A > 0.5).astype(int)
    return predictions.flatten()


# Membuat dataset contoh
np.random.seed(0)
tinggi_badan = np.random.normal(170, 10, 1000)  # contoh tinggi badan
jenis_kelamin = np.random.randint(0, 2, 1000)  # contoh jenis kelamin (0 atau 1)

# Menambahkan kolom satu untuk bias
X = tinggi_badan.reshape(-1, 1)
X = np.hstack((np.ones((X.shape[0], 1)), X))

y = jenis_kelamin.reshape(-1, 1)


# Pembagian data menjadi data latih dan data uji
def split_train_test(X, y, test_size=0.2):
    m = X.shape[0]
    indices = np.random.permutation(m)
    test_size = int(m * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_train_test(X, y)

# Melatih model regresi logistik
w, b = train_logistic_regression(X_train, y_train)

# Membuat prediksi dengan data uji untuk evaluasi sederhana
predictions = predict_logistic_regression(X_test, w, b)

# Menampilkan hasil evaluasi
accuracy = np.mean(predictions == y_test.flatten())
print(f"Accuracy: {accuracy * 100:.2f}%")

