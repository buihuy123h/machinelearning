class Perceptron:
    def __init__(self, learning_rate=1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        # Khởi tạo trọng số bằng 0
        so_thuoc_tinh = len(X[0])
        self.weights = [0] * so_thuoc_tinh
        self.bias = 0

        # Huấn luyện
        for epoch in range(self.epochs):
            so_mau_sai = 0

            for i in range(len(X)):
                # Tính w.x + bias
                gia_tri = self.bias

                for j in range(so_thuoc_tinh):
                    gia_tri += self.weights[j] * X[i][j]

                # Dự đoán nhãn
                if gia_tri >= 0:
                    nhan_du_doan = 1
                else:
                    nhan_du_doan = -1

                # Nếu dự đoán sai thì cập nhật trọng số và bias
                if nhan_du_doan != y[i]:
                    for j in range(so_thuoc_tinh):
                        self.weights[j] += (
                            self.learning_rate * y[i] * X[i][j]
                        )

                    self.bias += self.learning_rate * y[i]
                    so_mau_sai += 1

            # Nếu không còn mẫu nào sai thì dừng
            if so_mau_sai == 0:
                break

        return self

    def predict(self, X):
        ket_qua = []

        for mau in X:
            gia_tri = self.bias

            for j in range(len(self.weights)):
                gia_tri += self.weights[j] * mau[j]

            if gia_tri >= 0:
                ket_qua.append(1)
            else:
                ket_qua.append(-1)

        return ket_qua


# Dữ liệu huấn luyện: mỗi hàng là một mẫu, mỗi cột là một thuộc tính
X_train = [
    [-2, -1],
    [-1, -2],
    [ 1,  2],
    [ 2,  1]
]

# Nhãn đúng tương ứng
y_train = [-1, -1, 1, 1]

# Tạo và huấn luyện mô hình
mo_hinh = Perceptron(learning_rate=1, epochs=100)
mo_hinh.fit(X_train, y_train)

# Dự đoán dữ liệu mới
X_moi = [
    [3, 1],
    [-3, -1]
]

du_doan = mo_hinh.predict(X_moi)

print("Trọng số đã học:", mo_hinh.weights)
print("Bias đã học:", mo_hinh.bias)
print("Nhãn dự đoán:", du_doan)