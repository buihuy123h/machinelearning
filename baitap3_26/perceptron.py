#3.27
import numpy as np

def main():
    x=np.array([[3], [4], [1]])
    w=np.array([[1, 2, -10]])
    wx=np.dot(w, x)
    print("wx =", wx)
    if (wx > 0 or wx == 0):
        print("Output: 1 => Du lieu duoc phan lop sai")
    else:
        print("Output: -1 => Du lieu duoc phan lop dung")
main()

#3.28
import numpy as np

def bai3_28():
    x2 = np.array([[2], [3], [1]])
    w2 = np.array([[-2, 1, 0]])

    wx2 = np.dot(w2, x2)
    print("wx2 =", wx2)

    if (wx2 > 0 or wx2 == 0):
        print("Output: 1 => nhan cua du lieu la 1, nhan du doan dung")
    else:
        print("Output: -1 => nhan cua du lieu la -1, nhan du doan sai")

        learning_rate = 0.5

        # Lap cho den khi w^T*x >= 0
        while wx2 < 0:
            w2 = w2 + learning_rate * x2.T
            wx2 = np.dot(w2, x2)

            print("w2 sau khi cap nhat =", w2)
            print("wx2 =", wx2)

    print("Gia tri sau khi cap nhat la:", wx2)


bai3_28()


#3.29
def __init__(self, learning_rate=0.5, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None

    # Ham huan luyen
    def fit(self, X, y):

        # Khoi tao w = 0
        self.w = np.zeros((X.shape[1], 1))

        for epoch in range(self.max_iter):

            sai = 0

            for i in range(len(X)):

                # Lay tung diem du lieu
                x = X[i].reshape(-1, 1)

                # Tinh w^T x
                wx = np.dot(self.w.T, x)

                # Du doan
                if wx >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Neu du doan sai thi cap nhat w
                if y_pred != y[i]:
                    self.w = self.w + self.learning_rate * y[i] * x
                    sai += 1

            # Neu khong con mau nao sai thi dung
            if sai == 0:
                break

        return self

    # Ham du doan
    def predict(self, X):

        result = []

        for i in range(len(X)):

            x = X[i].reshape(-1, 1)

            wx = np.dot(self.w.T, x)

            if wx >= 0:
                result.append(1)
            else:
                result.append(-1)

        return np.array(result)