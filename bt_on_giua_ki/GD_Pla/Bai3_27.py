# w và x được biểu diễn dưới dạng vector cột
w = [
    [1],
    [2],
    [-10]
]

x = [
    [3],
    [4],
    [1]
]

# Chuyển w thành vector hàng w^T
w_T = [[w[0][0], w[1][0], w[2][0]]]

# Tính w^T x
tong = 0
for i in range(len(w_T[0])):
    tong += w_T[0][i] * x[i][0]

# Dự đoán nhãn bằng Perceptron
if tong >= 0:
    nhan_du_doan = 1
else:
    nhan_du_doan = -1

nhan_thuc_te = -1

print("w^T =", w_T)
print("x =", x)
print("w^T x =", tong)
print("Nhãn dự đoán =", nhan_du_doan)
print("Nhãn thực tế =", nhan_thuc_te)

if nhan_du_doan != nhan_thuc_te:
    print("Điểm dữ liệu bị phân lớp sai.")
else:
    print("Điểm dữ liệu được phân lớp đúng.")