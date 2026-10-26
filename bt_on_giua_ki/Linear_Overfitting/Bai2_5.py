# Overfitting# Mỗi dòng gồm: giờ học, chuyên cần, điểm cũ, điểm thi
data = [
    (1,  60, 50, 47),
    (2,  70, 60, 58),
    (3,  80, 70, 69),
    (4,  90, 80, 80),
    (5, 100, 90, 91),
    (1,  85, 70, 60),
    (2,  95, 50, 59),
    (3,  65, 90, 74),
    (4,  75, 60, 69),
    (5,  85, 80, 84)
]

# 8 dòng đầu để huấn luyện, 2 dòng cuối để kiểm tra
train_data = data[:8]
test_data = data[8:]

# Tạo ma trận X.
# Số 1 đầu mỗi dòng là cột bias, dùng để tính hệ số b.
X = []
y = []

for row in train_data:
    gio_hoc = row[0]
    chuyen_can = row[1]
    diem_cu = row[2]
    diem_thi = row[3]

    X.append([1, gio_hoc, chuyen_can, diem_cu])
    y.append(diem_thi)

# Tính X^T X
so_thuoc_tinh = len(X[0])
so_mau = len(X)

XtX = []

for i in range(so_thuoc_tinh):
    hang = []

    for j in range(so_thuoc_tinh):
        tong = 0

        for k in range(so_mau):
            tong += X[k][i] * X[k][j]

        hang.append(tong)

    XtX.append(hang)

# Tính X^T y
Xty = []

for i in range(so_thuoc_tinh):
    tong = 0

    for k in range(so_mau):
        tong += X[k][i] * y[k]

    Xty.append(tong)

# Giải hệ XtX * w = Xty bằng phép khử Gauss
for i in range(so_thuoc_tinh):

    # Tìm hàng có phần tử trụ lớn nhất
    hang_tru = i

    for r in range(i + 1, so_thuoc_tinh):
        if abs(XtX[r][i]) > abs(XtX[hang_tru][i]):
            hang_tru = r

    # Đổi hàng nếu cần
    XtX[i], XtX[hang_tru] = XtX[hang_tru], XtX[i]
    Xty[i], Xty[hang_tru] = Xty[hang_tru], Xty[i]

    # Đưa phần tử trụ về 1
    phan_tu_tru = XtX[i][i]

    for j in range(i, so_thuoc_tinh):
        XtX[i][j] = XtX[i][j] / phan_tu_tru

    Xty[i] = Xty[i] / phan_tu_tru

    # Khử các phần tử khác trong cùng cột
    for r in range(so_thuoc_tinh):
        if r != i:
            he_so = XtX[r][i]

            for j in range(i, so_thuoc_tinh):
                XtX[r][j] -= he_so * XtX[i][j]

            Xty[r] -= he_so * Xty[i]

# Các hệ số tìm được
b = Xty[0]
w1 = Xty[1]
w2 = Xty[2]
w3 = Xty[3]

print("b =", round(b, 4))
print("w1 - hệ số giờ học =", round(w1, 4))
print("w2 - hệ số chuyên cần =", round(w2, 4))
print("w3 - hệ số điểm cũ =", round(w3, 4))

# Dự đoán trên tập kiểm tra
actual = []
predicted = []

for row in test_data:
    gio_hoc = row[0]
    chuyen_can = row[1]
    diem_cu = row[2]
    diem_thi = row[3]

    diem_du_doan = b + w1 * gio_hoc + w2 * chuyen_can + w3 * diem_cu

    actual.append(diem_thi)
    predicted.append(diem_du_doan)

    print(
        "Dữ liệu:", gio_hoc, chuyen_can, diem_cu,
        "| Điểm thật:", diem_thi,
        "| Điểm dự đoán:", round(diem_du_doan, 2)
    )

# Tính R2, NSE, MAE, RMSE
so_mau_test = len(actual)
diem_trung_binh = sum(actual) / so_mau_test

sse = 0
sst = 0
tong_sai_so_tuyet_doi = 0

for i in range(so_mau_test):
    sai_so = actual[i] - predicted[i]

    sse += sai_so ** 2
    sst += (actual[i] - diem_trung_binh) ** 2
    tong_sai_so_tuyet_doi += abs(sai_so)

r2 = 1 - sse / sst
nse = 1 - sse / sst
mae = tong_sai_so_tuyet_doi / so_mau_test
rmse = (sse / so_mau_test) ** 0.5

print("R2 =", round(r2, 4))
print("NSE =", round(nse, 4))
print("MAE =", round(mae, 4))
print("RMSE =", round(rmse, 4))