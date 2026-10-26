def chuyen_vi(ma_tran):
    return [list(hang) for hang in zip(*ma_tran)]

def tich_hang_cot(hang, cot):
    tong = 0
    for i in range(len(hang[0])):
        tong += hang[0][i] * cot[i][0]
    return tong

w = [[-2], [1], [0]]
x = [[2], [3], [1]]  # Phần tử 1 cuối cùng là bias
y_thuc_te = 1
learning_rate = 1

# Tính và dự đoán ban đầu
gia_tri = tich_hang_cot(chuyen_vi(w), x)
nhan_du_doan = 1 if gia_tri >= 0 else -1

print("w^T x ban đầu =", gia_tri)
print("Nhãn dự đoán ban đầu =", nhan_du_doan)

# Cập nhật nếu dự đoán sai
if nhan_du_doan != y_thuc_te:
    for i in range(len(w)):
        w[i][0] += learning_rate * y_thuc_te * x[i][0]

# Tính lại sau cập nhật
gia_tri_moi = tich_hang_cot(chuyen_vi(w), x)

print("w sau cập nhật =", w)
print("w^T x sau cập nhật =", gia_tri_moi)