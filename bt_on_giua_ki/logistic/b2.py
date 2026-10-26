
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. DOC DU LIEU IRIS
# ==========================================

duong_dan = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/iris/iris.data"
)

ten_cot = [
    "chieu_dai_dai_hoa",
    "chieu_rong_dai_hoa",
    "chieu_dai_canh_hoa",
    "chieu_rong_canh_hoa",
    "loai_hoa"
]

du_lieu = pd.read_csv(
    duong_dan,
    header=None,
    names=ten_cot
)

du_lieu = du_lieu.dropna()

# Chi lay hai lop Setosa va Versicolor
du_lieu = du_lieu[
    du_lieu["loai_hoa"].isin([
        "Iris-setosa",
        "Iris-versicolor"
    ])
]

# Lay hai dac trung dau tien
X = du_lieu[
    ["chieu_dai_dai_hoa", "chieu_rong_dai_hoa"]
].to_numpy(dtype=float)

# Gan nhan: Setosa = 0, Versicolor = 1
y = (
    du_lieu["loai_hoa"] == "Iris-versicolor"
).to_numpy(dtype=float)

# Chuan hoa du lieu
gia_tri_trung_binh = X.mean(axis=0)
do_lech_chuan = X.std(axis=0)

X_chuan_hoa = (
    X - gia_tri_trung_binh
) / do_lech_chuan

print("So mau du lieu:", len(y))
print("So dac trung:", X_chuan_hoa.shape[1])


# ==========================================
# 2. CAC HAM CUA LOGISTIC REGRESSION
# ==========================================

def ham_sigmoid(z):
    # Gioi han z de tranh tran so
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


def tinh_loss(X, y, w, b):
    # So luong mau du lieu
    so_mau = len(y)

    # Tinh gia tri dau vao cua ham sigmoid
    z = X @ w + b

    # Tinh xac suat du doan
    xac_suat = ham_sigmoid(z)

    # Tranh log(0)
    epsilon = 1e-12
    xac_suat = np.clip(
        xac_suat,
        epsilon,
        1 - epsilon
    )

    # Tinh Binary Cross-Entropy Loss
    loss = -np.mean(
        y * np.log(xac_suat)
        + (1 - y) * np.log(1 - xac_suat)
    )

    return loss


# ==========================================
# 3. HUAN LUYEN BANG GRADIENT DESCENT
# ==========================================

def huan_luyen_logistic(
    X,
    y,
    toc_do_hoc=0.1,
    so_epoch=1000
):
    so_mau, so_dac_trung = X.shape

    # Khoi tao trong so va he so dich
    w = np.zeros(so_dac_trung)
    b = 0.0

    lich_su_loss = []

    for epoch in range(so_epoch):

        # Buoc 1: Tinh xac suat du doan
        z = X @ w + b
        xac_suat = ham_sigmoid(z)

        # Buoc 2: Tinh sai so du doan
        sai_so = xac_suat - y

        # Buoc 3: Tinh gradient cua w va b
        dao_ham_w = (X.T @ sai_so) / so_mau
        dao_ham_b = np.mean(sai_so)

        # Buoc 4: Cap nhat w va b bang Gradient Descent
        w = w - toc_do_hoc * dao_ham_w
        b = b - toc_do_hoc * dao_ham_b

        # Buoc 5: Tinh va luu Loss sau khi cap nhat
        loss = tinh_loss(X, y, w, b)
        lich_su_loss.append(loss)

    return w, b, lich_su_loss


# ==========================================
# 4. THU NGHIEM 3 TOC DO HOC
# ==========================================

cac_toc_do_hoc = [0.01, 0.1, 0.5]
so_epoch = 1000

ket_qua = {}

for toc_do_hoc in cac_toc_do_hoc:

    w, b, lich_su_loss = huan_luyen_logistic(
        X_chuan_hoa,
        y,
        toc_do_hoc=toc_do_hoc,
        so_epoch=so_epoch
    )

    ket_qua[toc_do_hoc] = {
        "w": w,
        "b": b,
        "loss": lich_su_loss
    }

    print("\nToc do hoc:", toc_do_hoc)
    print("Trong so w:", w)
    print("He so b:", b)
    print("Loss cuoi cung:", lich_su_loss[-1])


# ==========================================
# 5. VE DO THI LOSS THEO EPOCH
# ==========================================

plt.figure(figsize=(9, 5))

for toc_do_hoc in cac_toc_do_hoc:

    plt.plot(
        range(1, so_epoch + 1),
        ket_qua[toc_do_hoc]["loss"],
        label=f"Toc do hoc = {toc_do_hoc}"
    )

plt.xlabel("Epoch")
plt.ylabel("Ham mat mat Cross-Entropy")
plt.title("Logistic Regression - Loss theo Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 6. VE DUONG BIEN PHAN LOP 2D
# ==========================================

fig, cac_truc = plt.subplots(
    1,
    3,
    figsize=(17, 5)
)

# Xac dinh pham vi truc toa do theo du lieu goc
x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5

y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

# Tao luoi diem de ve duong bien phan lop
luoi_x, luoi_y = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250)
)

# Chuyen luoi ve cung thang do chuan hoa voi du lieu hoc
luoi_goc = np.c_[
    luoi_x.ravel(),
    luoi_y.ravel()
]

luoi_chuan_hoa = (
    luoi_goc - gia_tri_trung_binh
) / do_lech_chuan

for truc, toc_do_hoc in zip(
    cac_truc,
    cac_toc_do_hoc
):

    w = ket_qua[toc_do_hoc]["w"]
    b = ket_qua[toc_do_hoc]["b"]

    # Tinh xac suat cho tung diem tren luoi
    xac_suat = ham_sigmoid(
        luoi_chuan_hoa @ w + b
    )

    xac_suat = xac_suat.reshape(
        luoi_x.shape
    )

    # Ve vung du doan cua hai lop
    truc.contourf(
        luoi_x,
        luoi_y,
        xac_suat,
        levels=[0, 0.5, 1],
        alpha=0.2
    )

    # Ve duong bien phan lop P(y=1|x) = 0.5
    truc.contour(
        luoi_x,
        luoi_y,
        xac_suat,
        levels=[0.5],
        colors="black",
        linewidths=2
    )

    # Ve cac diem thuoc lop Setosa
    truc.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Setosa (0)",
        marker="o"
    )

    # Ve cac diem thuoc lop Versicolor
    truc.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Versicolor (1)",
        marker="x"
    )

    truc.set_xlabel("Chieu dai dai hoa (cm)")
    truc.set_ylabel("Chieu rong dai hoa (cm)")

    truc.set_title(
        f"Toc do hoc = {toc_do_hoc}"
    )

    truc.legend()
    truc.grid(True, alpha=0.3)

plt.suptitle(
    "Duong bien phan lop Logistic Regression"
)

plt.tight_layout()
plt.show()
