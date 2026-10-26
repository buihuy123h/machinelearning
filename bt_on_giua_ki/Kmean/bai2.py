
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# 1. DOC DU LIEU IRIS
# ==========================================

ten_cot = [
    "chieu_dai_dai_hoa",
    "chieu_rong_dai_hoa",
    "chieu_dai_canh_hoa",
    "chieu_rong_canh_hoa",
    "loai_hoa"
]

cac_nguon_du_lieu = [
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/iris/iris.data",

    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/iris.csv"
]

du_lieu = None

for duong_dan in cac_nguon_du_lieu:
    try:
        if duong_dan.endswith("iris.csv"):
            bang_tam = pd.read_csv(duong_dan)
            bang_tam = bang_tam.rename(
                columns={"species": "loai_hoa"}
            )
        else:
            bang_tam = pd.read_csv(
                duong_dan,
                header=None,
                names=ten_cot
            )

        bang_tam = bang_tam.dropna()

        if all(cot in bang_tam.columns for cot in ten_cot):
            du_lieu = bang_tam
            print("Doc du lieu thanh cong!")
            break

    except Exception as loi:
        print("Khong doc duoc nguon:", duong_dan)
        print("Ly do:", loi)


# Neu khong co Internet, doc file iris.data tai may
if du_lieu is None:
    tep_du_lieu = Path(__file__).with_name("iris.data")

    if tep_du_lieu.exists():
        du_lieu = pd.read_csv(
            tep_du_lieu,
            header=None,
            names=ten_cot
        ).dropna()

        print("Da doc du lieu tu may tinh.")
    else:
        raise RuntimeError(
            "Khong doc duoc du lieu Iris.\n"
            "Hay kiem tra Internet hoac dat file "
            "iris.data cung thu muc voi bai2.py."
        )


# Chi lay hai lop Setosa va Versicolor
du_lieu = du_lieu[
    du_lieu["loai_hoa"].isin([
        "Iris-setosa",
        "Iris-versicolor",
        "setosa",
        "versicolor"
    ])
].copy()

# Lay hai dac trung dau tien
X = du_lieu[
    ["chieu_dai_dai_hoa", "chieu_rong_dai_hoa"]
].to_numpy(dtype=float)

# Tao nhan that: Setosa = 0, Versicolor = 1
nhan_that = du_lieu["loai_hoa"].isin([
    "Iris-versicolor",
    "versicolor"
]).to_numpy(dtype=int)

print("So mau du lieu:", len(X))
print("So dac trung:", X.shape[1])


# ==========================================
# 2. CAI DAT THUAT TOAN K-MEANS
# ==========================================

class MyKMeans:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tam_cum = None
        self.nhan_cum = None
        self.wcss = None

    def fit(self, X, K, max_iters=100):
        X = np.asarray(X, dtype=float)
        so_mau = X.shape[0]

        if K < 1 or K > so_mau:
            raise ValueError("So cum K khong hop le.")

        # Khoi tao K tam cum ngau nhien
        bo_sinh_so = np.random.default_rng(
            self.random_state
        )

        vi_tri = bo_sinh_so.choice(
            so_mau,
            size=K,
            replace=False
        )

        tam_cum = X[vi_tri].copy()

        # Lap lai viec gan cum va cap nhat tam
        for lan_lap in range(max_iters):

            # Buoc 1: Tinh khoang cach tu tung diem
            # den tung tam cum
            khoang_cach = np.linalg.norm(
                X[:, None, :] - tam_cum[None, :, :],
                axis=2
            )

            # Buoc 2: Gan moi diem vao cum gan nhat
            nhan_cum = np.argmin(
                khoang_cach,
                axis=1
            )

            # Buoc 3: Tinh lai tam cua moi cum
            tam_moi = tam_cum.copy()

            for k in range(K):
                cac_diem = X[nhan_cum == k]

                if len(cac_diem) > 0:
                    tam_moi[k] = np.mean(
                        cac_diem,
                        axis=0
                    )

            # Neu tam khong thay doi thi dung
            if np.allclose(tam_moi, tam_cum):
                tam_cum = tam_moi
                break

            tam_cum = tam_moi

        # Gan lai nhan theo cac tam cuoi cung
        khoang_cach = np.linalg.norm(
            X[:, None, :] - tam_cum[None, :, :],
            axis=2
        )

        nhan_cum = np.argmin(
            khoang_cach,
            axis=1
        )

        # Luu ket qua
        self.tam_cum = tam_cum
        self.nhan_cum = nhan_cum

        # Tinh tong binh phuong khoang cach WCSS
        self.wcss = np.sum(
            (X - tam_cum[nhan_cum]) ** 2
        )

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        khoang_cach = np.linalg.norm(
            X[:, None, :] - self.tam_cum[None, :, :],
            axis=2
        )

        return np.argmin(khoang_cach, axis=1)


# ==========================================
# 3. CHAY THUAT TOAN K-MEANS
# ==========================================

mo_hinh = MyKMeans(random_state=42)

mo_hinh.fit(
    X,
    K=2,
    max_iters=100
)

nhan_du_doan = mo_hinh.nhan_cum

print("\nTam cac cum:")
print(mo_hinh.tam_cum)

print("\nWCSS:")
print(mo_hinh.wcss)

print("\nSo diem trong tung cum:")

for k in range(2):
    so_diem = np.sum(nhan_du_doan == k)
    print(f"Cum {k}: {so_diem} diem")


# ==========================================
# 4. VE KET QUA PHAN CUM
# ==========================================

plt.figure(figsize=(8, 5))

for k in range(2):
    plt.scatter(
        X[nhan_du_doan == k, 0],
        X[nhan_du_doan == k, 1],
        label=f"Cum {k}"
    )

plt.scatter(
    mo_hinh.tam_cum[:, 0],
    mo_hinh.tam_cum[:, 1],
    marker="X",
    s=200,
    label="Tam cum"
)

plt.xlabel("Chieu dai dai hoa (cm)")
plt.ylabel("Chieu rong dai hoa (cm)")
plt.title("Ket qua phan cum K-means")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 5. VE NHAN THAT DE SO SANH
# ==========================================

plt.figure(figsize=(8, 5))

for nhan, ten_loai in [
    (0, "Setosa"),
    (1, "Versicolor")
]:
    plt.scatter(
        X[nhan_that == nhan, 0],
        X[nhan_that == nhan, 1],
        label=ten_loai
    )

plt.xlabel("Chieu dai dai hoa (cm)")
plt.ylabel("Chieu rong dai hoa (cm)")
plt.title("Nhan that cua du lieu Iris")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 6. TINH RAND INDEX
# ==========================================

def tinh_rand_index(nhan_that, nhan_du_doan):

    nhan_that = np.asarray(nhan_that)
    nhan_du_doan = np.asarray(nhan_du_doan)

    so_mau = len(nhan_that)

    # So sanh nhan cua moi cap diem
    cung_nhan_that = (
        nhan_that[:, None] == nhan_that[None, :]
    )

    cung_nhan_du_doan = (
        nhan_du_doan[:, None]
        == nhan_du_doan[None, :]
    )

    # Chi lay cac cap diem khac nhau
    nua_tren = np.triu(
        np.ones((so_mau, so_mau), dtype=bool),
        k=1
    )

    # Dem cac cap co quan he giong nhau
    so_cap_dung = np.sum(
        (cung_nhan_that == cung_nhan_du_doan)
        & nua_tren
    )

    tong_so_cap = np.sum(nua_tren)

    return so_cap_dung / tong_so_cap


chi_so_rand = tinh_rand_index(
    nhan_that,
    nhan_du_doan
)

print("\nRand Index:", chi_so_rand)
