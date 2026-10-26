
import math
from collections import Counter


# Du lieu Mua may tinh
data = [
    {"Tuổi": "Trẻ", "Thu nhập": "Cao",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Không"},

    {"Tuổi": "Trẻ", "Thu nhập": "Cao",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc",
     "Mua máy tính": "Không"},

    {"Tuổi": "Trung niên", "Thu nhập": "Cao",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Có"},

    {"Tuổi": "Già", "Thu nhập": "Trung bình",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Có"},

    {"Tuổi": "Già", "Thu nhập": "Thấp",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Có"},

    {"Tuổi": "Già", "Thu nhập": "Thấp",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc",
     "Mua máy tính": "Không"},

    {"Tuổi": "Trung niên", "Thu nhập": "Thấp",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc",
     "Mua máy tính": "Có"},

    {"Tuổi": "Trẻ", "Thu nhập": "Trung bình",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Không"},

    {"Tuổi": "Trẻ", "Thu nhập": "Thấp",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Có"},

    {"Tuổi": "Già", "Thu nhập": "Trung bình",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Có"},

    {"Tuổi": "Trẻ", "Thu nhập": "Trung bình",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc",
     "Mua máy tính": "Có"},

    {"Tuổi": "Trung niên", "Thu nhập": "Trung bình",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc",
     "Mua máy tính": "Có"},

    {"Tuổi": "Trung niên", "Thu nhập": "Cao",
     "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt",
     "Mua máy tính": "Có"},

    {"Tuổi": "Già", "Thu nhập": "Trung bình",
     "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc",
     "Mua máy tính": "Không"}
]

TARGET = "Mua máy tính"


# Tinh Entropy
def entropy(rows):
    if not rows:
        return 0

    counts = Counter(row[TARGET] for row in rows)
    total = len(rows)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


# Tinh Information Gain
def information_gain(rows, attribute):
    total_entropy = entropy(rows)
    total = len(rows)

    values = set(row[attribute] for row in rows)
    weighted_entropy = 0

    for value in values:
        subset = [
            row for row in rows
            if row[attribute] == value
        ]

        weighted_entropy += (
            len(subset) / total
        ) * entropy(subset)

    return total_entropy - weighted_entropy


# Tim nhan pho bien nhat
def majority_label(rows, default="Không"):
    if not rows:
        return default

    counts = Counter(row[TARGET] for row in rows)

    # Neu hoa, uu tien nhan "Có"
    if counts.get("Có", 0) >= counts.get("Không", 0):
        return "Có"

    return "Không"


# Xay dung cay quyet dinh ID3
def build_tree(rows, attributes, default="Không"):
    # Truong hop tap du lieu rong
    if not rows:
        return default

    labels = [row[TARGET] for row in rows]

    # Tat ca mau co cung nhan
    if len(set(labels)) == 1:
        return labels[0]

    # Khong con thuoc tinh de chia
    if not attributes:
        return majority_label(rows, default)

    # Chon thuoc tinh co Information Gain lon nhat
    best_attribute = max(
        attributes,
        key=lambda attr: information_gain(rows, attr)
    )

    tree = {best_attribute: {}}
    remaining = [
        attr for attr in attributes
        if attr != best_attribute
    ]

    # Tao nhanh cho tung gia tri thuoc tinh
    values = sorted(set(
        row[best_attribute] for row in rows
    ))

    for value in values:
        subset = [
            row for row in rows
            if row[best_attribute] == value
        ]

        tree[best_attribute][value] = build_tree(
            subset,
            remaining,
            majority_label(rows, default)
        )

    return tree


# Du doan nhan cho mot mau moi
def predict(tree, sample, default="Không"):
    # Neu la nut la, tra ve nhan
    if isinstance(tree, str):
        return tree

    attribute = next(iter(tree))
    value = sample.get(attribute)

    # Neu gia tri chua xuat hien khi hoc
    if value not in tree[attribute]:
        return default

    subtree = tree[attribute][value]

    return predict(subtree, sample, default)


# Xay dung cay
attributes = [
    "Tuổi",
    "Thu nhập",
    "Sinh viên",
    "Đánh giá tín dụng"
]

tree = build_tree(data, attributes)

print("Cay quyet dinh ID3:")
print(tree)


# Kiem tra tren mot so mau thu
test_samples = [
    {
        "Tuổi": "Trẻ",
        "Thu nhập": "Cao",
        "Sinh viên": "Không",
        "Đánh giá tín dụng": "Tốt"
    },
    {
        "Tuổi": "Trẻ",
        "Thu nhập": "Thấp",
        "Sinh viên": "Có",
        "Đánh giá tín dụng": "Tốt"
    },
    {
        "Tuổi": "Trung niên",
        "Thu nhập": "Cao",
        "Sinh viên": "Không",
        "Đánh giá tín dụng": "Tốt"
    },
    {
        "Tuổi": "Già",
        "Thu nhập": "Thấp",
        "Sinh viên": "Có",
        "Đánh giá tín dụng": "Xuất sắc"
    }
]

for sample in test_samples:
    print(sample, "=>", predict(tree, sample))
