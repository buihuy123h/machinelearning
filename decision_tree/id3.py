"""Cay quyet dinh ID3 cho bai toan rui ro tin dung.
Khong dung sklearn: tu tinh Entropy va Information Gain.
"""
from __future__ import annotations

from collections import Counter
from math import log2
from typing import Any

# Cac cot: ID, Do tuoi, Hon nhan, So huu BDS, Thu nhap, Rui ro tin dung
DATA = [
    {"ID": 1, "Độ tuổi": "25-33", "Hôn nhân": "Độc thân", "Sở hữu BĐS": "Ở cùng bố mẹ", "Thu nhập": "5-10 triệu", "Rủi ro": 0},
    {"ID": 2, "Độ tuổi": "33-41", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Nhà sở hữu", "Thu nhập": "15-20 triệu", "Rủi ro": 0},
    {"ID": 3, "Độ tuổi": "33-41", "Hôn nhân": "Từng ly hôn", "Sở hữu BĐS": "Nhà thuê", "Thu nhập": "10-15 triệu", "Rủi ro": 1},
    {"ID": 4, "Độ tuổi": "25-33", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Ở cùng bố mẹ", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
    {"ID": 5, "Độ tuổi": "25-33", "Hôn nhân": "Độc thân", "Sở hữu BĐS": "Nhà thuê", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
    {"ID": 6, "Độ tuổi": "33-41", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Nhà sở hữu", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
    {"ID": 7, "Độ tuổi": "41-49", "Hôn nhân": "Độc thân", "Sở hữu BĐS": "Nhà thuê", "Thu nhập": "5-10 triệu", "Rủi ro": 0},
    {"ID": 8, "Độ tuổi": "25-33", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Nhà thuê", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
    {"ID": 9, "Độ tuổi": "33-41", "Hôn nhân": "Từng ly hôn", "Sở hữu BĐS": "Ở cùng bố mẹ", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
    {"ID": 10, "Độ tuổi": "25-33", "Hôn nhân": "Độc thân", "Sở hữu BĐS": "Nhà thuê", "Thu nhập": "10-15 triệu", "Rủi ro": 0},
    {"ID": 11, "Độ tuổi": "33-41", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Nhà sở hữu", "Thu nhập": "15-20 triệu", "Rủi ro": 0},
    {"ID": 12, "Độ tuổi": "41-49", "Hôn nhân": "Độc thân", "Sở hữu BĐS": "Nhà sở hữu", "Thu nhập": "10-15 triệu", "Rủi ro": 1},
    {"ID": 13, "Độ tuổi": "41-49", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Nhà sở hữu", "Thu nhập": "10-15 triệu", "Rủi ro": 0},
    {"ID": 14, "Độ tuổi": "25-33", "Hôn nhân": "Độc thân", "Sở hữu BĐS": "Nhà thuê", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
    {"ID": 15, "Độ tuổi": "25-33", "Hôn nhân": "Đã kết hôn", "Sở hữu BĐS": "Ở cùng bố mẹ", "Thu nhập": "5-10 triệu", "Rủi ro": 1},
]

TARGET = "Rủi ro"
FEATURES = ["Độ tuổi", "Hôn nhân", "Sở hữu BĐS", "Thu nhập"]


def entropy(rows: list[dict[str, Any]]) -> float:
    """H(S) = -sum(p_i * log2(p_i))."""
    if not rows:
        return 0.0
    counts = Counter(row[TARGET] for row in rows)
    n = len(rows)
    return -sum((count / n) * log2(count / n) for count in counts.values())


def partitions(rows: list[dict[str, Any]], feature: str) -> dict[Any, list[dict[str, Any]]]:
    result: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(row[feature], []).append(row)
    return result


def information_gain(rows: list[dict[str, Any]], feature: str) -> float:
    """IG(S,A) = H(S) - sum_v |S_v|/|S| * H(S_v)."""
    if not rows:
        return 0.0
    n = len(rows)
    remainder = sum(
        len(group) / n * entropy(group)
        for group in partitions(rows, feature).values()
    )
    return entropy(rows) - remainder


def majority_class(rows: list[dict[str, Any]]) -> Any:
    """Lop du phong neu du lieu con khong tach duoc hoan toan."""
    return Counter(row[TARGET] for row in rows).most_common(1)[0][0]


def id3(rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any] | Any:
    labels = [row[TARGET] for row in rows]

    # 1. Nut thuan nhat: tra ve nhan duy nhat.
    if len(set(labels)) == 1:
        return labels[0]

    # 2. Het thuoc tinh: tra ve lop xuat hien nhieu nhat.
    if not features:
        return majority_class(rows)

    # max() giu thu tu FEATURES khi co hai IG bang nhau.
    best_feature = max(features, key=lambda f: information_gain(rows, f))
    tree: dict[str, Any] = {
        "feature": best_feature,
        "default": majority_class(rows),
        "branches": {},
    }

    remaining = [feature for feature in features if feature != best_feature]
    for value, group in partitions(rows, best_feature).items():
        tree["branches"][value] = id3(group, remaining)
    return tree


def print_tree(tree: Any, indent: str = "") -> None:
    """In cay theo dang de doc."""
    if not isinstance(tree, dict):
        print(f"{indent}-> Rui ro = {tree}")
        return

    feature = tree["feature"]
    branches = tree["branches"]
    for i, (value, child) in enumerate(branches.items()):
        connector = "└──" if i == len(branches) - 1 else "├──"
        print(f"{indent}{connector} {feature} = {value}")
        print_tree(child, indent + ("    " if i == len(branches) - 1 else "│   "))


def predict(tree: Any, row: dict[str, Any]) -> Any:
    """Du doan mot khach hang moi."""
    if not isinstance(tree, dict):
        return tree
    value = row.get(tree["feature"])
    child = tree["branches"].get(value)
    # Neu gap gia tri chua tung xuat hien, dung lop pho bien tai nut.
    return tree["default"] if child is None else predict(child, row)


if __name__ == "__main__":
    print("Entropy ban dau:", round(entropy(DATA), 6))
    print("\nInformation Gain tai nut goc:")
    for feature in FEATURES:
        print(f"  {feature:12s}: {information_gain(DATA, feature):.6f}")

    tree = id3(DATA, FEATURES)
    print("\nCAY ID3:")
    print_tree(tree)

    new_customer = {
        "Độ tuổi": "25-33",
        "Hôn nhân": "Độc thân",
        "Sở hữu BĐS": "Nhà thuê",
        "Thu nhập": "5-10 triệu",
    }
    print("\nDu doan khach hang moi:", predict(tree, new_customer))
