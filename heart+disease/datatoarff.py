#!/usr/bin/env python3
"""Chuyển dữ liệu Heart Disease của UCI sang định dạng ARFF.

Hỗ trợ cả ``cleveland.data`` (raw, 76 thuộc tính) và
``processed.cleveland.data`` (CSV, 14 thuộc tính).

Lưu ý: ``cleveland.data`` trong thư mục này bị hỏng từ bản ghi 283 (xem
WARNING). Chương trình chỉ ghi các bản ghi hợp lệ trước vị trí hỏng và in cảnh
báo, không tạo các dòng ARFF lệch cột.

Cách chạy:
    python datatoarff.py
    python datatoarff.py cleveland.data cleveland.arff
    python datatoarff.py processed.cleveland.data processed.cleveland.arff
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


RAW_ATTRIBUTE_NAMES = [
    "id", "ccf", "age", "sex", "painloc", "painexer", "relrest", "pncaden",
    "cp", "trestbps", "htn", "chol", "smoke", "cigs", "years", "fbs",
    "dm", "famhist", "restecg", "ekgmo", "ekgday", "ekgyr", "dig",
    "prop", "nitr", "pro", "diuretic", "proto", "thaldur", "thaltime",
    "met", "thalach", "thalrest", "tpeakbps", "tpeakbpd", "dummy", "exang",
    "xhypo", "oldpeak", "oldpeak2", "oldpeak3", "dummy2", "slope", "ca",
    "restckm", "exerckm", "restef", "restwm", "exeref", "exerwm", "thal",
    "thalsev", "thalpul", "earlobe", "cmo", "cday", "cyr", "num", "lmt",
    "ladprox", "laddist", "diag", "cxmain", "ramus", "om1", "om2", "rcaprox",
    "rcadist", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "cathef", "junk", "name",
]
PROCESSED_ATTRIBUTE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
NUMBER = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")


def is_number(value: str) -> bool:
    return value == "?" or bool(NUMBER.fullmatch(value))


def read_raw(path: Path) -> tuple[list[list[str]], str | None]:
    """Đọc raw data theo từng khối 10 dòng."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="latin-1").splitlines()
        if line.strip()
    ]
    rows: list[list[str]] = []
    warning = None

    for start in range(0, len(lines), 10):
        block = lines[start:start + 10]
        fields = " ".join(block).split()
        valid = (
            len(block) == 10
            and len(fields) == len(RAW_ATTRIBUTE_NAMES)
            and fields[-1] == "name"
            and all(is_number(value) for value in fields[:-1])
        )
        if not valid:
            warning = (
                f"Dữ liệu raw bị hỏng tại bản ghi {start // 10 + 1}; "
                "đã bỏ phần còn lại của file."
            )
            break
        rows.append(fields)

    return rows, warning


def read_processed(path: Path) -> tuple[list[list[str]], str | None]:
    """Đọc file processed dạng CSV gồm 14 thuộc tính mỗi dòng."""
    rows: list[list[str]] = []
    warning = None
    with path.open("r", encoding="latin-1", newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            row = [value.strip() for value in row]
            if len(row) != len(PROCESSED_ATTRIBUTE_NAMES) or not all(is_number(value) for value in row):
                warning = f"Bỏ dòng {line_number} vì không có đúng 14 giá trị số hoặc '?'."
                continue
            rows.append(row)
    return rows, warning


def detect_format(path: Path) -> str:
    with path.open("r", encoding="latin-1") as file:
        for line in file:
            if line.strip():
                return "processed" if "," in line else "raw"
    raise ValueError("File dữ liệu rỗng.")


def quote_string(value: str) -> str:
    return "'" + value.replace("'", "\\'") + "'"


def write_arff(
    input_path: Path,
    output_path: Path,
    relation: str,
    data_format: str = "auto",
) -> tuple[int, str | None]:
    data_format = detect_format(input_path) if data_format == "auto" else data_format
    if data_format == "raw":
        names = RAW_ATTRIBUTE_NAMES
        rows, warning = read_raw(input_path)
    elif data_format == "processed":
        names = PROCESSED_ATTRIBUTE_NAMES
        rows, warning = read_processed(input_path)
    else:
        raise ValueError(f"Định dạng không hợp lệ: {data_format}")

    types = [
        "string" if data_format == "raw" and i == len(names) - 1 else "numeric"
        for i in range(len(names))
    ]
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"@relation {quote_string(relation)}\n\n")
        for name, value_type in zip(names, types):
            output.write(f"@attribute {name} {value_type}\n")
        output.write("\n@data\n")
        for row in rows:
            values = [
                quote_string(value) if value_type == "string" else value
                for value, value_type in zip(row, types)
            ]
            output.write(",".join(values) + "\n")
    return len(rows), warning


def main() -> int:
    parser = argparse.ArgumentParser(description="Đổi file Heart Disease sang ARFF.")
    parser.add_argument(
        "input", nargs="?", default="cleveland.data",
        help="File dữ liệu đầu vào (mặc định: cleveland.data)",
    )
    parser.add_argument(
        "output", nargs="?",
        help="File ARFF đầu ra (mặc định: đổi đuôi input thành .arff)",
    )
    parser.add_argument("--format", choices=("auto", "raw", "processed"), default="auto")
    parser.add_argument("--relation", default="cleveland")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".arff")
    if not input_path.is_file():
        raise SystemExit(f"Không tìm thấy file: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("File input và output phải khác nhau.")

    try:
        count, warning = write_arff(input_path, output_path, args.relation, args.format)
    except (OSError, ValueError, csv.Error) as error:
        raise SystemExit(f"Không chuyển được dữ liệu: {error}") from error

    print(f"Đã chuyển {count} bản ghi sang: {output_path.resolve()}")
    if warning:
        print(f"CẢNH BÁO: {warning}", file=sys.stderr)
        if detect_format(input_path) == "raw":
            print(
                "File processed.cleveland.data mới là bản đầy đủ 303 dòng với 14 thuộc tính.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

