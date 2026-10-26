def f(x):
    return x**2 - 4*x + 5

def dao_ham(x):
    return 2*x - 4

x = 5
learning_rate = 0.2

print("Bước 0: x =", x, "| f(x) =", f(x))

for step in range(1, 5):
    gradient = dao_ham(x)
    x = x - learning_rate * gradient

    print(
        "Bước", step,
        "| Đạo hàm =", round(gradient, 6),
        "| x mới =", round(x, 6),
        "| f(x mới) =", round(f(x), 8)
    )