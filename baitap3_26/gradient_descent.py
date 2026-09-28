def grad(x):
    return 2*x-4
def cost(x):
    return x**2-4*x+5
def gradient_descent(starting_point, learning_rate):
    x = [starting_point]
    while grad(x[-1]) > 0.0001:
        x.append(x[-1] - learning_rate * grad(x[-1]))
    return x
def main():
    starting_point = 5
    learning_rate = 0.2
    x_values = gradient_descent(starting_point, learning_rate)
    for i, x in enumerate(x_values):
        print(f"Iteration {i}: x = {x}, cost = {cost(x)}")
    print("ham so ho tu tai X =",x_values[-1], "voi gia tri ham so ho tu =", cost(x_values[-1]))
main()


