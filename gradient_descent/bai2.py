# import sympy as sp
# x = sp.symbols('x')
# def grad(fx, x):
#     return sp.diff(fx, x)
# def cost(fx, x):
#     return fx
# def gradient_descent(fx, starting_point, learning_rate):
#     x_value=[starting_point]
#     for it in range(100):  # Number of iterations
#         grad_value = grad(fx, x)
#         grad_value = float(grad_value.subs(x, x_value[-1]))
#         x_new = x_value[-1] - learning_rate * grad_value
#         x_value.append(x_new)
#         if abs(grad_value) < 1e-3:  # Convergence check
#             break
#     return (x_value, it)
# def main():
#     fx = input("Enter a function f(x): ")
#     (x1, it1) = gradient_descent(sp.sympify(fx), 10, 0.3)
#     print("Final point: %f, Iterations: %d" % (x1[-1], it1))
#     (x2, it2) = gradient_descent(sp.sympify(fx), -10, 0.3)
#     print("Final point: %f, Iterations: %d" % (x2[-1], it2))
# main()

def grad(x):
    return x**2-1
def cost(x):
    return (1/3)*x**3-x
def gradient_descent(starting_point, learning_rate):
    x = [starting_point]
    for it in range(100):  # Number of iterations
        x_new= x[-1] - learning_rate * grad(x[-1])
        x.append(x_new)
        if abs(grad(x_new)) < 1e-3:  # Convergence check
            break
    return (x,it)
(x1,it1)=gradient_descent(10, 0.03)
print("Final point for starting point 10:x1=%f, Iterations: %d"%(x1[-1], it1))