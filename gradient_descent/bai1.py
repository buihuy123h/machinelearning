def grad(x):
    return 2 * x
def cost(x):
    return x ** 2
def gradient_descent(starting_point, learning_rate):
    x = [starting_point]
    for it in range(100):  # Number of iterations
        x_new= x[-1] - learning_rate * grad(x[-1])
        x.append(x_new)
        if abs(grad(x_new)) < 1e-3:  # Convergence check
            break
    return (x,it)
(x1,it1)=gradient_descent(10, 0.3)
(x2,it2)=gradient_descent(-10, 0.3)
print("Final point for starting point 10:x1=%f, Iterations: %d"%(x1[-1], it1))
print("Final point for starting point -10:x2=%f, Iterations: %d"%(x2[-1], it2))