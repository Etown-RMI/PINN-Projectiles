import numpy as np
import matplotlib.pyplot as plt
import pinns

x = pinns.Domain(0, 10, 100)
accel = -9.8
v0 = 10
x0 = 0


def pde(x, y):
    return pinns.Grad.hessian(y, x) - accel


ic1 = pinns.IC(x_ic=0, f=v0, y_der=1)
ic2 = pinns.IC(x_ic=0, f=x0, y_der=0)

model = pinns.net(inputs=1, layers=2 * [32], activation='softplus', outputs=1)
pinns.train(model, x, pde, [ic1, ic2], epochs=200, lr=0.01)


x_test = np.linspace(0, 10, 100)
y_true = (accel/2)*x_test**2 + v0*x_test + x0
y_pred = model(x_test)

def plot_gif(i):
    predi=y_pred[i]
    truei=y_true[i]
    plt.scatter(2,predi, c="orange")
    plt.scatter(2,truei, c="blue")
    plt.xlim(0,4)
    plt.ylim(min(y_true), max(y_true))
    
frames = [plot_gif(i) for i in range(100)]
gif.save(frames, "test.gif", duration=100)