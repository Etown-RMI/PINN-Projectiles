import numpy as np
import matplotlib.pyplot as plt
import pinns

x = pinns.Domain(-1, 1, 100)


def pde(x, y):
    dy_xx = pinns.Grad.hessian(y, x)
    dy_xxx = pinns.Grad.jacobian(dy_xx, x)
    return dy_xxx - 2


def ic_out(x_in, y):
    dyx = pinns.Grad.jacobian(y, x_in)
    return 2*dyx-5


ic1 = pinns.IC(x_ic=1, f=ic_out, y_der=2)
ic2 = pinns.IC(x_ic=-1, f=1, y_der=1)
ic3 = pinns.IC(x_ic=-1, f=0, y_der=0)

model = pinns.net(inputs=1, layers=3 * [60], activation='tanh', outputs=1)
pinns.train(model, x, pde, [ic1, ic2, ic3], epochs=2000, lr=0.001)


x_test = np.linspace(-1, 1, 100)
y_true = 1/3*x_test**3 + 5/6*x_test**2 + 5/3*x_test + 7/6
y_pred = model(x_test)

plt.plot(y_true)
plt.plot(y_pred)
plt.title('Evaluation')
plt.legend(['Real', 'Predicted'])
plt.show()
