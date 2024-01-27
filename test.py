import numpy as np
import matplotlib.pyplot as plt
import pinns

# Define the domain for each input
x_domain = pinns.Domain([0, -10, 0, 0], [10, 10, 20, 10], 100)


def pde(inputs, y):
    a = inputs.get(1)
    dy_xx = pinns.Grad.hessian(y, inputs, xi=0)
    return dy_xx - a


# Define the initial conditions
ic1 = pinns.IC(x_ic=[0, -10, 0, 0], f=0, y_der=1)
ic2 = pinns.IC(x_ic=[0, -10, 0, 0], f=0, y_der=0)


# Create a model with 4 inputs
model = pinns.net(inputs=4, layers=2 * [32], activation='softplus', outputs=1)

# Train the model with the 4 inputs
pinns.train(model, x_domain, pde, [ic1, ic2], epochs=2000, lr=0.01)

# Test the model
x_test = np.linspace(0, 10, 100)
accel_test = np.full(100, -9.8)
v0_test = np.full(100, 10)
x0_test = np.full(100, 0)
inputs = np.concatenate([x_test[:, np.newaxis], accel_test[:, np.newaxis],
                              v0_test[:, np.newaxis], x0_test[:, np.newaxis]], axis=1)
y_true = (accel_test/2)*x_test**2 + v0_test*x_test + x0_test
y_pred = model(inputs)

# Plot the results
plt.plot(y_true)
plt.plot(y_pred)
plt.title('Evaluation')
plt.legend(['Real', 'Predicted'])
plt.show()
