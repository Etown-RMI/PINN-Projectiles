#taipy test 3
import numpy as np
import matplotlib.pyplot as plt
import pinns
import taipy as tp
import os

x = pinns.Domain(-1, 1, 100)

def pde(x, y):
    dy_xx = pinns.Grad.hessian(y, x)
    dy_xxx = pinns.Grad.jacobian(dy_xx, x)
    return dy_xxx - 2

def ic_out(x_in, y):
    dyx = pinns.Grad.jacobian(y, x_in)
    return 2 * dyx - 5

ic1 = pinns.IC(x_ic=1, f=ic_out, y_der=2)
ic2 = pinns.IC(x_ic=-1, f=1, y_der=1)
ic3 = pinns.IC(x_ic=-1, f=0, y_der=0)

model = pinns.net(inputs=1, layers=3 * [60], activation='tanh', outputs=1)

# Function to train the model
def train_model():
    pinns.train(model, x, pde, [ic1, ic2, ic3], epochs=2000, lr=0.001)

def generate_and_save_plot():
    x_test = np.linspace(-1, 1, 100)
    y_true = 1/3 * x_test**3 + 5/6 * x_test**2 + 5/3 * x_test + 7/6
    y_pred = model(x_test)

    plt.plot(x_test, y_true, label='Real')
    plt.plot(x_test, y_pred, label='Predicted')
    plt.title('Evaluation')
    plt.legend()
    plt.savefig('output_plot.png')

def build_message(name: str):
    # Call the function to train the model
    train_model()
    
    # Call the function to generate and save the plot
    generate_and_save_plot()

    return f"Graph generated for {name}!"

input_name_data_node_cfg = tp.Config.configure_data_node(id="input_name")
message_data_node_cfg = tp.Config.configure_data_node(id="message")
build_msg_task_cfg = tp.Config.configure_task("build_msg", build_message, input_name_data_node_cfg, message_data_node_cfg)
scenario_cfg = tp.Config.configure_scenario("scenario", task_configs=[build_msg_task_cfg])

def submit_scenario(state):
    state.scenario.input_name.write(state.input_name)
    state.scenario.submit(wait=True)
    state.message = state.scenario.message.read()

if __name__ == "__main__":
    tp.Core().run()
    scenario = tp.create_scenario(scenario_cfg)

    page = """
    Name: <|{input_name}|input|>
    <|submit|button|on_action=submit_scenario|>

    Message: <|{message}|text|>

    Generated Plot:
    <|output_plot.png|image|>
    """

    tp.Gui(page).run()