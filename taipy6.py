# multi in to multi output, projectile model evaluation is flawed though
import numpy as np
import matplotlib.pyplot as plt
import pinns
import taipy as tp

model = None
x = pinns.Domain(-1, 1, 100)
accel = -9.8
v0 = 10
x0 = 0

def pde(x, y):
    return pinns.Grad.hessian(y, x) - accel

def generate_and_save_plot():
    global model
    ic1 = pinns.IC(x_ic=0, f=v0, y_der=1)
    ic2 = pinns.IC(x_ic=0, f=x0, y_der=0)
    model = pinns.net(inputs=1, layers=3 * [60], activation='tanh', outputs=1)
    pinns.train(model, x, pde, [ic1, ic2], epochs=100, lr=0.01)

    x_test = np.linspace(0, 10, 100)
    y_true = (accel/2)*x_test**2 + v0*x_test + x0
    y_pred = model(x_test)

    plt.plot(y_true)
    plt.plot(y_pred)
    plt.title('Evaluation')
    plt.legend(['Real', 'Predicted'])
    plt.legend()

    # Save the plot as an image file
    plt.savefig('output_plot.png')
    plt.close()

def build_message(name: str):
    return f"Please wait... Generating plot!"

def build_message_accel(name: str):
    return f"Acceleration: {name}"

def build_message_vel(name: str):
    return f"Init Velocity: {name}"

def build_message_pos(name: str):
    return f"Init Position: {name}"

input_name_data_node_cfg = tp.Config.configure_data_node(id="input_name")
input_name_data_node_cfg2 = tp.Config.configure_data_node(id="input_name2")
input_name_data_node_cfg3 = tp.Config.configure_data_node(id="input_name3")

loading_message_cfg = tp.Config.configure_data_node(id="loading_message")
message_data_node_cfg = tp.Config.configure_data_node(id="message")
message_data_node_cfg2 = tp.Config.configure_data_node(id="message2")
message_data_node_cfg3 = tp.Config.configure_data_node(id="message3")

plot_data_node_cfg = tp.Config.configure_data_node(id="plot")  # Add a data node for the plot

build_loading_msg = tp.Config.configure_task("build_msg", build_message, input_name_data_node_cfg, loading_message_cfg)
build_msg_task_cfg = tp.Config.configure_task("build_msg_accel", build_message_accel, input_name_data_node_cfg, message_data_node_cfg)
build_msg_task_cfg2 = tp.Config.configure_task("build_msg_vel", build_message_vel, input_name_data_node_cfg2, message_data_node_cfg2)
build_msg_task_cfg3 = tp.Config.configure_task("build_msg_pos", build_message_pos, input_name_data_node_cfg3, message_data_node_cfg3)

scenario_cfg = tp.Config.configure_scenario("scenario", task_configs=[build_loading_msg, build_msg_task_cfg, build_msg_task_cfg2, build_msg_task_cfg3])

page = """
Acceleration: <|{input_name}|input|>
Velocity: <|{input_name2}|input|>
Position: <|{input_name3}|input|>
<|submit|button|on_action=submit_scenario|>

<|{loading_message}|text|>

<|{message}|text|>

<|{message2}|text|>

<|{message3}|text|>

<|{plot}|image|> 
"""

input_name = ""
input_name2 = ""
input_name3 = ""

loading_message = None
message = None
message2 = None
message3 = None
plot = None  # Initialize plot

def submit_scenario(state):
    global plot, accel, v0, x0
    state.scenario.input_name.write(state.input_name)
    state.scenario.input_name2.write(state.input_name2)
    state.scenario.input_name3.write(state.input_name3)

    # Update variables with input values
    accel = float(state.input_name)
    v0 = float(state.input_name2)
    x0 = float(state.input_name3)

    state.scenario.submit(wait=True)
    state.loading_message = state.scenario.loading_message.read()
    state.message = state.scenario.message.read()
    state.message2 = state.scenario.message2.read()
    state.message3 = state.scenario.message3.read()

    # Call the function to generate and save the plot
    generate_and_save_plot()

    # Directly assign the plot file path to the state to update it in real-time
    state.plot = 'output_plot.png'

if __name__ == "__main__":
    tp.Core().run()
    scenario = tp.create_scenario(scenario_cfg)
    tp.Gui(page).run()
