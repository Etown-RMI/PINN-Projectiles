import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pinns
import gif
import taipy as tp
from taipy.gui import invoke_long_callback, notify
matplotlib.use('agg')


model = None
x = pinns.Domain(0, 10, 100)
accel = -9.8
v0 = 10
x0 = 0
epo = 2000


def pde(x, y):
    return pinns.Grad.hessian(y, x) - accel


def generate_and_save_plots():
    global model
    ic1 = pinns.IC(x_ic=0, f=v0, y_der=1)
    ic2 = pinns.IC(x_ic=0, f=x0, y_der=0)

    model = pinns.net(inputs=1, layers=2 *
                      [32], activation='softplus', outputs=1)
    pinns.train(model, x, pde, [ic1, ic2], epochs=epo, lr=0.01)

    x_test = np.linspace(0, 10, 100)
    y_true = (accel/2)*x_test**2 + v0*x_test + x0
    y_pred = model(x_test)
    y_pred_np = y_pred.numpy().flatten()
    dy_true = np.gradient(y_true, x_test)
    ddy_true = np.gradient(dy_true, x_test)
    dy_pred = np.gradient(y_pred_np, x_test)
    ddy_pred = np.gradient(dy_pred, x_test)

    plt.plot(y_true)
    plt.plot(y_pred)
    plt.title('Position vs Time')
    plt.legend(['Real', 'Predicted'])

    # Save the plot as an image file
    plt.savefig('position_plot.png')
    plt.close()

    plt.plot(dy_true)
    plt.plot(dy_pred)
    plt.title('Velocity vs Time')
    plt.legend(['Real', 'Predicted'])

    # Save the plot as an image file
    plt.savefig('velocity_plot.png')
    plt.close()

    plt.plot(ddy_true)
    plt.plot(ddy_pred)
    plt.title('Acceleration vs Time')
    plt.legend(['Real', 'Predicted'])

    # Save the plot as an image file
    plt.savefig('acceleration_plot.png')
    plt.close()


def gen_and_save_gif():
    global model, accel, v0, x0
    x_test = np.linspace(0, 10, 100)
    y_true = (accel/2)*x_test**2 + v0*x_test + x0
    y_pred = model(x_test)

    @gif.frame
    def plot_gif(i):
        predi = y_pred[i]
        truei = y_true[i]
        plt.scatter(truei, 2, c="blue")
        plt.scatter(predi, 2, c="orange")
        plt.ylim(0, 4)
        plt.xlim(min(y_true), max(y_true))
        plt.title('Simulation')
        plt.legend(['Real', 'Predicted'])

    frames = [plot_gif(i) for i in range(100)]
    gif.save(frames, "sim.gif", duration=100)


def build_message(name: str):
    return f" "


def build_message_accel(name: str):
    return f"A: {name}"


def build_message_vel(name: str):
    return f"V0: {name}"


def build_message_pos(name: str):
    return f"X0: {name}"


def build_message_epo(name: str):
    return f"Epochs: {name}"


input_name_data_node_cfg = tp.Config.configure_data_node(id="input_name")
input_name_data_node_cfg2 = tp.Config.configure_data_node(id="input_name2")
input_name_data_node_cfg3 = tp.Config.configure_data_node(id="input_name3")
input_name_data_node_cfg4 = tp.Config.configure_data_node(id="input_name4")


loading_message_cfg = tp.Config.configure_data_node(id="loading_message")
message_data_node_cfg = tp.Config.configure_data_node(id="message")
message_data_node_cfg2 = tp.Config.configure_data_node(id="message2")
message_data_node_cfg3 = tp.Config.configure_data_node(id="message3")
message_data_node_cfg4 = tp.Config.configure_data_node(id="message4")

plot_data_node_cfg = tp.Config.configure_data_node(id="plot")


build_loading_msg = tp.Config.configure_task(
    "build_msg", build_message, input_name_data_node_cfg, loading_message_cfg)
build_msg_task_cfg = tp.Config.configure_task(
    "build_msg_accel", build_message_accel, input_name_data_node_cfg, message_data_node_cfg)
build_msg_task_cfg2 = tp.Config.configure_task(
    "build_msg_vel", build_message_vel, input_name_data_node_cfg2, message_data_node_cfg2)
build_msg_task_cfg3 = tp.Config.configure_task(
    "build_msg_pos", build_message_pos, input_name_data_node_cfg3, message_data_node_cfg3)
build_msg_task_cfg4 = tp.Config.configure_task(
    "build_msg_epo", build_message_epo, input_name_data_node_cfg4, message_data_node_cfg4)

scenario_cfg = tp.Config.configure_scenario("scenario", task_configs=[
                                            build_loading_msg, build_msg_task_cfg, build_msg_task_cfg2, build_msg_task_cfg3, build_msg_task_cfg4])


page = """

<|layout|columns=1 1fr 1|
<|part|>
<|part|
<center><h1>PINN Projectiles</h1></center>
|>
<|toggle|theme|class_name=nolabel|>
|>

<|layout|
<|card|
<center><|     Acceleration: |> <|{input_name}|input|></center>

<center><| Initial Velocity: |> <|{input_name2}|input|></center>

<center><| Initial Position: |> <|{input_name3}|input|></center>

<center><|  Training Epochs: |> <|{input_name4}|input|></center>

<center><|submit|button|on_action=submit_scenario|></center>
|>
<|card|
<|{loading_message}|text|>

<center><|{sim}|image|></center>
<|layout|columns=4*1|gap=5px|
<|{message}|text|>

<|{message2}|text|>

<|{message3}|text|>

<|{message4}|text|>
|>
|>
|>
<|Comparison Plots|expandable|
<center>
<|{pos_plot}|image|>
<|{vel_plot}|image|>
<|{acc_plot}|image|>
</center>
|>
"""


input_name = "-9.8"
input_name2 = "100"
input_name3 = "0"
input_name4 = "2000"


loading_message = None
message = 'A: -9.8'
message2 = 'V0: 100'
message3 = 'X0: 0'
message4 = 'Epochs: 2000'
pos_plot = 'position_plot.png'
vel_plot = 'velocity_plot.png'
acc_plot = 'acceleration_plot.png'
sim = 'sim.gif'


def heavy_function_status(state, status):
    if status:
        state.pos_plot = 'position_plot.png'
        state.vel_plot = 'velocity_plot.png'
        state.acc_plot = 'acceleration_plot.png'
        notify(state, "Success!", f"The model has finished!")
        invoke_long_callback(state, gen_and_save_gif,
                             [], heavy_function_status2)
        notify(state, "Please Wait!", "The simulation is generating.")
    else:
        notify(state, "Error", f"The model has failed somehow...")


def heavy_function_status2(state, status):
    if status:
        state.sim = 'sim.gif'
        notify(state, "Success!", f"The simulation has finished!")
    else:
        notify(state, "Error", f"The simulation has failed somehow...")


def submit_scenario(state):
    global pos_plot, accel, v0, x0, epo
    state.scenario.input_name.write(state.input_name)
    state.scenario.input_name2.write(state.input_name2)
    state.scenario.input_name3.write(state.input_name3)
    state.scenario.input_name4.write(state.input_name4)

    state.pos_plot = 'loading_gears.gif'
    state.vel_plot = 'loading_gears.gif'
    state.acc_plot = 'loading_gears.gif'
    state.sim = 'loading_gears.gif'

    # Update variables with input values
    accel = float(state.input_name)
    v0 = float(state.input_name2)
    x0 = float(state.input_name3)
    epo = int(state.input_name4)

    state.scenario.submit(wait=True)
    state.loading_message = state.scenario.loading_message.read()
    state.message = state.scenario.message.read()
    state.message2 = state.scenario.message2.read()
    state.message3 = state.scenario.message3.read()
    state.message4 = state.scenario.message4.read()

    # Call the function to generate and save the plot
    notify(state, "Please Wait!", f"The model is training.")
    invoke_long_callback(state, generate_and_save_plots,
                         [], heavy_function_status)


if __name__ == "__main__":
    tp.Core().run()
    scenario = tp.create_scenario(scenario_cfg)
    dark_theme = {"palette": {"background": {
        "default": "#2e2c9a"}, "primary": {"main": "#eec8ed"}, }}
    light_theme = {"palette": {"primary": {"main": "#382d72"}, }}
    stylekit = {'color_paper_dark': '#120348',
                'color_paper_light': '#a080e1', }

    tp.Gui(page).run(
        host="0.0.0.0",
        title="PINN Projectiles",
        watermark="© 2024 Etown RMI",
        favicon="./logo.png",
        dark_theme=dark_theme,
        light_theme=light_theme,
        stylekit=stylekit,
    )
