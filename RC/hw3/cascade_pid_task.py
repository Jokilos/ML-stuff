import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)
viewer.cam.distance = 4.
viewer.cam.lookat = np.array([0, 0, 1])
viewer.cam.elevation = -30.

from drone_simulator import DroneSimulator
from pid import PID

if __name__ == '__main__':
    desired_altitude = 2

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, viewer, desired_altitude = desired_altitude,
        altitude_sensor_freq = 0.01, wind_change_prob = 0.1, rendering_freq = 1
        )

    # TODO: Create necessary PID controllers using PID class
    import plotly.express as px
    from types import MethodType 

    innerPID_freq = 1

    innerPID = PID(
        gain_prop = 5/9,
        gain_int = 2/5,
        gain_der = 0,
        sensor_period = drone_simulator.altitude_sensor_period * (0.01 / innerPID_freq),
    )

    outerPID = PID(
        gain_prop = 0.028 + 2/3000,
        gain_int = 0,
        gain_der = 0.27 + 1/300,
        sensor_period = drone_simulator.altitude_sensor_period,
    )

    # not used in final solution, but important for finding constants
    fast_ascend_thrust = 3.3
    slow_ascend_thrust = 3.25
    slow_ascend_acceleration = 0.15

    G = 9.81
    
    initial_acc = drone_simulator.data.sensor("body_linacc").data[2] - G
    drone_simulator.measured_acceleration = np.array([initial_acc] * 2, dtype = float)

    def acceleration_sensor(self):
        self.measured_acceleration[1] = self.measured_acceleration[0]
        self.measured_acceleration[0] = self.data.sensor("body_linacc").data[2] - G
        return self.measured_acceleration

    drone_simulator.acceleration_sensor = MethodType(acceleration_sensor, drone_simulator)

    # Increase the number of iterations for a longer simulation

    thrust = 0
    alts = []

    for i in range(4000):
        # TODO: Use the PID controllers in a cascade designe to control the drone

        drone_simulator.acceleration_sensor()

        wanted_acc = outerPID.output_signal(
            desired_altitude,
            drone_simulator.measured_altitudes
        )

        thrust_addition = innerPID.output_signal(
            wanted_acc,
            drone_simulator.measured_acceleration,
        )

        thrust += thrust_addition

        alts += [drone_simulator.altitude_history[0]]

        drone_simulator.sim_step(thrust)

# Uncomment to see altitude plot
# fig = px.line(y=alts, labels={'y':'altitude'})
# fig.show()