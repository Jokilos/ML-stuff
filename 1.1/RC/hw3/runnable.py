import mujoco
from mujoco import viewer
import numpy as np
import plotly.express as px
import contextlib, io
from drone_simulator import DroneSimulator
from pid import PID
import sys

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# viewer = viewer.launch_passive(model, data)
# viewer.cam.distance = 4.
# viewer.cam.lookat = np.array([0, 0, 1])
# viewer.cam.elevation = -30.

desired_altitude = 2

drone_simulator = DroneSimulator(
    model, data, viewer, desired_altitude = desired_altitude,
    altitude_sensor_freq = 0.01,
    wind_change_prob = 0.1,
    rendering_freq = 20,
)

innerPID_freq = 1

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python script.py <float1> <float2>")
        sys.exit(1)
    try:
        prop = float(sys.argv[1])
        integ = float(sys.argv[2])

    except ValueError:
        sys.exit(1)

    innerPID = PID(
        gain_prop = prop,
        gain_int = integ,
        gain_der = 0,
        sensor_period = drone_simulator.altitude_sensor_period * (0.01 / innerPID_freq),
    )

    outerPID = PID(
        gain_prop = 0.0286666,
        gain_int = 0,
        gain_der = 0.27333,
        sensor_period = drone_simulator.altitude_sensor_period,
    )

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

    from types import MethodType 
    drone_simulator.acceleration_sensor = MethodType(acceleration_sensor, drone_simulator)

    thrust = 0
    prev_alt_measurement = -1
    alts = []

    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(4000):
            # TODO: Use the PID controllers in a cascade designe to control the drone

            drone_simulator.acceleration_sensor()

            if drone_simulator.measured_altitudes[0] != prev_alt_measurement:
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

            drone_simulator.sim_step(thrust, view = False)

    alts = np.array(alts[-1500:]) - desired_altitude
    error = np.sum(np.abs(alts))
    print(error)