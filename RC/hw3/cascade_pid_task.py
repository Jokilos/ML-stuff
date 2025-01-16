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
        altitude_sensor_freq = 0.01,
        wind_change_prob = 0.1,
        rendering_freq = 1,
    )

    # TODO: Create necessary PID controllers using PID class
    innerPID_freq = 1

    innerPID = PID(
        gain_prop = 0.3,
        gain_int = 0.2,
        gain_der = 0,
        sensor_period = drone_simulator.altitude_sensor_period * (0.01 / innerPID_freq),
    )

    outerPID = PID(
        gain_prop = 0.7,
        gain_int = 0.7,
        gain_der = 0.8,
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

    # Increase the number of iterations for longer simulation
    # for i in range(2000):
        # desired_thrust = pid_altitude.output_signal(desired_altitude, drone_simulator.measured_altitudes)
        # drone_simulator.sim_step(desired_thrust)

    # Increase the number of iterations for a longer simulation
    thrust = 0

    for i in range(4000):
        # TODO: Use the PID controllers in a cascade designe to control the drone

        drone_simulator.acceleration_sensor()
        threshhold = 600

        divisor = 1 if i < threshhold else -1#(i - threshhold + 1)
        wanted_acc = slow_ascend_acceleration / divisor

        thrust_addition = innerPID.output_signal(
            wanted_acc,
            drone_simulator.measured_acceleration,
        )

        thrust += thrust_addition

        print(thrust_addition)
        print(drone_simulator.measured_acceleration)
        print(thrust)
        print(f"{wanted_acc=}")
        print("")

        drone_simulator.sim_step(thrust)
        # drone_simulator.sim_step(desired_thrust)