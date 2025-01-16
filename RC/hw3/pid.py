# TODO: implement a class for PID controller
import numpy as np

class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.timestep = sensor_period
        self.integral = 0

        # TODO: add aditional variables to store the current state of the controller

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):

        error, prev_error = np.array([commanded_variable] * 2) - sensor_readings

        prop = self.gain_prop * error

        der = (error - prev_error) * self.gain_der / self.timestep 

        self.integral += (error * self.timestep) 
        
        integ = self.integral * self.gain_int

        print(prop, der, integ)

        forward_torque = prop + der + integ

        return forward_torque