# TODO: implement a class for PID controller
class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.timestep = sensor_period
        self.prev_error = 0
        self.integral = 0

        # TODO: add aditional variables to store the current state of the controller

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):
        error = commanded_variable - sensor_readings[-1]

        if self.prev_error == 0:
            self.prev_error = error

        prop = self.gain_prop * error

        der = (error - self.prev_error) * self.gain_der / self.timestep 

        self.integral += (error * self.timestep) 
        
        integ = self.integral * self.gain_int

        # print(prop, der, integ, sensor_readings[-1])

        forward_torque = prop + der + integ

        self.prev_error = error

        return forward_torque