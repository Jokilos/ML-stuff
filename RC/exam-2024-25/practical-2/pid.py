class PID:
    def __init__(
            self, gain_prop: int, gain_int: int, gain_der: int, sensor_period: float,
            output_limits: tuple[float, float]
            ):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        self.limits = output_limits
        self.integral = 0

        # TODO: define additional attributes you might need
        # END OF TODO


    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        import numpy as np

        error, prev_error = np.array([commanded_variable] * 2) - sensor_readings

        prop = self.gain_prop * error

        der = (error - prev_error) * self.gain_der / self.sensor_period

        self.integral += (error * self.sensor_period) 
        
        integ = self.integral * self.gain_int

        forward_torque = prop + der + integ

        # print(prop, der, integ)

        return np.clip(forward_torque, self.limits[0], self.limits[1])

    # END OF TODO
