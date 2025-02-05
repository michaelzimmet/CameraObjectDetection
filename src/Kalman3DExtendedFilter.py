import numpy as np
from filterpy.kalman import KalmanFilter


class Kalman3D:
    def __init__(self):
        # dim_x = (x, y, z, vx, vy, vz), dim_z = (x, y, z)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # Time between two frames
        delta_time = 0.7
        self.delta_time = delta_time

        # State Matrix containing the current position and velocity for x, y, z
        #                      x, y, z,         vx,         vy,          vz
        self.kf.F = np.array([[1, 0, 0, delta_time,          0,          0],
                              [0, 1, 0,          0, delta_time,          0],
                              [0, 0, 1,          0,          0, delta_time],
                              [0, 0, 0,          1,          0,          0],
                              [0, 0, 0,          0,          1,          0],
                              [0, 0, 0,          0,          0,          1]])

        # Measurement Matrix containing only given sensor values. In this case x, y, z
        self.kf.H = np.array([[1, 0, 0,          0,          0,          0],
                              [0, 1, 0,          0,          0,          0],
                              [0, 0, 1,          0,          0,          0]])

        # Sensor noise -> if value < 1, the sensor is trusted more
        self.kf.R *= 0.1

        # Process noise ->
        self.kf.Q = np.eye(6) * 0.01

        # Start noise ->
        #self.kf.P *= .1

    def initialize(self, x, y, z):
        """
        Initialize the start position of a detected Object
        :param x: x coordinate
        :param y: y coordinate
        :param z: depth coordinate
        :return: None
        """
        self.kf.x[:] = np.array([[x], [y], [z], [0], [0], [0]])

    def predict(self):
        """
        Predicts the next position of the object based on the current position and velocity.
        :return: List containing the new position of x, y, z
        """
        self.kf.predict()
        return self.kf.x[:3].flatten()

    def update(self, new_position):
        """
        Update the current position and velocity of the object based on the given measurement
        :param new_position: new position of the object if its moving. Contains x, y, z coordinates
        :return: None
        """
        self.kf.update(np.array(new_position).reshape(3, 1))
