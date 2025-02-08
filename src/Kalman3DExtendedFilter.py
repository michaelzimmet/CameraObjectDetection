import numpy as np
from filterpy.kalman import KalmanFilter


class Kalman3D:
    def __init__(self, pred_class, delta_time=0, moving_objects=False):
        # dim_x = (x, y, z, vx, vy, vz), dim_z = (x, y, z)
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.pred_class = pred_class
        # Time between two frames
        self.delta_time = delta_time
        self.history = []
        moving_objects = int(moving_objects == True)
        # State Matrix containing the current position and velocity for x, y, z
        #                      x, y, z,         vx,         vy,          vz
        self.kf.F = np.array([[1, 0, 0,     delta_time,              0,              0],
                              [0, 1, 0,              0,     delta_time,              0],
                              [0, 0, 1,              0,              0,     delta_time],
                              [0, 0, 0, moving_objects,              0,              0],
                              [0, 0, 0,              0, moving_objects,              0],
                              [0, 0, 0,              0,              0, moving_objects]])

        # Measurement Matrix containing only given sensor values. In this case x, y, z
        self.kf.H = np.array([[1, 0, 0,          0,          0,          0],
                              [0, 1, 0,          0,          0,          0],
                              [0, 0, 1,          0,          0,          0]])

        # Sensor noise -> if value < 1, the sensor is trusted more
        self.kf.R *= 0.5

        # Process noise -> the smaller the value, the more the model assumes that the objects are static
        #self.kf.Q = np.eye(6) * .1
        self.kf.Q = np.eye(6) * 0.001
        # Start noise
        #self.kf.P *= 10

    def initialize(self, x, y, z):
        """
        Initialize the start position of a detected Object
        :param x: x coordinate
        :param y: y coordinate
        :param z: depth coordinate
        :return: None
        """
        self.kf.x[:] = np.array([[x], [y], [z], [0], [0], [0]])
        self.history.append(np.array([x, y, z]))

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
        self.history.append(np.array(new_position))
