#!/usr/bin/env python

from __future__ import division

"""
Implementation of an aided strapdown inertial navigation system.
"""

import os
import yaml
import numpy as np

G_ACCEL = 9.806

class StrapdownINS:

    def __init__(self):
        pass

    def propagate(self):
        pass

    def update(self):
        pass

    def compute_gravity_vector(self,estimate):
        """
        Compute the gravity vector in both the navigation and the body frames,
        in order to compensate acceleration measurements.

        Parameters
        ----------
        estimate
            most current state estimate, used for both orientation and location
            (if using higher fidelity gravity model)

        Returns
        -------
        grav_nav
            gravity vector in navigation frame
        grav_body
            gravity vector in body frame
        """
        pass

class Sensor(object):

    def __init__(self,cfg_path=None):
        if cfg_path is None:
            filename = os.path.abspath(os.path.join(os.path.dirname(__file__),
                        '../config/'+type(self).__name__.lower()+'.yaml'))
        else:
            filename = os.path.abspath(cfg_path)

        if os.path.exists(filename):
            cfg = self.load_config(filename)
        else:
            print('-------')
            print('Couldn\'t load file with path {}.'.format(filename))
            print('Please provide a sensor configuration file at the above location.')
            print('-------')
            raise IOError

        self.rate = cfg['rate'] # sensor update rate
        self.noise = np.array(cfg['noise']) # sensor white gaussian random noise covariance matrix
        # self.error_models = cfg['error_models'] # error models used in sensor, e.g. 1st order Gauss-Markov, etc.

        self.cfg = cfg

    def load_config(self,filename):
        """
        Loads sensor config.
        """
        with open(filename,'r') as f:
            cfg = yaml.load(f,Loader=yaml.SafeLoader)
        return cfg

    def gen_measurement(self):
        raise NotImplementedError

class IMU(Sensor):

    def __init__(self,cfg_path=None):
        super(IMU,self).__init__(cfg_path=cfg_path)

        self.accel_scale_factor = self.cfg['accel_scale_factor']
        self.accel_bias = self.cfg['accel_bias']
        self.accel_bias_tc = self.cfg['accel_bias_tc']
        self.accel_noise = self.cfg['accel_noise']

        self.gyro_scale_factor = self.cfg['gyro_scale_factor']
        self.gyro_bias = self.cfg['gyro_bias']
        self.gyro_bias_tc = self.cfg['gyro_bias_tc']
        self.gyro_noise = self.cfg['gyro_noise']


        self.last_accel_state = np.zeros((1,3))
        self.last_gyro_state = np.zeros((1,3))
        
        self.last_accel_bias = np.zeros((1,3))
        self.last_gyro_bias = np.zeros((1,3))

    def gen_measurement(self,ground_truth):
        """
        Generate simulated IMU measurements (accelerometer and gyroscope).
        Uses random walk drift models for accel and gyro.

        Parameters
        ----------
        ground_truth
            true 12 DOF state of vehicle

        Returns
        -------
        measurement
            computed measurement for sensor
        """
        accel_gt = np.array([ground_truth[1],ground_truth[3],ground_truth[5] - G_ACCEL]) - self.last_accel_state
        accel_bias = self.last_accel_bias + np.exp(-(1/self.rate)/self.accel_bias_tc)*np.random.multivariate_normal([0,0,0],np.eye(3)*self.accel_bias)
        accel_noise = np.random.multivariate_normal([0,0,0],self.accel_noise)
        accel_meas = accel_gt + accel_bias + accel_noise

        self.last_accel_bias = accel_bias
        self.last_accel_state = np.array([ground_truth[1],ground_truth[3],ground_truth[5]])

        return accel_meas

class GPS(Sensor):

    def __init__(self,cfg_path=None):
        super(GPS,self).__init__(cfg_path=cfg_path)

    def gen_measurement(self,ground_truth):
        """
        Generate simulated GPS measurement using ground truth data 
        and gaussian noise.

        Parameters
        ----------
        ground_truth
            true 12 DOF state of vehicle

        Returns
        -------
        measurement
            computed measurement for sensor
        """
        true_pos = np.array([ground_truth[0],ground_truth[2],ground_truth[4]])
        noise = np.random.multivariate_normal([0,0,0],self.noise)
        measurement = true_pos + noise
        return measurement

class Compass(Sensor):

    def __init__(self,cfg_path=None):
        super(Compass,self).__init__(cfg_path=cfg_path)

    def gen_measurement(self,ground_truth):
        """
        Generate simulated heading measurement using ground truth data 
        and gaussian noise.

        Parameters
        ----------
        ground_truth
            true 12 DOF state of vehicle

        Returns
        -------
        measurement
            computed measurement for sensor
        """
        true_heading = np.array([ground_truth[10]])
        noise = np.random.normal(0,self.noise)
        measurement = true_heading + noise
        return measurement

def main():
    GPS()
    IMU()
    Compass()

if __name__ == "__main__":
    main()