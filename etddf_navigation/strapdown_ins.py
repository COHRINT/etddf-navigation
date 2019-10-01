#!/usr/bin/env python

from __future__ import division

"""
Implementation of an aided strapdown inertial navigation system.
"""

import os
import yaml
import numpy as np
from scipy.linalg import block_diag

# import pudb; pudb.set_trace()

G_ACCEL = 9.80665 # gravitational acceleration [m/s/s]

class StrapdownINS:
    """
    Implementation of an aided strapdown intertial navigation filter.  
    State: [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, 
            q_0, q_1, q_2, q_3, b_ax, b_ay, b_az, b_wx, b_wy, b_wz]

    Parameters
    ----------
    sensors
        list of sensors to be used in filter
    dt
        timestep filter will run at
    init_est [optional]
        state to initialize the filter with
    init_cov [optional]
        covariance to initialize the filter with
    """
    def __init__(self, sensors, dt, init_est=None, init_cov=None):
        # filter timestep
        self.dt = dt

        # sensors
        self.sensors = sensors

        # initialize estimate and covariance
        initial_pos = np.array([0,0,0],ndmin=1)
        initial_vel = np.array([0,0,0],ndmin=1)
        initial_attitude = np.array([0,0,-np.pi/2],ndmin=1) # euler angles [deg]
        initial_accel_bias = np.array([0,0,0],ndmin=1) # initial bias values for accelerometer
        initial_gyro_bias = np.array([0,0,0],ndmin=1) # initial bias values for gyroscope

        # convert initial attitude to quaternion
        init_att_quat = self.euler2quat(initial_attitude,deg=True)

        self.x = np.concatenate([initial_pos,
                                initial_vel,
                                init_att_quat,
                                initial_accel_bias,
                                initial_gyro_bias])

        init_pos_cov = 1000*np.eye(3)
        init_vel_cov = 10*np.eye(3)
        init_att_cov = 1*np.eye(4)
        init_ab_cov = 1*np.eye(3)
        init_gb_cov = 1*np.eye(3)

        self.P = block_diag(init_pos_cov,
                            init_vel_cov,
                            init_att_cov,
                            init_ab_cov,
                            init_gb_cov)

        # initialize xdot vector
        self.xdot = np.zeros([16])

    def get_estimate(self,cov=False):
        """
        Return estimate and (optionally) covariance.
        """
        if cov:
            return self.x, self.P
        else:
            return self.x

    def propagate(self,imu_measurement):
        """
        Propagate state and covariance forward in time by dt using process model
        and IMU measurement.

        Parameters
        ----------
        imu_measurement
            Accelerometer and gyroscope measurements. numpy array of form:
            [a_x, a_y, a_z, w_x, w_y, w_z]
        """
        # save last xdot
        last_xdot = self.xdot

        # extract relevant estimate states, and measurement elements for clarity
        x_pos = self.x[0]; y_pos = self.x[1]; z_pos = self.x[2]
        x_vel = self.x[3]; y_vel = self.x[4]; z_vel = self.x[5]
        q0 = self.x[6]; q1 = self.x[7]; q2 = self.x[8]; q3 = self.x[9]
        b_ax = self.x[10]; b_ay = self.x[11]; b_az = self.x[12]
        b_wx = self.x[13]; b_wy = self.x[14]; b_wz = self.x[15]

        a_x = imu_measurement[0]; a_y = imu_measurement[1]; a_z = imu_measurement[2]
        w_x = imu_measurement[3]; w_y = imu_measurement[4]; w_z = imu_measurement[5]
        
        # TODO: check the mismatches in biases and gyro measurements
        quaterion_stm = np.array([[0, -(w_x-b_wx), -(w_y-b_wy), -(w_z-b_wz)],
                                    [w_x-b_wx, 0, w_z-b_wz, -(w_y-b_wy)],
                                    [w_y-b_wy, -(w_z-b_wz), 0, w_x-b_wx],
                                    [w_z-b_wz, w_y-b_wy, -(w_x-b_wx), 0]])

        # transformation from body frame to inertial frame
        body2inertial = np.array([[1-2*(q2**2+q3**2),2*(q1*q2-q0*q3),2*(q1*q3+q0*q2)],
                                    [2*(q1*q2+q0*q3),1-2*(q1**2+q3**2),2*(q2*q3-q0*q1)],
                                    [2*(q1*q3-q0*q2),2*(q2*q3-q0*q1),1-2*(q1**2+q2**2)]])

        # create updated xdot
        self.xdot = np.zeros([16])

        pos_dot = np.array([x_vel,y_vel,z_vel])
        vel_dot = np.dot(body2inertial,(np.array([a_x,a_y,a_z])-np.array([b_ax,b_ay,b_az])).transpose()) + np.array([0,0,G_ACCEL])
        q_dot = 0.5*np.dot(quaterion_stm,np.array([q0,q1,q2,q3]).transpose())
        # q_dot = 0.5*quaterion_stm

        self.xdot[0:3] = pos_dot
        self.xdot[3:6] = vel_dot
        self.xdot[6:10] = q_dot

        # derivative of attitude change wrt itself
        F_att_der = 0.5*np.array([[0, b_wx-w_x, b_wy-w_y, b_wz-w_z],
                            [-b_wx+w_x, 0, -b_wz+w_z, b_wy-w_y],
                            [-b_wy+w_y, b_wz-w_z, 0, -b_wx+w_x],
                            [-b_wz+w_z, -b_wy+w_y, b_wx-w_x,0]])

        # derivative of attitude change wrt gyro bias estimate
        F_att_gb_der = 0.5*np.array([[q1,q2,q3],
                            [-q0,q3,-q2],
                            [-q3,-q0,q1],
                            [q2,-q1,-q0]])

        # convenience definitions of bias-corrected IMU acceleration measurements
        al_x = b_ax-a_x; al_y = b_ay-a_y; al_z = b_az-a_z

        # derivative of velocity wrt attitude
        F_vel_att_der = np.array([[2*(q3*al_y-q2*al_z), -2*(q2*al_y+q3*al_z), 2*(2*q2*al_x-q1*al_y-q0*al_z), 2*(2*q3*al_x+q0*al_y-q1*al_z)],
                        [2*(q1*al_z-q3*al_x), 2*(2*q1*al_y-q2*al_x+q0*al_z), -2*(q1*al_x+q3*al_z), 2*(2*q3*al_y-q0*al_x-q2*al_z)],
                        [2*(q2*al_x-q1*al_y), 2*(2*q1*al_z-q3*al_x-q0*al_y), 2*(2*q2*al_z-q3*al_y+q0*al_x), -2*(q1*al_x+q2*al_y)]])

        # derivative of velocity wrt accelerometer bias estimate
        F_vel_ab_der = np.array([[-1+2*(q2**2+q3**2), -2*(q1*q2-q0*q3), -2*(q1*q3+q0*q2)],
                        [-2*(q0*q3+q1*q2), -1+2*(q1**2+q3**2), -2*(q2*q3-q0*q1)],
                        [-2*(q1*q3-q0*q2), -2*(q0*q1+q2*q3), -1+2*(q1**2+q2**2)]])

        # derivative of position wrt velocity
        F_pos_der = np.eye(3)

        # construct linearized dynamics stm
        F = np.zeros([16,16])
        F[0:3,3:6] = F_pos_der
        F[3:6,6:10] = F_vel_att_der
        F[3:6,10:13] = F_vel_ab_der
        F[6:10,6:10] = F_att_der
        F[6:10,13:16] = F_att_gb_der

        # construct process noise matrix
        Q = np.zeros((16,16))
        Q[3:6,3:6] = 0.1*np.array(self.sensors['IMU'].accel_noise)
        Q[6:10,6:10] = 0.1*self.sensors['IMU'].gyro_noise[0][0]*np.eye(4)

        # numerically integrate dynamics model to propagate estimate and covariance
        # Johnson paper: implements integration with Euler 2nd order for estimate, Euler 1st order for covariance 
        # TODO: implement RK4 or other method
        self.x = self.x + 0.5*self.dt*(self.xdot + last_xdot)
        self.P = self.P + (np.dot(F,self.P)+np.dot(self.P,F.transpose())+Q)*self.dt

        # enforce symmetry
        self.P = 0.5*self.P + 0.5*self.P.transpose()

        # k1 = self.dt*self.xdot
        # self.x = self.x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

        # renormalize quaternion attitude
        self.x[6:10] = self.x[6:10]/np.linalg.norm(self.x[6:10])

    def update(self,measurement,measurement_type):
        """
        Correct estimate using available aiding measurements.

        Parameters
        ----------
        measurements
            available measurements at current timestep
        """
        if measurement_type == 'GPS':
            H = np.zeros((3,16))
            H[0:3,0:3] = np.eye(3)
            R = self.sensors['GPS'].noise
            h = self.x[0:3]
        elif measurement_type == 'Depth':
            H = np.zeros((1,16))
            H[0,2] = 1
            R = self.sensors['Depth'].noise
            h = self.x[2]
        # elif measurement_type == 'Compass':
            # H = np.zeros((1,16))
            # H[0,6] = 

        # compute the Kalman gain for the measurement
        K = np.dot(self.P,np.dot(H.transpose(),np.linalg.inv(np.dot(H,np.dot(self.P,H.transpose()))+R)))
        print(K)
        
        try:
            assert(K.shape == (self.x.shape[0],H.shape[0]))
        except AssertionError:
            print('K is the wrong shape!: Is {}, should be {}'.format(K.shape,(self.x.shape[0],H.shape[0])))
            raise AssertionError

        x = np.atleast_2d(self.x).transpose() + np.dot(K,(measurement-h).transpose())
        self.x = np.squeeze(x)
        # self.P = np.dot(np.eye(16)-np.dot(K,H),self.P)
        # joseph form
        self.P = np.dot(np.eye(16)-np.dot(K,H),np.dot(self.P,(np.eye(16)-np.dot(K,H)).transpose())) + np.dot(K,np.dot(R,K.transpose()))
        # enforce symmetry
        self.P = 0.5*self.P + 0.5*self.P.transpose()

        # renormalize quaternion attitude
        print(np.linalg.norm(self.x[6:10]))
        print(self.x[6:10]/np.linalg.norm(self.x[6:10]))
        self.x[6:10] = self.x[6:10]/np.linalg.norm(self.x[6:10])

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

    def euler2quat(self,angles,deg=False):
        """
        Convert euler angle representation to quaternion.
        From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        
        Parameters
        ----------
        angles
            vector of euler angles: follows [roll, pitch, yaw] convention
        deg (optional)
            units of angles -- defaults to False for radians

        Returns
        -------
        quat
            quaternion representation: follows [q0, q1, q2, q3]
        """
        # extract individual angles
        roll = angles[0]; pitch = angles[1]; yaw = angles[2]

        # check if conversion from radians to degress is necessary
        if deg:
            roll *= np.pi/180
            pitch *= np.pi/180
            yaw *= np.pi/180

        # convenience intermediate calculations
        cy = np.cos(0.5*yaw)
        sy = np.sin(0.5*yaw)
        cp = np.cos(0.5*pitch)
        sp = np.sin(0.5*pitch)
        cr = np.cos(0.5*roll)
        sr = np.sin(0.5*roll)

        q0 = cy*cp*cr + sy*sp*sr
        q1 = cy*cp*sr - sy*sp*cr
        q2 = sy*cp*sr + cy*sp*cr
        q3 = sy*cp*cr - cy*sp*sr

        return np.array([q0,q1,q2,q3],ndmin=1)

    def quat2euler(self,quat,deg=False):
        """
        Convert quaternion representation to euler angles.
        From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        Parameters
        ----------
        quat
            quaternion representation: follows [q0, q1, q2, q3]
        deg (optional)
            units of output euler angles -- defaults to False for radians

        Returns
        -------
        angles
            vector of euler angles: follows [roll, pitch, yaw] convention
        """
        # extract quaternion components
        [q0,q1,q2,q3] = quat

        # roll
        sinr_cosp = 2*(q0*q1 + q2*q3)
        cosr_cosp = 1-2*(q1**2 + q2**2)
        roll = np.arctan2(sinr_cosp,cosr_cosp)

        # pitch
        sinp = 2*(q0*q2 - q3*q1)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi/2,sinp)
        else:
            pitch = np.arcsin(sinp)

        # yaw
        siny_cosp = 2*(q0*q3 + q1*q2)
        cosy_cosp = 1-2*(q2**2 + q3**2)
        yaw = np.arctan2(siny_cosp,cosy_cosp)

        if deg:
            roll *= 180/np.pi
            pitch *= 180/np.pi
            yaw *= 180/np.pi

        return np.array([roll,pitch,yaw],ndmin=1)

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

        gyro_gt = np.array([0,0,ground_truth[10]]) - self.last_gyro_state
        gyro_bias = self.last_gyro_bias + np.exp(-(1/self.rate)/self.gyro_bias_tc)*np.random.multivariate_normal([0,0,0],np.eye(3)*self.gyro_bias)
        gyro_noise = np.random.multivariate_normal([0,0,0],self.gyro_noise)
        gyro_meas = gyro_gt + gyro_bias + gyro_noise

        self.last_gyro_bias = gyro_bias
        self.last_gyro_state = np.array([0,0,ground_truth[10]])

        return accel_meas, gyro_meas, accel_gt, gyro_gt, accel_bias, gyro_bias

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
        measurement = np.mod(measurement,2*np.pi)
        return measurement

class Depth(Sensor):

    def __init__(self,cfg_path=None):
        super(Depth,self).__init__(cfg_path=cfg_path)

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
        true_depth = np.array([ground_truth[2]])
        noise = np.random.normal(0,self.noise)
        measurement = true_depth + noise
        return measurement

def main():
    g = GPS()
    i = IMU()
    c = Compass()

    nf = StrapdownINS([g,i,c],0.1)

    euler_angles = [1.75,0.4938,2.4583]
    quat = nf.euler2quat(euler_angles)
    euler_cov = nf.quat2euler(quat)

    print(euler_angles)
    print(euler_cov)

if __name__ == "__main__":
    main()