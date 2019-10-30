#!/usr/bin/env python

from __future__ import division

"""
Implementation of an aided strapdown inertial navigation system.
"""

import os
import yaml
import numpy as np
from scipy.linalg import block_diag, sqrtm

# import pudb; pudb.set_trace()

G_ACCEL = 9.80665 # gravitational acceleration [m/s/s]
# G_ACCEL = 9.799999981484898

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
        if init_est is not None:
            self.x = init_est
        else:
            initial_pos = np.array([0.0,0.0,0.0],ndmin=1)
            initial_vel = np.array([0.0,0.0,0.0],ndmin=1)
            initial_attitude = np.array([0.0,0.0,0.0],ndmin=1) # euler angles [deg]
            initial_accel_bias = np.array([0.0,0.0,0.0],ndmin=1) # initial bias values for accelerometer
            initial_gyro_bias = np.array([0.0,0.0,0.0],ndmin=1) # initial bias values for gyroscope

            # convert initial attitude to quaternion
            # init_att_quat = self.euler2quat(initial_attitude,deg=True)

            self.x = np.concatenate([initial_pos,
                                    initial_vel,
                                    initial_attitude,
                                    initial_accel_bias,
                                    initial_gyro_bias])

        init_pos_cov = 10*np.eye(3)
        init_vel_cov = 5*np.eye(3)
        init_att_cov = ((5*np.pi/180)**2)*np.eye(3)
        init_ab_cov = 1e-5*np.eye(3)
        init_gb_cov = 1e-5*np.eye(3)

        self.P = block_diag(init_pos_cov,
                            init_vel_cov,
                            init_att_cov,
                            init_ab_cov,
                            init_gb_cov)

        # initialize xdot vector
        self.xdot = np.zeros([15])

        self.dcm = self.ypr_rotation_b2r(initial_attitude[0],initial_attitude[1],initial_attitude[2])

        # measurement residual storage
        self.gps_residuals = np.empty((1,3))
        self.depth_residuals = np.empty((1,1))
        self.compass_residuals = np.empty((1,1))

        # Kalman gain storage
        self.gps_gains = np.empty((15,3,1))

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
        roll = self.x[6]; pitch = self.x[7]; yaw = self.x[8]
        b_ax = self.x[9]; b_ay = self.x[10]; b_az = self.x[11]
        b_wx = self.x[12]; b_wy = self.x[13]; b_wz = self.x[14]

        a_x = imu_measurement[0]; a_y = imu_measurement[1]; a_z = imu_measurement[2]
        w_x = imu_measurement[3]; w_y = imu_measurement[4]; w_z = imu_measurement[5]

        # create updated xdot
        self.xdot = np.zeros([15])

        # xpos_dot = xvel
        self.xdot[0:3] = last_xdot[3:6]

        # create transformation from intertial to body
        # rotation_mat_i2b = self.ypr_rotation_r2b(roll,pitch,yaw)

        # subtract bias best estimate from imu measurements
        force_vec_bias_corrected = np.array([a_x-b_ax,a_y-b_ay,a_z-b_az])
        angle_rate_bias_corrected = np.array([w_x-b_wx,w_y-b_wy,w_z-b_wz])

        force_vec_bias_uncorrected = np.array([a_x,a_y,a_z])
        angle_rate_bias_uncorrected = np.array([w_x,w_y,w_z])

        gravity_vec = np.array([0,0,G_ACCEL])

        rate_matrix = np.array([[0,-angle_rate_bias_corrected[2],angle_rate_bias_corrected[1]],
                                [angle_rate_bias_corrected[2],0,-angle_rate_bias_corrected[0]],
                                [-angle_rate_bias_corrected[1],angle_rate_bias_corrected[0],0]])
        
        # rotate rate matrix to ENU
        # rate_matrix = np.dot(self.ypr_rotation_b2r(np.pi,0,np.pi),rate_matrix)


        # rate_matrix = np.array([[0,-angle_rate_bias_uncorrected[2],angle_rate_bias_uncorrected[1]],
        #                         [angle_rate_bias_uncorrected[2],0,-angle_rate_bias_uncorrected[0]],
        #                         [-angle_rate_bias_uncorrected[1],angle_rate_bias_uncorrected[0],0]])

        # integrate rotation DCM -- first order approximation
        last_dcm = np.copy(self.dcm)
        # create rotation from platform NED to inertial NED using attitude estimate
        self.dcm = self.ypr_rotation_b2r(self.x[6],self.x[7],self.x[8])
        # propagate DCM of body to inertial NED with rate matrix
        # self.dcm = np.dot(self.dcm,np.eye(3)+rate_matrix*self.dt+0.5*np.dot(rate_matrix,rate_matrix)*self.dt**2+(1/6)*np.dot(rate_matrix,np.dot(rate_matrix,rate_matrix))*self.dt**3)
        # self.dcm = np.dot(self.dcm,np.eye(3)+rate_matrix*self.dt+0.5*np.dot(rate_matrix,rate_matrix)*self.dt**2)
        self.dcm = np.dot(self.dcm,np.eye(3)+rate_matrix*self.dt)

        # orthonormalize freshly computed DCM to prevent numerical error buildup
        dcm_u,dcm_sigma,dcm_vt = np.linalg.svd(self.dcm)
        # self.dcm = np.dot(self.dcm,sqrtm(np.linalg.inv(np.dot(self.dcm.transpose(),self.dcm))))
        self.dcm = np.dot(dcm_u,dcm_vt)
        self.dcm = self.dcm.astype(np.float)
        # self.dcm[0,:] = self.dcm[0,:]/np.linalg.norm(self.dcm[0,:])
        # self.dcm[1,:] = self.dcm[1,:]/np.linalg.norm(self.dcm[1,:])
        # self.dcm[2,:] = self.dcm[2,:]/np.linalg.norm(self.dcm[2,:])

        # rotate accelerations to intertial NED
        force_platform = np.dot(self.dcm,force_vec_bias_corrected)
        # force_platform = np.dot(0.5*(last_dcm + self.dcm),force_vec_bias_corrected)
        # NED to ENU
        # force_platform[1] = -1*force_platform[1]
        # force_platform[2] = -1*force_platform[2]
        # force_platform = self.NED2ENU(force_platform)

        # subtract gravity to get accelerations in intertial NED
        xvel_dot = force_platform #- gravity_vec
        # xvel_dot = np.dot(self.dcm,force_vec_bias_corrected) - gravity_vec
        # xvel_dot = np.dot(self.dcm,force_vec_bias_uncorrected) - gravity_vec
        self.x[3:6] = self.x[3:6] + xvel_dot*self.dt

        xpos_dot = self.x[3:6]
        self.x[0:3] = self.x[0:3] + xpos_dot*self.dt + 0.5*xvel_dot*self.dt**2

        roll = np.arctan2(self.dcm[2,1],self.dcm[2,2])
        # if roll < 0:
            # roll = 2*np.pi - roll
        pitch = np.arcsin(-self.dcm[2,0])
        yaw = np.arctan2(self.dcm[1,0],self.dcm[0,0])
        # if yaw < 0:
            # yaw = 2*np.pi - yaw

        self.x[6] = np.copy(roll)
        self.x[7] = np.copy(pitch)
        self.x[8] = np.copy(yaw)

        # self.x[9] = self.x[9]*np.exp(-self.dt/self.sensors['IMU'].accel_bias_tc)
        # self.x[10] = self.x[10]*np.exp(-self.dt/self.sensors['IMU'].accel_bias_tc)
        # self.x[11] = self.x[11]*np.exp(-self.dt/self.sensors['IMU'].accel_bias_tc)
        # self.x[10] += -(1/self.sensors['IMU'].accel_bias_tc)*self.x[10]
        # self.x[11] += -(1/self.sensors['IMU'].accel_bias_tc)*self.x[11]

        # self.x[12] = self.x[12]*np.exp(-self.dt/self.sensors['IMU'].gyro_bias_tc)
        # self.x[13] = self.x[13]*np.exp(-self.dt/self.sensors['IMU'].gyro_bias_tc)
        # self.x[14] = self.x[14]*np.exp(-self.dt/self.sensors['IMU'].gyro_bias_tc)
        # self.x[12] += -(1/self.sensors['IMU'].gyro_bias_tc)*self.x[12]
        # self.x[13] += -(1/self.sensors['IMU'].gyro_bias_tc)*self.x[13]
        # self.x[14] += -(1/self.sensors['IMU'].gyro_bias_tc)*self.x[14]

        # propagate covariance
        self.stm = np.eye(15)
        # position
        self.stm[0:3,3:6] = np.eye(3)*self.dt
        # velocity
        # rot_force = np.dot(self.dcm,force_vec_bias_corrected)
        rot_force = -1*np.dot(self.dcm,force_vec_bias_corrected)
        # rot_force = -1*np.dot(self.dcm,force_vec_bias_uncorrected)
        self.stm[3:6,6:9] = self.dt*np.array([[0,-rot_force[2],rot_force[1]],
                                    [rot_force[2],0,-rot_force[0]],
                                    [-rot_force[1],rot_force[0],0]])
        # self.stm[3:6,6:9] = self.dt*np.array([[0,-yaw,pitch],
        #                                     [yaw,0,-roll],
        #                                     [-pitch,roll,0]])
        self.stm[3:6,9:12] = self.dt*np.copy(self.dcm)
        # attitude
        self.stm[6:9,12:15] = self.dt*np.copy(self.dcm)

        # accel & gyro biases random walk
        # self.stm[9:12,9:12] = np.exp(-self.dt/self.sensors['IMU'].accel_bias_tc)*np.eye(3)
        # self.stm[12:15,12:15] = np.exp(-self.dt/self.sensors['IMU'].gyro_bias_tc)*np.eye(3)

        # noise
        Q = np.zeros((15,15))
        Q[3:6,3:6] = np.array(self.sensors['IMU'].accel_noise)*self.dt
        Q[6:9,6:9] = np.array(self.sensors['IMU'].gyro_noise)*self.dt
        Q[9:12,9:12] = self.sensors['IMU'].accel_bias*np.eye(3)*self.dt#*self.sensors['IMU'].accel_bias_tc
        Q[12:15,12:15] = self.sensors['IMU'].gyro_bias*np.eye(3)*self.dt#*self.sensors['IMU'].gyro_bias_tc

        self.P = np.dot(self.stm,np.dot(self.P,self.stm.transpose())) + Q

    def update(self,measurement,measurement_type):
        """
        Correct estimate using available aiding measurements.

        Parameters
        ----------
        measurements
            available measurements at current timestep
        """
        if measurement_type == 'GPS':
            H = np.zeros((3,15))
            H[0:3,0:3] = np.eye(3)
            R = self.sensors['GPS'].noise
            h = self.x[0:3]
            # self.gps_residuals = np.concatenate((self.gps_residuals,measurement-h))
        elif measurement_type == 'Depth':
            H = np.zeros((1,15))
            H[0,2] = 1
            R = self.sensors['Depth'].noise
            h = self.x[2]
            # self.depth_residuals = np.concatenate((self.depth_residuals,measurement-h))
        elif measurement_type == 'GPS_x':
            H = np.zeros((1,15))
            H[0,0] = 1
            R = self.sensors['GPS'].noise[0,0]
            h = self.x[0]
        elif measurement_type == 'GPS_y':
            H = np.zeros((1,15))
            H[0,1] = 1
            R = self.sensors['GPS'].noise[1,1]
            h = self.x[1]
        elif measurement_type == 'GPS_z':
            H = np.zeros((1,15))
            H[0,2] = 1
            R = self.sensors['GPS'].noise[2,2]
            h = self.x[2]
        elif measurement_type == 'Compass':
            H = np.zeros((1,15))
            H[0,8] = 1
            R = self.sensors['Compass'].noise
            h = self.x[8] 
            # self.compass_residuals = np.concatenate((self.compass_residuals,measurement-h))
        elif measurement_type == 'DVL':
            H = np.zeros((3,15))
            H[0:3,3:6] = np.eye(3)
            R = self.sensors['DVL'].noise
            h = self.x[3:6]
        elif measurement_type == 'Magnetometer':
            H = np.zeros((3,15))
            H[0:3,6:9] = np.eye(3)
            R = self.sensors['Magnetometer'].noise
            h = self.x[6:9]

        # compute the Kalman gain for the measurement
        K = np.dot(self.P,np.dot(H.transpose(),np.linalg.inv(np.dot(H,np.dot(self.P,H.transpose()))+R)))
        if measurement_type == 'GPS':
            self.gps_gains = np.concatenate((self.gps_gains,np.atleast_3d(K)),axis=2)
        
        try:
            assert(K.shape == (self.x.shape[0],H.shape[0]))
        except AssertionError:
            print('K is the wrong shape!: Is {}, should be {}'.format(K.shape,(self.x.shape[0],H.shape[0])))
            raise AssertionError

        x = self.x + np.dot(K,(measurement-h).transpose())
        self.x = np.squeeze(x)
        # self.P = np.dot(np.eye(15)-np.dot(K,H),self.P)
        # joseph form
        self.P = np.dot(np.eye(15)-np.dot(K,H),np.dot(self.P,(np.eye(15)-np.dot(K,H)).transpose())) + np.dot(K,np.dot(R,K.transpose()))
        # enforce symmetry
        self.P = 0.5*self.P + 0.5*self.P.transpose()

        # renormalize quaternion attitude
        # print(np.linalg.norm(self.x[6:10]))
        # print(self.x[6:10]/np.linalg.norm(self.x[6:10]))
        # self.x[6:10] = self.x[6:10]/np.linalg.norm(self.x[6:10])

        dcm = self.ypr_rotation_b2r(np.copy(self.x[6]),np.copy(self.x[7]),np.copy(self.x[8]))
        dcm = np.dot(dcm,sqrtm(np.linalg.inv(np.dot(dcm.transpose(),dcm))))
        dcm = dcm.astype(np.float)
        dcm[0,:] = dcm[0,:]/np.linalg.norm(dcm[0,:])
        dcm[1,:] = dcm[1,:]/np.linalg.norm(dcm[1,:])
        dcm[2,:] = dcm[2,:]/np.linalg.norm(dcm[2,:])
        roll = np.arctan2(dcm[2,1],dcm[2,2])
        pitch = np.arcsin(-dcm[2,0])
        yaw = np.arctan2(dcm[1,0],dcm[0,0])
        self.x[6] = np.copy(roll)
        self.x[7] = np.copy(pitch)
        self.x[8] = np.copy(yaw)

    def ENU2NED(self,enu_vec):
        """
        Convert 3-vector in ENU coordinates to NED.
        """
        ned_vec = np.copy(enu_vec)
        ned_vec[0] = enu_vec[1]
        ned_vec[1] = enu_vec[0]
        ned_vec[2] = -1*enu_vec[2]
        return ned_vec

    def NED2ENU(self,ned_vec):
        """
        Convert 3-vector in NED coordinates to ENU.
        """
        enu_vec = np.copy(ned_vec)
        enu_vec[0] = ned_vec[1]
        enu_vec[1] = ned_vec[0]
        enu_vec[2] = -1*ned_vec[2]
        return enu_vec

    def unwrap_angle(self,angle,target_angle,range=[-np.pi,np.pi]):
        """
        'Unwrap' and angle estimate or measurement by mapping it to the closest point
        to the angle range actually in use. 
        
        For example, for a yaw range of -180 to 180 deg,
        a measurement of -175 deg when the estimate is 175 deg is should be
        wrapped to 185 deg.
        """
        pass

    def wrap_angle(self,angle,range=[-np.pi,np.pi]):
        """
        'Wrap angle estimate or measurement from an unwrapped state back to respecting the
        range of values in use.

        For a yaw range of -180 to 180 deg, and estimate that was updated to be 185 deg
        should be mapped back to 0 -175 deg.
        """
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

    def ypr_rotation_b2r(self,roll,pitch,yaw,deg=False):
        """
        Create a yaw pitch roll rotation matrix from body to reference frame. Assumes radian inputs by default.
        Use deg flag to specify degrees.
        """
        if deg:
            roll *= np.pi/180
            pitch *= np.pi/180
            yaw *= np.pi/180

        roll_mat = np.array([[1,0,0],
                            [0,np.cos(roll),-np.sin(roll)],
                            [0,np.sin(roll),np.cos(roll)]])

        pitch_mat = np.array([[np.cos(pitch),0,np.sin(pitch)],
                            [0,1,0],
                            [-np.sin(pitch),0,np.cos(pitch)]])

        yaw_mat = np.array([[np.cos(yaw),-np.sin(yaw),0],
                            [np.sin(yaw),np.cos(yaw),0],
                            [0,0,1]])

        return np.dot(roll_mat,np.dot(pitch_mat,yaw_mat))

    def ypr_rotation_r2b(self,roll,pitch,yaw,deg=False):
        """
        Create a yaw pitch roll rotation matrix from reference frame to body. Assumes radian inputs by default.
        Use deg flag to specify degrees.
        """
        if deg:
            roll *= np.pi/180
            pitch *= np.pi/180
            yaw *= np.pi/180

        roll_mat = np.array([[1,0,0],
                            [0,np.cos(roll),np.sin(roll)],
                            [0,-np.sin(roll),np.cos(roll)]])

        pitch_mat = np.array([[np.cos(pitch),0,-np.sin(pitch)],
                            [0,1,0],
                            [np.sin(pitch),0,np.cos(pitch)]])

        yaw_mat = np.array([[np.cos(yaw),np.sin(yaw),0],
                            [-np.sin(yaw),np.cos(yaw),0],
                            [0,0,1]])

        return np.dot(roll_mat,np.dot(pitch_mat,yaw_mat))

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
            quaternion representation: follows [q0, q1, q2, q3] = [qw, qx, qy, qz]
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
            quaternion representation: follows [q0, q1, q2, q3] = [qw, qx, qy, qz]
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

    def gen_measurement_from_accel(self,accel):
        """
        Generate simulated IMU measurement from ground truth accelerations (not included in other gt vector).
        """
        accel_gt = accel[0:3]
        accel_bias = self.last_accel_bias + np.exp(-(1/self.rate)/self.accel_bias_tc)*np.random.multivariate_normal([0,0,0],np.eye(3)*self.accel_bias)
        accel_noise = np.random.multivariate_normal([0,0,0],self.accel_noise)
        accel_meas = accel_gt + accel_noise + accel_bias

        self.last_accel_bias = accel_bias
        self.last_accel_state = accel[0:3]

        gyro_gt = accel[3:]
        gyro_bias = self.last_gyro_bias + np.exp(-(1/self.rate)/self.gyro_bias_tc)*np.random.multivariate_normal([0,0,0],np.eye(3)*self.gyro_bias)
        gyro_noise = np.random.multivariate_normal([0,0,0],self.gyro_noise)
        gyro_meas = gyro_gt + gyro_noise + gyro_bias

        self.last_gyro_bias = gyro_bias
        self.last_gyro_state = accel[3:]

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
        true_pos = np.array([ground_truth[0],ground_truth[1],ground_truth[2]])
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
        true_heading = np.array([ground_truth[8]])
        noise = np.random.normal(0,self.noise)
        measurement = true_heading + noise
        measurement = np.mod(measurement,2*np.pi)
        if measurement > np.pi: measurement -= 2*np.pi
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

class DVL(Sensor):

    def __init__(self,cfg_path=None):
        super(DVL,self).__init__(cfg_path=cfg_path)

    def gen_measurement(self,ground_truth):
        """
        Generate simulated velocity measurements using ground truth data 
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
        true_pos = np.array([ground_truth[3],ground_truth[4],ground_truth[5]])
        noise = np.random.multivariate_normal([0,0,0],self.noise)
        measurement = true_pos + noise
        return measurement

class Magnetometer(Sensor):

    def __init__(self,cfg_path=None):
        super(Magnetometer,self).__init__(cfg_path=cfg_path)

    def gen_measurement(self,ground_truth):
        """
        Generate simulated roll pitch yaw measurements using ground truth data 
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
        true_att = np.array([ground_truth[6],ground_truth[7],ground_truth[8]])
        noise = np.random.multivariate_normal([0,0,0],self.noise)
        measurement = true_att + noise

        measurement[0] = np.mod(measurement[0],2*np.pi)
        if measurement[0] > np.pi: measurement[0] -= 2*np.pi

        measurement[1] = np.mod(measurement[1],2*np.pi)
        if measurement[1] > np.pi: measurement[1] -= 2*np.pi

        measurement[2] = np.mod(measurement[2],2*np.pi)
        if measurement[2] > np.pi: measurement[2] -= 2*np.pi

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