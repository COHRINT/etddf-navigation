#!/usr/bin/env python

"""
Simple simulator for testing navigation filters.
"""

from __future__ import division

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

from strapdown_ins import StrapdownINS, IMU, GPS, Compass, Depth, DVL, Magnetometer

import pudb; pudb.set_trace()

G_ACCEL = 9.80665 # gravitational acceleration [m/s/s]
# G_ACCEL = 9.799999981484898

np.random.seed(901282)

class SimpleNavSim:

    def __init__(self,dt=0.01,sim_time=[0,50],navfilter=None):
        
        self.dt = dt
        self.sim_time = sim_time

        self.filter = navfilter

    def create_sim_data(self,initial_state=np.zeros((15,))):
        """
        Create sim data by generating acceleration and angular rate data and integrating.
        """
        # sim time steps
        time_steps = np.arange(0,self.sim_time[1],self.dt)
        # assume linear accelerations only along body x axis in NED body frame
        x_accel = lambda t: 0.0*np.sin(t*3*np.pi/self.sim_time[1])
        y_accel = lambda t: 0.0*np.sin(t*3*np.pi/self.sim_time[1])
        z_accel = lambda t: 0.1*np.sin(t*3*np.pi/self.sim_time[1])
        # generate yaw rates
        roll_rate = lambda t: 0.0*np.sin(t*2*np.pi/self.sim_time[1])
        pitch_rate = lambda t: 0.1*np.sin(t*2*np.pi/self.sim_time[1])
        yaw_rate = lambda t: 0.0*np.sin(t*2*np.pi/self.sim_time[1])

        def dydt(t,y):
            roll,pitch,yaw = y[6:9]
            xvel,yvel,zvel = y[3:6]
            p,q,r = roll_rate(t),pitch_rate(t),yaw_rate(t)
            xacc,yacc,zacc = np.dot(self.ypr_rotation_b2r(roll,pitch,yaw),np.array([x_accel(t),y_accel(t),z_accel(t)]))
            return [xvel,yvel,zvel,xacc,yacc,zacc,p,q,r]

        # integrate accelerations and ang. rates to get vehicle trajectory
        true_state = np.zeros((time_steps.shape[0],15))
        y0 = [0,0,0,0,0,0,0,0,0]
        # for i,t in enumerate(time_steps):
        soln = solve_ivp(dydt,[0,self.sim_time[1]],y0,t_eval=time_steps)
        soln = np.transpose(soln.y)

        true_state[:,0:6] = soln[:,0:6]
        true_state[:,6] = x_accel(time_steps)
        true_state[:,7] = y_accel(time_steps)
        true_state[:,8] = z_accel(time_steps) #+ G_ACCEL 
        true_state[:,9:12] = soln[:,6:9]
        true_state[:,12] = roll_rate(time_steps)
        true_state[:,13] = pitch_rate(time_steps)
        true_state[:,14] = yaw_rate(time_steps)

        return time_steps, true_state

    def run_filter(self,time_steps,true_state):
        """
        Run navigation filter on ground truth data, generating measurements for filter consumption.
        """
        # create filter results variables
        filter_state = np.zeros((time_steps.shape[0],16))
        filter_cov = np.zeros((time_steps.shape[0],16,16))

        # create sensor measurement results dictionary
        # sensor_measurements = {s_name: np.zeros((time_steps.shape[0],s.noise.shape[0])) for s_name,s in self.filter.sensors.items()}
        sensor_measurements = {}
        sensor_measurements['IMU'] = np.zeros((time_steps.shape[0],12))
        sensor_measurements_cnt = {}
        for s_name,s in self.filter.sensors.items():
            try:
                if s_name != 'IMU':
                    sensor_measurements[s_name] = np.zeros((int(time_steps[-1]/(1/s.rate))+1,s.noise.shape[0]))
            except IndexError:
                sensor_measurements[s_name] = np.zeros((int(time_steps[-1]/(1/s.rate))+1,1))
            sensor_measurements_cnt[s_name] = 0

        # loop over sim time
        for i,t in enumerate(time_steps):
            # generate imu measurements
            accel_state = np.array([true_state[i,6],true_state[i,7],true_state[i,8],true_state[i,12],true_state[i,13],true_state[i,14]])
            accel_meas,gyro_meas,_,_,accel_bias,gyro_bias = self.filter.sensors['IMU'].gen_measurement_from_accel(accel_state)
            imu_meas = np.reshape(np.concatenate((accel_meas,gyro_meas)),(6,))
            # accel_bias = np.reshape(accel_bias,(3,))
            # gyro_bias = np.reshape(gyro_bias,(3,))
            sensor_measurements['IMU'][i,:] = np.reshape(np.concatenate((accel_meas,gyro_meas,accel_bias,gyro_bias)),(12,))

            # propagate filter
            self.filter.propagate(imu_meas)

            # generate any sensor measurements 
            for s_name,s in self.filter.sensors.items():
                if np.mod(i,(1/(s.rate*self.dt))) == 0 and s_name != 'IMU':
                    meas_state = np.array([true_state[i,0],
                                            true_state[i,1],
                                            true_state[i,2],
                                            true_state[i,3],
                                            true_state[i,4],
                                            true_state[i,5],
                                            true_state[i,9],
                                            true_state[i,10],
                                            true_state[i,11],
                                            true_state[i,0],
                                            true_state[i,0],
                                            0])
                    meas = s.gen_measurement(meas_state)
                    sensor_measurements[s_name][sensor_measurements_cnt[s_name],:] = meas
                    sensor_measurements_cnt[s_name] += 1

                    self.filter.update(meas,s_name)

            est,cov = self.filter.get_estimate(cov=True)
            filter_state[i,:] = np.reshape(est,filter_state[i,:].shape)
            filter_cov[i,:,:] = cov

        return filter_state, filter_cov, sensor_measurements

    def plot_results(self,time_steps,true_state,filter_state,filter_cov,sensor_measurements):
        
        plt.figure()
        plt.grid(True)
        plt.plot(time_steps,true_state[:,6])
        plt.plot(time_steps,true_state[:,7])
        plt.plot(time_steps,true_state[:,8])
        plt.plot(time_steps,true_state[:,12])
        plt.plot(time_steps,true_state[:,13])
        plt.plot(time_steps,true_state[:,14])
        plt.xlabel('Time [s]')
        plt.ylabel('Measured val [m/s/s or rad/s]')
        # plt.title('Body X accel and Yaw rate')
        plt.legend(['x acc','y acc','z acc','roll rate','pitch rate','yaw rate'])

        plt.figure()
        plt.grid(True)
        plt.plot(time_steps,true_state[:,0],'--')
        plt.plot(time_steps,true_state[:,1],'--')
        plt.plot(time_steps,true_state[:,2],'--')
        plt.plot(time_steps,filter_state[:,0])
        plt.plot(time_steps,filter_state[:,1])
        plt.plot(time_steps,filter_state[:,2])
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['GPS'].rate),sensor_measurements['GPS'][:,0])
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['GPS'].rate),sensor_measurements['GPS'][:,1])
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['GPS'].rate),sensor_measurements['GPS'][:,2])
        plt.title('position gt and estimate')
        plt.legend(['x gt','y gt','z gt','x est','y est','z est'])

        plt.figure()
        plt.grid(True)
        plt.plot(time_steps,true_state[:,3],'--')
        plt.plot(time_steps,true_state[:,4],'--')
        plt.plot(time_steps,true_state[:,5],'--')
        plt.plot(time_steps,filter_state[:,3])
        plt.plot(time_steps,filter_state[:,4])
        plt.plot(time_steps,filter_state[:,5])
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['DVL'].rate),sensor_measurements['DVL'][:,0])
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['DVL'].rate),sensor_measurements['DVL'][:,1])
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['DVL'].rate),sensor_measurements['DVL'][:,2])
        plt.title('velocity gt and estimate')
        plt.legend(['x gt','y gt','z gt','x est','y est','z est'])

        plt.figure()
        plt.plot(time_steps,true_state[:,9]*180/np.pi,'--')
        plt.plot(time_steps,true_state[:,10]*180/np.pi,'--')
        plt.plot(time_steps,true_state[:,11]*180/np.pi,'--')
        euler = np.zeros((time_steps.shape[0],3))
        for i,t in enumerate(time_steps):
            euler[i,:] = self.quat2euler(filter_state[i,6:10])
        plt.plot(time_steps,euler[:,0]*180/np.pi)
        plt.plot(time_steps,euler[:,1]*180/np.pi)
        plt.plot(time_steps,euler[:,2]*180/np.pi)
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['Magnetometer'].rate),sensor_measurements['Magnetometer'][:,0]*180/np.pi)
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['Magnetometer'].rate),sensor_measurements['Magnetometer'][:,1]*180/np.pi)
        # plt.plot(np.arange(self.sim_time[0],self.sim_time[1],1/self.filter.sensors['Magnetometer'].rate),sensor_measurements['Magnetometer'][:,2]*180/np.pi)
        plt.title('attitude gt and estimate')
        plt.legend(['roll gt','pitch gt','yaw gt','roll est','pitch est','yaw est'])

        plt.figure()
        quat = np.zeros((time_steps.shape[0],4))
        for i,t in enumerate(time_steps):
            quat[i,:] = self.euler2quat(true_state[i,9:12])
        plt.plot(time_steps,quat[:,0],'--')
        plt.plot(time_steps,quat[:,1],'--')
        plt.plot(time_steps,quat[:,2],'--')
        plt.plot(time_steps,quat[:,3],'--')    
        plt.plot(time_steps,filter_state[:,6])
        plt.plot(time_steps,filter_state[:,7])
        plt.plot(time_steps,filter_state[:,8])
        plt.plot(time_steps,filter_state[:,9])
        plt.title('Quaternion estimate and gt')
        plt.legend(['q0','q1','q2','q3','q0 est','q1 est','q2 est','q3 est'])

        plt.figure()
        plt.plot(time_steps,sensor_measurements['IMU'][:,6],'--')
        plt.plot(time_steps,sensor_measurements['IMU'][:,7],'--')
        plt.plot(time_steps,sensor_measurements['IMU'][:,8],'--')
        plt.plot(time_steps,filter_state[:,10])
        plt.plot(time_steps,filter_state[:,11])
        plt.plot(time_steps,filter_state[:,12])
        plt.legend(['b_ax gt','b_ay gt','b_az gt','b_ax est','b_ay est','b_az est'])

        plt.figure()
        plt.plot(time_steps,sensor_measurements['IMU'][:,9],'--')
        plt.plot(time_steps,sensor_measurements['IMU'][:,10],'--')
        plt.plot(time_steps,sensor_measurements['IMU'][:,11],'--')
        plt.plot(time_steps,filter_state[:,13])
        plt.plot(time_steps,filter_state[:,14])
        plt.plot(time_steps,filter_state[:,15])
        plt.legend(['b_wx gt','b_wy gt','b_wz gt','b_wx est','b_wy est','b_wz est'])

        fig1 = plt.figure()
        plt.grid(True)
        ax = fig1.add_subplot(111, projection='3d')
        # ax = Axes3D(fig1)
        # plt.plot(gt_data[:,0], gt_data[:,1])
        ax.plot(true_state[:,1],true_state[:,0],-true_state[:,2])
        ax.plot(filter_state[:,1],true_state[:,0],-filter_state[:,2])
        # plt.axis(['equal'])

        plt.figure()
        plt.subplot(311)
        plt.grid(True)
        plt.plot(time_steps,filter_state[:,0]-true_state[:,0])
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,0,0]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,0,0]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,0,0]),2*np.sqrt(filter_cov[:,0,0]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        plt.ylim([-10,10])
        plt.ylabel('X est error [m]')
        plt.title('Position estimate error')

        plt.subplot(312)
        plt.grid(True)
        plt.plot(time_steps,filter_state[:,1]-true_state[:,1])
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,1,1]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,1,1]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,1,1]),2*np.sqrt(filter_cov[:,1,1]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        plt.ylim([-10,10])
        plt.ylabel('Y est error [m]')

        plt.subplot(313)
        plt.grid(True)
        plt.plot(time_steps,filter_state[:,2]-true_state[:,2])
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,2,2]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,2,2]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,2,2]),2*np.sqrt(filter_cov[:,2,2]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        plt.ylim([-10,10])
        plt.xlabel('Time [s]')
        plt.ylabel('Z est error [m]')

        plt.figure()
        plt.subplot(311)
        plt.grid(True)
        plt.plot(time_steps,filter_state[:,3]-true_state[:,3])
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,3,3]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,3,3]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,3,3]),2*np.sqrt(filter_cov[:,3,3]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        plt.ylim([-10,10])
        plt.ylabel('X est error [m]')
        plt.title('Velocity estimate error')

        plt.subplot(312)
        plt.grid(True)
        plt.plot(time_steps,filter_state[:,4]-true_state[:,4])
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,4,4]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,4,4]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,4,4]),2*np.sqrt(filter_cov[:,4,4]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        plt.ylim([-10,10])
        plt.ylabel('Y est error [m]')

        plt.subplot(313)
        plt.grid(True)
        plt.plot(time_steps,filter_state[:,5]-true_state[:,5])
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,5,5]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,5,5]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,5,5]),2*np.sqrt(filter_cov[:,5,5]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        plt.ylim([-10,10])
        plt.xlabel('Time [s]')
        plt.ylabel('Z est error [m]')

        plt.figure()
        plt.subplot(411)
        plt.grid(True)
        plt.plot(time_steps,(filter_state[:,6]-quat[:,0]))
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,6,6]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,6,6]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,6,6]),2*np.sqrt(filter_cov[:,6,6]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        # plt.ylim([-10,10])
        plt.ylabel(r'$q_0$ est error [deg]')
        plt.title('Attitude estimate error')

        plt.subplot(412)
        plt.grid(True)
        plt.plot(time_steps,(filter_state[:,7]-quat[:,1]))
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,7,7]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,7,7]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,7,7]),2*np.sqrt(filter_cov[:,7,7]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        # plt.ylim([-10,10])
        plt.ylabel(r'$q_1$ est error [deg]')

        plt.subplot(413)
        plt.grid(True)
        plt.plot(time_steps,(filter_state[:,8]-quat[:,2]))
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,8,8]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,8,8]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,8,8]),2*np.sqrt(filter_cov[:,8,8]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        # plt.ylim([-10,10])
        plt.xlabel('Time [s]')
        plt.ylabel(r'$q_2$ est error [deg]')

        plt.subplot(414)
        plt.grid(True)
        plt.plot(time_steps,(filter_state[:,9]-quat[:,3]))
        plt.plot(time_steps,2*np.sqrt(filter_cov[:,9,9]),'r--')
        plt.plot(time_steps,-2*np.sqrt(filter_cov[:,9,9]),'r--')
        plt.fill_between(time_steps,-2*np.sqrt(filter_cov[:,9,9]),2*np.sqrt(filter_cov[:,9,9]),alpha=0.1)
        # plt.xlim([-0.5,30.5])
        # plt.ylim([-10,10])
        plt.xlabel('Time [s]')
        plt.ylabel(r'$q_3$ est error [deg]')

        plt.show()

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

def main():

    # create sensor instances
    imu = IMU()
    gps = GPS()
    compass = Compass()
    depth = Depth()
    dvl = DVL()
    mag = Magnetometer()
    sensors = [imu,gps,compass,depth]
    
    # create nav filter instance
    # nav_filter = StrapdownINS(sensors={'IMU':imu,'GPS':gps,'Depth':depth,'Compass':compass,'DVL':dvl,'Magnetometer':mag}, dt=0.01)
    # nav_filter = StrapdownINS(sensors={'IMU':imu,'GPS':gps,'Depth':depth,'Compass':compass}, dt=0.01)
    nav_filter = StrapdownINS(sensors={'IMU':imu, 'GPS':gps}, dt=0.01)
    # nav_filter = StrapdownINS(sensors={'IMU':imu,'GPS':gps,'Depth':depth,'Compass':compass}, dt=0.01)

    # create sim instance
    sim = SimpleNavSim(dt=0.01,navfilter=nav_filter,sim_time=[0,50])

    time_steps, true_state = sim.create_sim_data()
    filter_state, filter_cov, sensor_measurements = sim.run_filter(time_steps,true_state)
    sim.plot_results(time_steps,true_state,filter_state,filter_cov,sensor_measurements)

if __name__ == '__main__':
    main()