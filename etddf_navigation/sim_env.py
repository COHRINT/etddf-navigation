#!/usr/bin/env python

from __future__ import division

import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from strapdown_ins import StrapdownINS, GPS, IMU, Compass, Depth
from helpers.angle_conversions import euler2quat, quat2euler

# import pudb; pudb.set_trace()

def run_sim(sim_time,dt,sensors,filter_):
    """
    Run a simulated trajectory with the specified filter.
    """

    # for all timesteps dt in sim_time
    
    # propagate vehicle one timestep forward
    # vehicle_state_truth = 

    # simulate sensor measurements

    # update filter with available measurements

    #################################################

    # seed random number generator for predictable results
    np.random.seed(100)

    # sim time
    tstart = sim_time[0]
    tfin = sim_time[1]

    # define initial estimate and covariance, and constant input (speed, and turning vel)
    # x0 = [0 0 0]';
    x0 = np.array( [ [0], [0], [0] ] ,ndmin=2)
    P0 = np.diag([1,1,0.001])
    u = np.array( [ [2], [0.05] ], ndmin=2)

    Q = np.diag([1,1,0.01])
    noise = block_diag(Q,[0,0])
    w = [[1]] # gaussian noise intensity

    # solve the initial value problem for each timestep
    soln1 = solve_ivp(dubin_uni,[0,tfin],np.concatenate((x0,u)).transpose()[0],t_eval=np.linspace(0,tfin,num=(tfin/dt)+1))
    soln1wnoise = solve_ivp(dubin_uni_noise,[0,tfin],np.concatenate((x0,u,w)).transpose()[0],t_eval=np.linspace(0,tfin,num=(tfin/dt)+1))

    # generate measurements for each sensor for each timestep
    measurements = {}
    estimate = np.empty((1,16))
    covariance = np.empty((1,16))
    accel_bias_gt = np.empty((1,3))
    gyro_bias_gt = np.empty((1,3))
    accel_gt = np.empty((1,3))
    gyro_gt = np.empty((1,3))
    for i in range(0,soln1wnoise.y.shape[1]):
        gt = soln1wnoise.y[:,i].transpose()
        for s in sensors:
            if type(s).__name__ not in measurements and type(s).__name__ != 'IMU':
                try:
                    measurements[type(s).__name__] = np.empty((1,s.noise.shape[0]))
                except IndexError:
                    measurements[type(s).__name__] = np.empty((1,1))
            elif type(s).__name__ not in measurements and type(s).__name__ == 'IMU':
                if type(s).__name__+'_ACCEL' not in measurements:
                    measurements[type(s).__name__ + '_ACCEL'] = np.empty((1,s.noise.shape[0]))
                if type(s).__name__ + '_GYRO' not in measurements:
                    measurements[type(s).__name__ + '_GYRO'] = np.empty((1,s.noise.shape[0]))
            if np.mod(i,(1/(s.rate*dt))) == 0:
                if type(s).__name__ == 'IMU':
                    meas_a, meas_g, gt_a, gt_g, bias_a, bias_g = s.gen_measurement([gt[0],gt[3]*np.cos(gt[2]),gt[1],gt[3]*np.sin(gt[2]),10,0,0,0,0,0,gt[2],gt[4]])
                    meas_a = np.atleast_2d(meas_a)
                    meas_g = np.atleast_2d(meas_g)
                    measurements[type(s).__name__ + '_ACCEL'] = np.concatenate((measurements[type(s).__name__ + '_ACCEL'],meas_a),axis=0)
                    measurements[type(s).__name__ + '_GYRO'] = np.concatenate((measurements[type(s).__name__ + '_GYRO'],meas_g),axis=0)

                    bias_a = np.atleast_2d(bias_a)
                    bias_g = np.atleast_2d(bias_g)
                    accel_bias_gt = np.concatenate((accel_bias_gt,bias_a),axis=0)
                    gyro_bias_gt = np.concatenate((gyro_bias_gt,bias_g),axis=0)

                    gt_a = np.atleast_2d(gt_a)
                    gt_g = np.atleast_2d(gt_g)
                    accel_gt = np.concatenate((accel_gt,gt_a),axis=0)
                    gyro_gt = np.concatenate((gyro_gt,gt_g),axis=0)

                    # filter IMU measurements
                    imu_meas = np.squeeze(np.concatenate((meas_a,meas_g),axis=1),axis=0)
                    filter_.propagate(imu_meas)
                else:
                    meas = s.gen_measurement([gt[0],gt[3]*np.cos(gt[2]),gt[1],gt[3]*np.sin(gt[2]),10,0,0,0,0,0,gt[2],gt[4]])
                    meas = np.atleast_2d(meas)
                    measurements[type(s).__name__] = np.concatenate((measurements[type(s).__name__],meas),axis=0)

                    if type(s).__name__ == 'GPS':
                        filter_.update(meas,type(s).__name__)


        est,cov = filter_.get_estimate(cov=True)
        est = np.atleast_2d(est)
        diag_cov = np.diag(cov)
        diag_cov = np.atleast_2d(diag_cov)
        estimate = np.concatenate((estimate,est),axis=0)
        covariance = np.concatenate((covariance,diag_cov),axis=0)

    return soln1wnoise, measurements, estimate, covariance, accel_bias_gt, gyro_bias_gt, accel_gt, gyro_gt

def run_sim_from_file(sim_time,dt,sensors,sensor_data,ground_truth,filter_):
    """
    Run simulated trajectory from collected data with specified filter.
    """
    generated_measurements = {}
    estimate = np.empty((1,15))
    covariance = np.empty((1,15))
    accel_bias_gt = np.empty((1,3))
    gyro_bias_gt = np.empty((1,3))
    accel_gt = np.empty((1,3))
    gyro_gt = np.empty((1,3))
    for i in range(0,sensor_data['imu'].shape[0]):
        # propagate filter with imu measurements
        imu_meas = sensor_data['imu'][i,:]
        # ad-hoc NED to ENU (or is it ENU TO NED?) conversion
        # imu_meas[1] *= -1
        # imu_meas[2] *= -1
        # imu_meas[4] *= -1
        # imu_meas[5] *= -1
        filter_.propagate(imu_meas)

        # add other sensors at specfied rates
        for s in sensors:
            if type(s).__name__ not in generated_measurements and type(s).__name__ != 'IMU':
                try:
                    generated_measurements[type(s).__name__] = np.empty((1,s.noise.shape[0]))
                except IndexError:
                    generated_measurements[type(s).__name__] = np.empty((1,1))

            if np.mod(i,(1/(s.rate*dt))) == 0:
                [roll,pitch,yaw] = quat2euler([ground_truth[i,6],ground_truth[i,3],ground_truth[i,4],ground_truth[i,5]])
                meas = s.gen_measurement([ground_truth[i,0],
                                            ground_truth[i,1],
                                            ground_truth[i,2],
                                            ground_truth[i,7],
                                            ground_truth[i,8],
                                            ground_truth[i,9],
                                            roll,
                                            pitch,
                                            yaw,
                                            ground_truth[i,10],
                                            ground_truth[i,11],
                                            ground_truth[i,12]])
                meas = np.atleast_2d(meas)
                generated_measurements[type(s).__name__] = np.concatenate((generated_measurements[type(s).__name__],meas),axis=0)

                if type(s).__name__ == 'GPS':
                    # ENU to NED
                    # ned_meas = np.array([meas[0,1],meas[0,0],-1*meas[0,2]],ndmin=2)
                    # meas[0,2] = -1*meas[0,2]
                    filter_.update(meas,type(s).__name__)
                    # filter_.update(ned_meas[0,0],'GPS_x')
                    # filter_.update(ned_meas[0,1],'GPS_y')
                    # filter_.update(ned_meas[0,2],'GPS_z')
                if type(s).__name__ == 'Depth':
                    # meas *= -1
                    filter_.update(meas,type(s).__name__)
                if type(s).__name__ == 'Compass':
                    # meas -= np.pi/2
                    # meas *= -1
                    filter_.update(meas,type(s).__name__)
        
        # record current estimate
        est,cov = filter_.get_estimate(cov=True)
        est = np.atleast_2d(est)
        diag_cov = np.diag(cov)
        diag_cov = np.atleast_2d(diag_cov)
        estimate = np.concatenate((estimate,est),axis=0)
        covariance = np.concatenate((covariance,diag_cov),axis=0)

    return generated_measurements,estimate,covariance

def plotting_from_file(sim_time,dt,sensor_data,generated_measurements,gt_data,est,cov):
    """
    Plots measurements, ground truth, and filter estimates.
    """
    print(sensor_data['imu'].shape[0]*dt)
    # imu_time = np.arange(sim_time[0],sim_time[1]+dt,dt)
    imu_time = np.arange(0,(sensor_data['imu'].shape[0]+1)*dt,dt)
    # gps_time = np.arange(sim_time[0],sim_time[1],0.1)

    fig1 = plt.figure(1)
    plt.grid(True)
    ax = fig1.add_subplot(111, projection='3d')
    # ax = Axes3D(fig1)
    # plt.plot(gt_data[:,0], gt_data[:,1])
    ax.plot(gt_data[:,0],gt_data[:,1],gt_data[:,2])
    # plt.plot(est[:,0],est[:,1])
    ax.plot(est[:,0],est[:,1],est[:,2])
    # ax.plot(generated_measurements['GPS'][:,0],generated_measurements['GPS'][:,1],generated_measurements['GPS'][:,2],'x')
    plt.title('Ground Truth 3D Position')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    ax.set_zlabel('Z Position [m]')
    plt.legend(['ground truth','ins estimate','gps measurements'])
    ax.set_xlim([-100,100])
    ax.set_ylim([-100,100])
    ax.set_zlim([-100,100])

    fig22 = plt.figure(22)
    plt.subplot(311)
    plt.grid(True)
    plt.plot(imu_time,est[:,3]-gt_data[:,7])
    plt.plot(imu_time,2*np.sqrt(cov[:,3]),'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,3]),'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,3]),2*np.sqrt(cov[:,3]),alpha=0.1)
    # plt.xlim([-0.5,30.5])
    plt.ylim([-10,10])
    plt.ylabel('X est error [m/s]')
    plt.title('Velocity estimate error')

    plt.subplot(312)
    plt.grid(True)
    plt.plot(imu_time,est[:,4]-gt_data[:,8])
    plt.plot(imu_time,2*np.sqrt(cov[:,4]),'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,4]),'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,4]),2*np.sqrt(cov[:,4]),alpha=0.1)
    # plt.xlim([-0.5,30.5])
    plt.ylim([-10,10])
    plt.ylabel('Y est error [m/s]')

    plt.subplot(313)
    plt.grid(True)
    plt.plot(imu_time,est[:,5]-gt_data[:,9])
    plt.plot(imu_time,2*np.sqrt(cov[:,5]),'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,5]),'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,5]),2*np.sqrt(cov[:,5]),alpha=0.1)
    # plt.xlim([-0.5,30.5])
    plt.ylim([-10,10])
    plt.xlabel('Time [s]')
    plt.ylabel('Z est error [m]')

    fig21 = plt.figure(21)
    plt.subplot(311)
    plt.grid(True)
    plt.plot(imu_time,est[:,0]-gt_data[:,0])
    plt.plot(imu_time,2*np.sqrt(cov[:,0]),'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,0]),'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,0]),2*np.sqrt(cov[:,0]),alpha=0.1)
    # plt.xlim([-0.5,30.5])
    plt.ylim([-10,10])
    plt.ylabel('X est error [m]')
    plt.title('Position estimate error')

    plt.subplot(312)
    plt.grid(True)
    plt.plot(imu_time,est[:,1]-gt_data[:,1])
    plt.plot(imu_time,2*np.sqrt(cov[:,1]),'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,1]),'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,1]),2*np.sqrt(cov[:,1]),alpha=0.1)
    # plt.xlim([-0.5,30.5])
    plt.ylim([-10,10])
    plt.ylabel('Y est error [m]')

    plt.subplot(313)
    plt.grid(True)
    plt.plot(imu_time,est[:,2]-gt_data[:,2])
    plt.plot(imu_time,2*np.sqrt(cov[:,2]),'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,2]),'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,2]),2*np.sqrt(cov[:,2]),alpha=0.1)
    # plt.xlim([-0.5,30.5])
    plt.ylim([-10,10])
    plt.xlabel('Time [s]')
    plt.ylabel('Z est error [m]')

    plt.figure(2)
    plt.grid(True)
    plt.plot(generated_measurements['Compass'])
    plt.title('Compass-measured heading')
    plt.xlabel('Timestep')
    plt.ylabel('Heading [rad]')

    plt.figure(3)
    plt.grid(True)
    plt.plot(generated_measurements['GPS'][:,0])
    plt.plot(generated_measurements['GPS'][:,1])
    plt.plot(generated_measurements['GPS'][:,2])
    plt.title('GPS Position Measurements')
    plt.xlabel('Timestep')
    plt.ylabel('Measurement [m]')
    plt.legend(['x','y','z'])

    plt.figure(4)
    plt.grid(True)
    plt.plot(generated_measurements['Depth'])
    plt.title('Measured Depth')
    plt.xlabel('Timestep')
    plt.ylabel('Depth [m]')

    # convert estimated quaternion to euler angles
    # filter_roll = np.empty((est[:,6].shape))
    filter_roll = est[:,6] * 180/np.pi
    filter_pitch = est[:,7] * 180/np.pi
    filter_yaw = est[:,8] * 180/np.pi
    # filter_pitch = np.empty((est[:,6].shape))
    # filter_yaw = np.empty((est[:,6].shape))
    gt_roll = np.empty((gt_data[:,3].shape))
    gt_pitch = np.empty((gt_data[:,3].shape))
    gt_yaw = np.empty((gt_data[:,3].shape))
    # for i in range(0,est[:,6].shape[0]):
        # [filter_roll[i], filter_pitch[i], filter_yaw[i]] = quat2euler(est[i,6:10],deg=True)
    for i in range(0,gt_data[:,3].shape[0]):
        [gt_roll[i], gt_pitch[i], gt_yaw[i]] = quat2euler([gt_data[i,6],gt_data[i,3],gt_data[i,4],gt_data[i,5]],deg=True)

    plt.figure(8)
    plt.subplot(311)
    plt.grid(True)
    plt.plot(imu_time,filter_roll[:]-gt_roll[:])
    # plt.plot(gt_roll[:],'--')
    plt.plot(imu_time,2*np.sqrt(cov[:,6])*180/np.pi,'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,6])*180/np.pi,'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,6])*180/np.pi,2*np.sqrt(cov[:,6])*180/np.pi,alpha=0.1)
    plt.title('Filter Attitude Estimate Error')
    # plt.xlabel('Time [s]')
    plt.ylabel('Roll est error [deg]')
    # plt.legend(['est','gt'])
    plt.legend(['est',r'$\pm 2\sigma$'])
    plt.ylim([-185,185])

    plt.subplot(312)
    plt.grid(True)
    plt.plot(imu_time,filter_pitch[:]-gt_pitch[:])
    # plt.plot(gt_pitch[:],'--')
    plt.plot(imu_time,2*np.sqrt(cov[:,7])*180/np.pi,'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,7])*180/np.pi,'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,7])*180/np.pi,2*np.sqrt(cov[:,7])*180/np.pi,alpha=0.1)
    # plt.title('Filter Attitude Estimate Error: Pitch')
    # plt.xlabel('Time [s]')
    plt.ylabel('Pitch est error [deg]')
    # plt.legend(['est','gt'])
    plt.legend(['est',r'$\pm 2\sigma$'])
    plt.ylim([-185,185])

    plt.subplot(313)
    plt.grid(True)
    plt.plot(imu_time,filter_yaw[:]-gt_yaw[:])
    # plt.plot(gt_yaw[:],'--')
    plt.plot(imu_time,2*np.sqrt(cov[:,8])*180/np.pi,'r--')
    plt.plot(imu_time,-2*np.sqrt(cov[:,8])*180/np.pi,'r--')
    plt.fill_between(imu_time,-2*np.sqrt(cov[:,8])*180/np.pi,2*np.sqrt(cov[:,8])*180/np.pi,alpha=0.1)
    # plt.title('Filter Attitude Estimate Error: Yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw est error [deg]')
    # plt.legend(['est','gt'])
    plt.legend(['est',r'$\pm 2\sigma$'])
    plt.ylim([-185,185])

    plt.figure(10)
    plt.subplot(211)
    plt.grid(True)
    plt.plot(imu_time[:-1],sensor_data['imu'][:,0])
    plt.plot(imu_time[:-1],sensor_data['imu'][:,1])
    plt.plot(imu_time[:-1],sensor_data['imu'][:,2])
    plt.xlabel('Time [s]')
    plt.ylabel('Measurement [m/s/s]')
    plt.title('Accelerometer Measurements')
    plt.legend(['x','y','z'])

    plt.subplot(212)
    plt.grid(True)
    plt.plot(imu_time[:-1],sensor_data['imu'][:,3])
    plt.plot(imu_time[:-1],sensor_data['imu'][:,4])
    plt.plot(imu_time[:-1],sensor_data['imu'][:,5])
    plt.xlabel('Time [s]')
    plt.ylabel('Measurement [rad/s]')
    plt.title('Rate Gyro Measurements')
    plt.legend(['x','y','z'])

    plt.figure(11)
    plt.subplot(211)
    plt.grid(True)
    plt.plot(imu_time,est[:,9])
    plt.plot(imu_time,est[:,10])
    plt.plot(imu_time,est[:,11])
    plt.xlabel('Time [s]')
    plt.ylabel('Bias [m/s/s]')
    plt.title('Est accel bias')

    plt.subplot(212)
    plt.grid(True)
    plt.plot(imu_time,est[:,12])
    plt.plot(imu_time,est[:,13])
    plt.plot(imu_time,est[:,14])
    plt.xlabel('Time [s]')
    plt.ylabel('Bias [rad/s]')
    plt.title('Est gyro bias')

    
    plt.show()

def measurement_plotting(dt, sim_time, gt_results, measurements, filter_results, filter_cov, accel_bias_gt, gyro_bias_gt, accel_gt, gyro_gt):
    """
    Plots collected measurements from all sensors.
    """
    # compute integrated imu position and velocity
    imu_dt = 0.01
    pos = 0.5*measurements['IMU_ACCEL'][0,:]*imu_dt**2 #+ np.array((3,0,0))*imu_dt
    vel = measurements['IMU_ACCEL'][0,:]*imu_dt #+ np.array((3,0,0))

    imu_int_vel = np.array([vel])
    imu_int_pos = np.array([pos])

    for i in range(1,measurements['IMU_ACCEL'].shape[0]):
        imu_int_vel = np.concatenate((imu_int_vel,np.atleast_2d(imu_int_vel[i-1,:] + measurements['IMU_ACCEL'][i,:]*imu_dt)),axis=0)
        # imu_int_pos = np.concatenate((imu_int_pos,np.atleast_2d(imu_int_pos[i-1,:] + measurements['IMU'][i,:]*imu_dt + 0.5*measurements['IMU'][i,:]*(imu_dt**2))),axis=0)
        imu_int_pos = np.concatenate((imu_int_pos,np.atleast_2d(imu_int_pos[i-1,:] + imu_int_vel[i-1,:]*imu_dt)),axis=0)

    imu_time = np.arange(sim_time[0],sim_time[1]+imu_dt,imu_dt)
    gps_time = np.arange(sim_time[0],sim_time[1],0.1)

    # convert estimated quaternion to euler angles
    filter_roll = np.empty((filter_results[:,6].shape))
    filter_pitch = np.empty((filter_results[:,6].shape))
    filter_yaw = np.empty((filter_results[:,6].shape))
    for i in range(0,filter_results[:,6].shape[0]):
        [filter_roll[i], filter_pitch[i], filter_yaw[i]] = quat2euler(filter_results[i,6:10],deg=True)

    plt.figure(1)
    plt.grid(True)
    # plt.plot(soln1.y[0],soln1.y[1])
    # plt.plot(soln1wnoise[0:,0],soln1wnoise[0:,1])
    plt.plot(gt_results.y[0], gt_results.y[1])
    plt.plot(imu_int_pos[:,0],imu_int_pos[:,1])
    plt.plot(filter_results[1:,0],filter_results[1:,1])
    # plt.plot(x_est_x,x_est_y)
    plt.title('Ground Truth 2D Position')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    
    plt.figure(2)
    plt.grid(True)
    plt.plot(measurements['Compass'])
    plt.title('Compass-measured heading')
    plt.xlabel('Timestep')
    plt.ylabel('Heading [rad]')

    plt.figure(3)
    plt.grid(True)
    plt.plot(measurements['GPS'][:,0])
    plt.plot(measurements['GPS'][:,1])
    plt.plot(measurements['GPS'][:,2])
    plt.title('GPS Position Measurements')
    plt.xlabel('Timestep')
    plt.ylabel('Measurement [m]')
    plt.legend(['x','y','z'])

    plt.figure(4)
    plt.grid(True)
    plt.plot(measurements['IMU_ACCEL'][:,0])
    plt.plot(measurements['IMU_ACCEL'][:,1])
    plt.plot(measurements['IMU_ACCEL'][:,2])
    plt.plot(accel_gt[:,0])
    plt.plot(accel_gt[:,1])
    plt.plot(accel_gt[:,2])
    plt.title('IMU Acceleration Measurements')
    plt.xlabel('Timestep')
    plt.ylabel('Measurement [m/s/s]')
    plt.legend(['x','y','z'])

    plt.figure(5)
    plt.grid(True)
    plt.plot(measurements['IMU_GYRO'][:,0])
    plt.plot(measurements['IMU_GYRO'][:,1])
    plt.plot(measurements['IMU_GYRO'][:,2])
    plt.plot(gyro_gt[:,0])
    plt.plot(gyro_gt[:,1])
    plt.plot(gyro_gt[:,2])
    plt.title('IMU Angular Rate Measurements')
    plt.xlabel('Timestep')
    plt.ylabel('Measurement [rad/s]')
    plt.legend(['x','y','z'])

    plt.figure(6)
    plt.grid(True)
    plt.plot(imu_time,filter_results[0:-1,0]-gt_results.y[0])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,0]),-2*np.sqrt(filter_cov[0:-1,0]),alpha=0.25)
    plt.plot(imu_time,filter_results[0:-1,1]-gt_results.y[1])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,1]),-2*np.sqrt(filter_cov[0:-1,1]),alpha=0.25)
    plt.plot(imu_time,filter_results[0:-1,2]-10)
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,2]),-2*np.sqrt(filter_cov[0:-1,2]),alpha=0.25)
    plt.title('Filter Position Estimate Errors')
    plt.xlabel('Timestep')
    plt.ylabel('Estimate')
    plt.legend(['x','y','z'])

    plt.figure(7)
    plt.grid(True)
    plt.plot(imu_time,filter_results[0:-1,3]-(gt_results.y[3]*np.cos(gt_results.y[2])))
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,3]),-2*np.sqrt(filter_cov[0:-1,3]),alpha=0.25)
    plt.plot(imu_time,filter_results[0:-1,4]-(gt_results.y[3]*np.sin(gt_results.y[2])))
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,4]),-2*np.sqrt(filter_cov[0:-1,4]),alpha=0.25)
    plt.plot(imu_time,filter_results[0:-1,5]-0)
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,5]),-2*np.sqrt(filter_cov[0:-1,5]),alpha=0.25)
    plt.title('Filter Velocity Estimate Errors')
    plt.xlabel('Timestep')
    plt.ylabel('Estimate')
    plt.legend(['x','y','z'])

    plt.figure(8)
    plt.grid(True)
    plt.plot(imu_time,filter_roll[:-1])
    plt.plot(imu_time,filter_pitch[:-1])
    plt.plot(imu_time,filter_yaw[:-1])
    plt.plot(imu_time,180*gt_results.y[2]/np.pi)
    plt.title('Filter Attitude Estimate')
    plt.xlabel('Time [s]')
    plt.ylabel('Estimate [deg]')
    plt.legend(['roll','pitch','yaw'])

    plt.figure(9)
    plt.grid(True)
    plt.plot(imu_time,filter_results[:-1,10]-accel_bias_gt[:-1,0])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,10]),-2*np.sqrt(filter_cov[0:-1,10]),alpha=0.25)
    plt.plot(imu_time,filter_results[:-1,11]-accel_bias_gt[:-1,1])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,11]),-2*np.sqrt(filter_cov[0:-1,11]),alpha=0.25)
    plt.plot(imu_time,filter_results[:-1,12]-accel_bias_gt[:-1,2])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,12]),-2*np.sqrt(filter_cov[0:-1,12]),alpha=0.25)
    plt.title('Filter Acceleration Bias Estimate Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Bias Error [m/s/s]')
    plt.legend(['b_ax','b_ay','b_az'])
    
    plt.figure(10)
    plt.grid(True)
    plt.plot(imu_time,filter_results[:-1,13]-gyro_bias_gt[:-1,0])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,13]),-2*np.sqrt(filter_cov[0:-1,13]),alpha=0.25)
    plt.plot(imu_time,filter_results[:-1,14]-gyro_bias_gt[:-1,1])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,14]),-2*np.sqrt(filter_cov[0:-1,14]),alpha=0.25)
    plt.plot(imu_time,filter_results[:-1,15]-gyro_bias_gt[:-1,2])
    plt.fill_between(imu_time,2*np.sqrt(filter_cov[0:-1,15]),-2*np.sqrt(filter_cov[0:-1,15]),alpha=0.25)
    plt.title('Filter Gyro Rate Bias Estimate Error')
    plt.xlabel('Time [s]')
    plt.ylabel('Bias Error [rad/s/s]')
    plt.legend(['b_wx','b_wy','b_wz'])

    plt.show()

def dubin_uni(t,y):
    x = y[0]
    y_ = y[1]
    theta = y[2]
    v = y[3]
    omega = y[4]
    # dydt = np.array( ((v*np.cos(theta)),(v*np.sin(theta)),(omega),(0),(0)) )
    dydt = [v*np.cos(theta),v*np.sin(theta),omega,0,0]
    return dydt

def dubin_uni_noise(t,y):
    x = y[0]
    y_ = y[1]
    theta = y[2]
    v = y[3]
    omega = y[4]
    w = y[5]
    # dydt = np.array( ((v*np.cos(theta)),(v*np.sin(theta)),(omega),(0),(0)) )
    dydt = [v*np.cos(theta) + np.random.normal(0,np.sqrt(w)),v*np.sin(theta) + np.random.normal(0,np.sqrt(w)),omega + np.random.normal(0,0.5),np.random.normal(0,0.05),np.random.normal(0,0.05),0]
    return dydt

def soln_plotting(truth,est_results):
    pass

def main(from_file=False):

    # sim params
    sim_time = [0,200]
    dt = 0.02
    
    # create sensor instances
    imu = IMU()
    gps = GPS()
    compass = Compass()
    depth = Depth()

    sensors = [gps,compass,depth]

    # # create nav filter instance
    nav_filter = StrapdownINS(sensors={'IMU': imu,'GPS':gps,'Compass':compass,'Depth': depth}, dt=dt)

    # run simulation
    if from_file:
        # sensor_data = {'imu': np.load('manual_ctl_still_imu_data.npy')[1:,:]}
        # gt_data = np.load('manual_ctl_still_ground_truth_data.npy')
        sensor_data = {'imu': np.load('manual_ctl_straight_x_line_imu_data.npy')[3:,:]}
        gt_data = np.load('manual_ctl_straight_x_line_ground_truth_data.npy')
        # sensor_data = {'imu': np.load('manual_ctl_5mbox_imu_data.npy')[:,:]}
        # gt_data = np.load('manual_ctl_5mbox_ground_truth_data.npy')
        # print(gt_data.shape)
        # print(gt_data[:,0:10])
        generated_measurements,est,cov = run_sim_from_file(sim_time,dt,sensors,sensor_data,gt_data[:,:],nav_filter)
        plotting_from_file(sim_time,dt,sensor_data,generated_measurements,gt_data[:,:],est,cov)
    else:
        gt_results, measurements, filter_results, filter_cov, accel_bias_gt, gyro_bias_gt, accel_gt, gyro_gt = run_sim(sim_time,dt,sensors,filter_=nav_filter)
        measurement_plotting(dt,sim_time,gt_results,measurements,filter_results,filter_cov,accel_bias_gt,gyro_bias_gt,accel_gt,gyro_gt)

    # plot results
    # nav_filter_plotting(results)
    


if __name__ == "__main__":
    # collect filenames for data

    main(from_file=True)