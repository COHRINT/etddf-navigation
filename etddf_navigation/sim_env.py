#!/usr/bin/env python

from __future__ import division

import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp

from strapdown_ins import StrapdownINS, GPS, IMU, Compass

def run_sim(sim_time,dt,sensors,filter):
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
    u = np.array( [ [3], [0.2] ], ndmin=2)

    Q = np.diag([1,1,0.01])
    noise = block_diag(Q,[0,0])
    w = [[2]] # gaussian noise intensity

    # solve the initial value problem for each timestep
    soln1 = solve_ivp(dubin_uni,[0,tfin],np.concatenate((x0,u)).transpose()[0],t_eval=np.linspace(0,tfin,num=(tfin/dt)+1))
    soln1wnoise = solve_ivp(dubin_uni_noise,[0,tfin],np.concatenate((x0,u,w)).transpose()[0],t_eval=np.linspace(0,tfin,num=(tfin/dt)+1))

    # generate measurements for each sensor for each timestep
    measurements = {}
    for i in range(0,soln1wnoise.y.shape[1]):
        gt = soln1wnoise.y[:,i].transpose()
        for s in sensors:
            if type(s).__name__ not in measurements:
                try:
                    measurements[type(s).__name__] = np.empty((1,s.noise.shape[0]))
                except IndexError:
                    measurements[type(s).__name__] = np.empty((1,1))
            if np.mod(i,(1/(s.rate*dt))) == 0:
                meas = s.gen_measurement([gt[0],0,gt[1],0,0,0,0,0,0,0,gt[2],gt[4]])
                meas = np.atleast_2d(meas)
                measurements[type(s).__name__] = np.concatenate((measurements[type(s).__name__],meas),axis=0)

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.grid(True)
    plt.plot(soln1.y[0],soln1.y[1])
    # plt.plot(soln1wnoise[0:,0],soln1wnoise[0:,1])
    plt.plot(soln1wnoise.y[0], soln1wnoise.y[1])
    # plt.plot(x_est_x,x_est_y)
    
    plt.figure(2)
    plt.plot(measurements['Compass'])

    plt.figure(3)
    plt.plot(measurements['GPS'][:,0])
    plt.plot(measurements['GPS'][:,1])
    plt.plot(measurements['GPS'][:,2])

    plt.figure(4)
    plt.plot(measurements['IMU'][:,0])
    plt.plot(measurements['IMU'][:,1])
    plt.plot(measurements['IMU'][:,2])

    # pos = 0.5*measurements['IMU'][0,:]**2
    # imu_dt = 0.01

    # for i in range(1,measurements['IMU'].shape[0]):


    
    plt.show()

    return soln1wnoise

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
    dydt = [v*np.cos(theta) + np.random.normal(0,np.sqrt(w)),v*np.sin(theta) + np.random.normal(0,np.sqrt(w)),omega,0,0,0]
    return dydt

def soln_plotting(truth,est_results):
    pass

def main():

    # sim params
    sim_time = [0,100]
    dt = 0.01
    
    # create sensor instances
    imu = IMU()
    gps = GPS()
    compass = Compass()

    sensors = [imu,gps,compass]

    # # create nav filter instance
    # nf = NavFilter(sensors = [imu,gps,compass])

    # run simulation
    results = run_sim(sim_time,dt,sensors,filter=None)

    # plot results
    # nav_filter_plotting(results)


if __name__ == "__main__":
    main()