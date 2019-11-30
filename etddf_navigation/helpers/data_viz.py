#!/usr/bin/env python

"""
Data visualization tools for OFFSET
"""

import os
import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
import argparse
# import pudb; pudb.set_trace()

from etddf_navigation.helpers.data_handling import load_sim_data, load_metadata

# def print_data_usage(path,agent_ids):
#     """
#     Print information about data transfer, including covariance intersection triggers and message sending rates.
#     """


def mse_plots(path,agent_ids):
    """
    Creates mean-squared-error plors for provided agent ids.

    Inputs:

        metadata -- sim run metadata
        data -- sim results data structure
        agent_ids -- list of agent ids to plot

    Outputs:

        plots -- matplotlib plot objects
    """
    # list of params for figures --> params not specified will have all values plotted
    # figs = [['delta10','drop00','tau5'],['delta10','drop00','tau7'],['delta20','drop00','tau5'],['delta20','drop00','tau7']]
    # figs = [['drop00','delta15','tau5'],['drop02','delta20','tau5']]
    # figs = [['delta10','drop00','tau5'],['delta10','drop00','tau7'],['delta20','drop00','tau5'],['delta20','drop00','tau7'],
    #         ['delta05','drop00','tau5'],['delta05','drop00','tau7'],['delta15','drop00','tau5'],['delta15','drop00','tau7'],
    #         ['delta05','drop00','tau35'],['delta10','drop00','tau35'],['delta15','drop00','tau35'],['delta20','drop00','tau35']]
    # figs = [['delta10','drop00','tau0'],['delta10','drop00','tau05'],['delta20','drop00','tau0'],['delta20','drop00','tau05'],
    #         ['delta05','drop00','tau0'],['delta05','drop00','tau05'],['delta15','drop00','tau0'],['delta15','drop00','tau05'],
    #         ['delta05','drop00','tau1'],['delta10','drop00','tau1'],['delta15','drop00','tau1'],['delta20','drop00','tau1']]
    # figs = [['delta05','drop00','tau15'],['delta10','drop00','tau15'],['delta15','drop00','tau15'],['delta20','drop00','tau15']]
    # figs = [['delta05','drop00','tau20'],['delta10','drop00','tau20'],['delta15','drop00','tau20'],['delta20','drop00','tau20'],
    #         ['delta05','drop00','tau25'],['delta10','drop00','tau25'],['delta15','drop00','tau25'],['delta20','drop00','tau25'],
    # figs = [['delta05','drop00','tau30'],['delta10','drop00','tau30'],['delta15','drop00','tau30'],['delta20','drop00','tau30']]
    # figs = [['delta15','drop00','tau30']]
    figs = [['delta15','drop00','tau70']]

    # load simulation metadata and get ids of agents to plot
    metadata = load_metadata(path)['cfg']
    if len(agent_ids) == 1 and agent_ids[0] == -1:
        agent_ids = list(range(0,len(metadata['agent_cfg']['conns'])))

    # for each fig to be created, get data
    for fig in figs:

        # get all sim data files with desired params
        all_files = os.listdir(path)
        files_to_load = []
        for file in all_files:
            keep_flag = True
            for param in fig:
                if param not in file:
                    keep_flag = False
            if keep_flag: files_to_load.append(file)
        
        data = []
        for file in files_to_load:
            data.append(load_sim_data(os.path.join(path,file)))
            
        # create time vector -- common to all plots
        time_vec = np.arange(start=0,
                            stop=metadata['max_time']+metadata['etddf_dt'],
                            step=metadata['etddf_dt'])

        nav_time_vec = np.arange(start=0,
                                    stop=metadata['max_time']+1*metadata['nav_dt'],
                                    step=metadata['nav_dt'])

        # create figure for figure parameter set
        plt.figure()
        legend_str = []

        # configure pyplot for using latex
        plt.rc('text', usetex=True)
        plt.rc('font',family='serif')
        plt.grid(True)
        plt.title(str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']))
        plt.xlabel('Time [s]')
        plt.ylabel(r'Est error [$m^2$]')

        # for each loaded data file
        for param_data in data:
            # for each agent generate mse plot
            for id_ in agent_ids:

                # extract agent data to plot
                mse_data = param_data['results']['etddf_mse'][:,id_]
                nav_mse_data = param_data['results']['nav_mse'][:,id_]

                plt.plot(time_vec[0:-1],mse_data)
                plt.plot(nav_time_vec[0:-1],nav_mse_data,'--')

                # plt.title(r'Agent {} ownship pos MSE: $\delta={}$, $\tau_g={}$, msg drop={}'.format(id_+1,param_data['metadata']['delta_value'],param_data['metadata']['tau_value'],param_data['metadata']['msg_drop_prob_value']))

            # legend_str.append(r'$\delta={}$'.format(param_data['metadata']['delta_value']))
                legend_str.append('{} etddf'.format(id_))
                legend_str.append('{} nav'.format(id_))

            print('-----')
            print(str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']))
            print('Total possible messages to send: {}'.format(param_data['results']['msgs_total']))
            print('Total messages sent: {}'.format(param_data['results']['msgs_sent']))
            print('CI triggers: {}'.format(param_data['results']['ci_total']))
            print('CI trigger rate: {}'.format(param_data['results']['ci_rate']))

        plt.legend(legend_str)
        plt.ylim([-1,100])

    # plt.show()

def time_trace_plots(path, agent_ids):
    """
    Creates time trace plots for provided agent ids

    Inputs:
    
        metadata -- sim run metadata
        data -- sim results data structure
        agent_ids -- list of agent ids to plot

    Outputs:

        plots -- matplotlib plot objects
    """
    # list of params for figures --> params not specified will have all values plotted
    # figs = [['delta10','drop00','tau5'],['delta10','drop00','tau7'],['delta20','drop00','tau5'],['delta20','drop00','tau7']]
    # figs = [['drop00','delta15','tau5'],['drop02','delta20','tau5']]
    # figs = [['delta10','drop00','tau5'],['delta10','drop00','tau7'],['delta20','drop00','tau5'],['delta20','drop00','tau7'],
    #         ['delta05','drop00','tau5'],['delta05','drop00','tau7'],['delta15','drop00','tau5'],['delta15','drop00','tau7'],
    #         ['delta05','drop00','tau35'],['delta10','drop00','tau35'],['delta15','drop00','tau35'],['delta20','drop00','tau35']]
    # figs = [['delta10','drop00','tau0'],['delta10','drop00','tau05'],['delta20','drop00','tau0'],['delta20','drop00','tau05'],
    #         ['delta05','drop00','tau0'],['delta05','drop00','tau05'],['delta15','drop00','tau0'],['delta15','drop00','tau05'],
    #         ['delta05','drop00','tau1'],['delta10','drop00','tau1'],['delta15','drop00','tau1'],['delta20','drop00','tau1']]
    # figs = [['delta05','drop00','tau15'],['delta10','drop00','tau15'],['delta15','drop00','tau15'],['delta20','drop00','tau15']]
    # figs = [['delta05','drop00','tau20'],['delta10','drop00','tau20'],['delta15','drop00','tau20'],['delta20','drop00','tau20'],
    #         ['delta05','drop00','tau25'],['delta10','drop00','tau25'],['delta15','drop00','tau25'],['delta20','drop00','tau25'],
    # figs = [['delta05','drop00','tau30'],['delta10','drop00','tau30'],['delta15','drop00','tau30'],['delta20','drop00','tau30']]
    # figs = [['delta15','drop00','tau30']]
    figs = [['delta15','drop00','tau70']]

     # load simulation metadata and get ids of agents to plot
    metadata = load_metadata(path)['cfg']
    if len(agent_ids) == 1 and agent_ids[0] == -1:
        agent_ids = list(range(0,len(metadata['agent_cfg']['conns'])))

    # for each fig to be created, get data
    for fig in figs:

        # get all sim data files with desired params
        all_files = os.listdir(path)
        files_to_load = []
        for file in all_files:
            keep_flag = True
            for param in fig:
                if param not in file:
                    keep_flag = False
            if keep_flag: files_to_load.append(file)
        
        data = []
        for file in files_to_load:
            data.append(load_sim_data(os.path.join(path,file)))
            
        # create time vector -- common to all plots
        time_vec = np.arange(start=0,
                            stop=metadata['max_time']+metadata['etddf_dt'],
                            step=metadata['etddf_dt'])

        nav_time_vec = np.arange(start=0,
                                    stop=metadata['max_time']+1*metadata['nav_dt'],
                                    step=metadata['nav_dt'])

    
        # create figure for figure parameter set
        # plt.figure()
        legend_str = []

        # configure pyplot for using latex
        plt.rc('text', usetex=True)
        plt.rc('font',family='serif')
        # plt.grid(True)
        # plt.title(str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']))
        # plt.xlabel('Time [s]')
        # plt.ylabel(r'Est error [$m$]')
        
        # for each loaded data file
        for param_data in data:
            # for each agent generate mse plot
            for id_ in agent_ids:

                # extract agent data to plot
                # mse_data = param_data['results']['etddf_mse'][:,id_]
                # nav_mse_data = param_data['results']['nav_mse'][:,id_]
                etddf_state_error = param_data['results']['etddf_state_error']
                etddf_cov_history = param_data['results']['etddf_cov_error']
                nav_state_error = param_data['results']['nav_state_error']
                nav_cov_history = param_data['results']['nav_cov_history']

                # print(etddf_state_error[id_].shape)
                # print(etddf_state_error[id_][7,:,:])
                # print(nav_cov_history[0].shape)
                # print(etddf_cov_history[0].shape)
                # # assert(False)

                # Position estimate error
                plt.figure()
                plt.subplot(311)
                plt.grid(True)
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,0],'C0')
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,0,0]),2*np.sqrt(etddf_cov_history[id_][:,0,0]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,0],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,0,0]),2*np.sqrt(nav_cov_history[id_][:,0,0]),alpha=0.1,color='C3')
                plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'Est error N [$m$]')
                plt.title('Agent ' + str(id_) + ', ' + str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Position error')

                plt.subplot(312)
                plt.grid(True)
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,2])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,2,2]),2*np.sqrt(etddf_cov_history[id_][:,2,2]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,1],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,1,1]),2*np.sqrt(nav_cov_history[id_][:,1,1]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'Est error E [$m$]')

                plt.subplot(313)
                plt.grid(True)
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,4])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,4,4]),2*np.sqrt(etddf_cov_history[id_][:,4,4]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,2],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,2,2]),2*np.sqrt(nav_cov_history[id_][:,2,2]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.xlabel('Time [s]')
                plt.ylabel(r'Est error D[$m$]')

                # Velocity estimate error
                plt.figure()
                plt.subplot(311)
                plt.grid(True)
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,1],'C0')
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,1,1]),2*np.sqrt(etddf_cov_history[id_][:,1,1]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,3],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,3,3]),2*np.sqrt(nav_cov_history[id_][:,3,3]),alpha=0.1,color='C3')
                plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'Est error N [$m/s$]')
                plt.title('Agent ' + str(id_) + ', ' + str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Velocity error')

                plt.subplot(312)
                plt.grid(True)
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,3])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,3,3]),2*np.sqrt(etddf_cov_history[id_][:,3,3]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,4],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,4,4]),2*np.sqrt(nav_cov_history[id_][:,4,4]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'Est error E [$m/s$]')

                plt.subplot(313)
                plt.grid(True)
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,5])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,5,5]),2*np.sqrt(etddf_cov_history[id_][:,5,5]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,5],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,5,5]),2*np.sqrt(nav_cov_history[id_][:,5,5]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.xlabel('Time [s]')
                plt.ylabel(r'Est error D[$m$]')

                # Attitude error
                plt.figure()
                plt.subplot(411)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,6],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,6,6]),2*np.sqrt(nav_cov_history[id_][:,6,6]),alpha=0.1,color='C3')
                plt.ylabel('q0 est error')
                plt.title('Agent ' + str(id_) + ', ' + str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Attitude error')
                plt.legend(['est error',r'$2\pm\sigma$'])

                plt.subplot(412)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,7],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,7,7]),2*np.sqrt(nav_cov_history[id_][:,7,7]),alpha=0.1,color='C3')
                plt.ylabel('q1 est error')

                plt.subplot(413)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,8],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,8,8]),2*np.sqrt(nav_cov_history[id_][:,8,8]),alpha=0.1,color='C3')
                plt.ylabel('q2 est error')

                plt.subplot(414)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,9],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,9,9]),2*np.sqrt(nav_cov_history[id_][:,9,9]),alpha=0.1,color='C3')
                plt.ylabel('q3 est error')

                # Euler angle attitude error
                euler_angles = np.zeros((nav_state_error[id_].shape[0],3))
                euler_cov = np.zeros((nav_state_error[id_].shape[0],3,3))
                for i in range(0,nav_state_error[id_].shape[0]):
                    [roll,pitch,yaw] = quat2euler(nav_state_error[id_][i,6:10])
                    euler_angles[i,0] = roll
                    euler_angles[i,1] = pitch
                    euler_angles[i,2] = yaw
                    euler_cov[i,:,:] = quat2euler_cov(nav_state_error[id_][i,6:10],nav_cov_history[id_][i,6:10,6:10])

                plt.figure()
                plt.subplot(311)
                plt.grid(True)
                plt.plot(nav_time_vec,euler_angles[:,0]*180/np.pi,'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(euler_cov[:,0,0])*180/np.pi,2*np.sqrt(euler_cov[:,0,0])*180/np.pi,alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Est error $\phi$ [$m/s/s$]')
                plt.title('Agent ' + str(id_) + ', ' + str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Euler attitude error')

                plt.subplot(312)
                plt.grid(True)
                plt.plot(nav_time_vec,euler_angles[:,1]*180/np.pi,'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(euler_cov[:,1,1])*180/np.pi,2*np.sqrt(euler_cov[:,1,1])*180/np.pi,alpha=0.1,color='C3')
                # plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Est error $\theta$ [$m/s/s$]')

                plt.subplot(313)
                plt.grid(True)
                plt.plot(nav_time_vec,euler_angles[:,2]*180/np.pi,'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(euler_cov[:,2,2])*180/np.pi,2*np.sqrt(euler_cov[:,2,2])*180/np.pi,alpha=0.1,color='C3')
                # plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.xlabel('Time [s]')
                plt.ylabel(r'Est error $\psi$ [$m/s/s$]')

                # Accelerometer Bias error
                plt.figure()
                plt.subplot(311)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,10],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,10,10]),2*np.sqrt(nav_cov_history[id_][:,10,10]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Est error X [$m/s/s$]')
                plt.title('Agent ' + str(id_) + ', ' + str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Accel bias error')

                plt.subplot(312)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,11],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,11,11]),2*np.sqrt(nav_cov_history[id_][:,11,11]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Est error Y [$m/s/s$]')

                plt.subplot(313)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,12],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,12,12]),2*np.sqrt(nav_cov_history[id_][:,12,12]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.xlabel('Time [s]')
                plt.ylabel(r'Est error Z [$m/s/s$]')

                # Gyro bias error
                plt.figure()
                plt.subplot(311)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,13],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,13,13]),2*np.sqrt(nav_cov_history[id_][:,13,13]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Est error X [$rad/s$]')
                plt.title('Agent ' + str(id_) + ', ' + str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Gyro bias error')

                plt.subplot(312)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,14],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,14,14]),2*np.sqrt(nav_cov_history[id_][:,14,14]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Est error Y [$rad/s$]')

                plt.subplot(313)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,15],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,15,15]),2*np.sqrt(nav_cov_history[id_][:,15,15]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.xlabel('Time [s]')
                plt.ylabel(r'Est error Z [$rad/s$]')



                # plt.title(r'Agent {} ownship pos MSE: $\delta={}$, $\tau_g={}$, msg drop={}'.format(id_+1,param_data['metadata']['delta_value'],param_data['metadata']['tau_value'],param_data['metadata']['msg_drop_prob_value']))

            # legend_str.append(r'$\delta={}$'.format(param_data['metadata']['delta_value']))
                legend_str.append('{} etddf'.format(id_))
                legend_str.append('{} nav'.format(id_))

            # print('-----')
            # print(str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']))
            # print('Total possible messages to send: {}'.format(param_data['results']['msgs_total']))
            # print('Total messages sent: {}'.format(param_data['results']['msgs_sent']))
            # print('CI triggers: {}'.format(param_data['results']['ci_total']))
            # print('CI trigger rate: {}'.format(param_data['results']['ci_rate']))

        # plt.legend(legend_str)

    # plt.show()

def quat2euler(quat,deg=False):
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

def quat2euler_cov(quat,quat_cov):
    """
    Converts quaternion covariance to euler angle covariance.
    See https://stats.stackexchange.com/questions/119780/what-does-the-covariance-of-a-quaternion-mean
    for more.
    """
    # unpack
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    # calculate jacobian
    # partial derivative of roll, phi wrt quaternion
    G00 = -(2*q1)/(((2*q0*q1 + 2*q2*q3)**2/(2*q1**2 + 2*q2**2 - 1)**2 + 1)*(2*q1**2 + 2*q2**2 - 1))
    G01 = -((2*q0)/(2*q1**2 + 2*q2**2 - 1) - (4*q1*(2*q0*q1 + 2*q2*q3))/(2*q1**2 + 2*q2**2 - 1)**2)/((2*q0*q1 + 2*q2*q3)**2/(2*q1**2 + 2*q2**2 - 1)**2 + 1)
    G02 = -((2*q3)/(2*q1**2 + 2*q2**2 - 1) - (4*q2*(2*q0*q1 + 2*q2*q3))/(2*q1**2 + 2*q2**2 - 1)**2)/((2*q0*q1 + 2*q2*q3)**2/(2*q1**2 + 2*q2**2 - 1)**2 + 1)
    G03 = -(2*q2)/(((2*q0*q1 + 2*q2*q3)**2/(2*q1**2 + 2*q2**2 - 1)**2 + 1)*(2*q1**2 + 2*q2**2 - 1))

    # parital derivative of pitch, theta, wrt quaternion
    G10 = (2*q2)/(1 - (2*q0*q2 - 2*q1*q3)**2)**(1/2)
    G11 = -(2*q3)/(1 - (2*q0*q2 - 2*q1*q3)**2)**(1/2)
    G12 = (2*q0)/(1 - (2*q0*q2 - 2*q1*q3)**2)**(1/2)
    G13 = -(2*q1)/(1 - (2*q0*q2 - 2*q1*q3)**2)**(1/2)

    # partial derivative of yaw, psi, wrt quaternion
    G20 = -(2*q3)/(((2*q0*q3 + 2*q1*q2)**2/(2*q2**2 + 2*q3**2 - 1)**2 + 1)*(2*q2**2 + 2*q3**2 - 1))
    G21 =-(2*q2)/(((2*q0*q3 + 2*q1*q2)**2/(2*q2**2 + 2*q3**2 - 1)**2 + 1)*(2*q2**2 + 2*q3**2 - 1))
    G22 = -((2*q1)/(2*q2**2 + 2*q3**2 - 1) - (4*q2*(2*q0*q3 + 2*q1*q2))/(2*q2**2 + 2*q3**2 - 1)**2)/((2*q0*q3 + 2*q1*q2)**2/(2*q2**2 + 2*q3**2 - 1)**2 + 1)
    G23 = -((2*q0)/(2*q2**2 + 2*q3**2 - 1) - (4*q3*(2*q0*q3 + 2*q1*q2))/(2*q2**2 + 2*q3**2 - 1)**2)/((2*q0*q3 + 2*q1*q2)**2/(2*q2**2 + 2*q3**2 - 1)**2 + 1)

    G = np.array([[G00, G01, G02, G03],
                [G10, G11, G12, G13],
                [G20, G21, G22, G23]])

    cov = np.dot(G,np.dot(quat_cov,G.T))

    return cov

def test_mse_plots():

    save_path = '../../data/sim_20190418-010908.pckl'
    data = load_sim_data(save_path)

    # pprint.pprint(data)
    time_trace_plots(data['results'][0]['metadata'],
                data['results'][0]['results'],
                [0,2,4])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot simulation results from simulation data structure, stored as pickle file.')
    parser.add_argument('agents',metavar='A',type=int,action='store',nargs='+',
                    help='ids of agents to plot (use -1 for all agents)')
    parser.add_argument('-t','--time-trace',dest='tt_flag',action='store_true',
                    help='plot time traces of estimate error')
    parser.add_argument('-m','--mse',dest='mse_flag',action='store_true',
                    help='plot mean-squared-errors (MSE)')
    parser.add_argument('-u','--data-usage',dest='data_usage',action='store_true',
                    help='Print information about data usage.')
    parser.add_argument('-f','--file-path',type=str,dest='file_path',action='store',
                    help='specify file path of sim data')
    parser.add_argument('-d','--dir-path',type=str,dest='dir_path',action='store',
                    help='specify path to sim data directory')
    args = parser.parse_args()

    # TODO: add arg for local, common, or both
    # TODO: figure out how to return plot objects and show after tt and mse plotting
    # TODO: default to plotting most recently save pickle file, instead of hardcoded path

    # set data path
    if args.file_path is None:
        save_path = '../../data/sim_20190418-010908.pckl'
    else:
        save_path = args.file_path

    # load data
    # data = load_sim_data(save_path)

    # get all agent ids if param is all agents (-1)
    agents = args.agents
    # if len(args.agents) == 1 and args.agents[0] == -1:
        # agents = list(range(0,data['results'][0]['metadata']['num_agents']))

    # generate plots
    if args.tt_flag:
        time_trace_plots(args.dir_path, agents)

    if args.mse_flag:
        mse_plots(args.dir_path, agents)

    plt.show()
    # if args.data_usage:
        # print_data_usage(args.dir_path, agents)