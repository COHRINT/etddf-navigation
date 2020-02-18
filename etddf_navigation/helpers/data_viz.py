#!/usr/bin/env python

"""
Data visualization tools for OFFSET
"""

import os
import sys
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import pudb; pudb.set_trace()

from etddf_navigation.helpers.data_handling import load_sim_data, load_metadata

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12

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
    # figs = [['delta10','drop00','tau35'],['delta10','drop00','tau50'],['delta10','drop00','tau70'],
    #         ['delta15','drop00','tau35'],['delta15','drop00','tau50'],['delta15','drop00','tau70'],
    #         ['delta20','drop00','tau35'],['delta20','drop00','tau50'],['delta20','drop00','tau70'],
    #         ['delta25','drop00','tau35'],['delta25','drop00','tau50'],['delta25','drop00','tau70']]
    figs = [['delta10','drop00','tau7'],['delta15','drop00','tau7']]
    # figs = [['delta25','drop00','tau7']]

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
        # create params title
        delta_str = fig[0].split('delta')[1]
        if int(delta_str) > 9:
            delta_str = str(int(delta_str)/10)
        tau_str = fig[2].split('tau')[1]
        if int(tau_str) > 9:
            tau_str = str(int(tau_str)/10)
        params_str = r'$\delta$=' + delta_str + r', $\tau$=' + tau_str

        # counter for color picking
        color_cnt = 0

        # configure pyplot for using latex
        plt.rc('text', usetex=True)
        plt.rc('font',family='serif')
        plt.grid(True)
        plt.title('Position RMSE, ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']),size=TITLE_SIZE)
        plt.xlabel('Time [s]',size=LABEL_SIZE)
        plt.ylabel(r'Est error [$m$]',size=LABEL_SIZE)

        for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
            label.set_fontsize(TICK_SIZE)

        # for each loaded data file
        for param_data in data:
            # for each agent generate mse plot
            baseline_mse_data = param_data['results']['nav_baseline_mse']
            baseline_mse_data_avg = np.mean(baseline_mse_data,axis=1)
            plt.plot(nav_time_vec[0:-1],np.sqrt(baseline_mse_data_avg),'--',color='C7')
            legend_str.append('baseline avg')

            for id_ in agent_ids:

                # extract agent data to plot
                mse_data = param_data['results']['etddf_mse'][:,id_]
                nav_mse_data = param_data['results']['nav_mse'][:,id_]

                etddf_mse_std_data = np.mean(param_data['results']['etddf_mse_std'][:,id_])
                nav_mse_std_data = np.mean(param_data['results']['nav_mse_std'][:,id_])
                print('Agent {} RMSE avg std: etdff - {}, nav - {}'.format(id_,np.sqrt(etddf_mse_std_data),np.sqrt(nav_mse_std_data)))

                color_str = 'C'+str(color_cnt%12)

                plt.plot(time_vec[0:-1],np.sqrt(mse_data),color=color_str)
                plt.plot(nav_time_vec[0:-1],np.sqrt(nav_mse_data),'--',color=color_str)

                # plt.title(r'Agent {} ownship pos MSE: $\delta={}$, $\tau_g={}$, msg drop={}'.format(id_+1,param_data['metadata']['delta_value'],param_data['metadata']['tau_value'],param_data['metadata']['msg_drop_prob_value']))

                color_cnt += 1

            # legend_str.append(r'$\delta={}$'.format(param_data['metadata']['delta_value']))
                legend_str.append('{} etddf'.format(id_))
                legend_str.append('{} nav'.format(id_))

            print('-----')
            print(str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']))
            print('Total possible messages to send: {}'.format(param_data['results']['msgs_total']))
            print('Total messages sent: {}'.format(param_data['results']['msgs_sent']))
            print('CI triggers: {}'.format(param_data['results']['ci_total']))
            print('CI trigger rate: {}'.format(param_data['results']['ci_rate']))

        plt.legend(legend_str,loc='upper left')
        plt.ylim([-1,40])

    # plt.show()

def rel_mse_plots(path,agent_ids):
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
    # figs = [['delta10','drop00','tau5'],['delta10','drop00','tau7'],['delta20','drop00','tau5'],['delta20','drop00','tau7']]#,
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
    # figs = [['delta15','drop00','tau70']]
    figs = [['delta10','drop00','tau35'],['delta10','drop00','tau50'],['delta10','drop00','tau70'],
            ['delta15','drop00','tau35'],['delta15','drop00','tau50'],['delta15','drop00','tau70'],
            ['delta20','drop00','tau35'],['delta20','drop00','tau50'],['delta20','drop00','tau70'],
            ['delta25','drop00','tau35'],['delta25','drop00','tau50'],['delta25','drop00','tau70']]

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
            data.append(load_sim_data(os.path.join(os.path.abspath(path),file)))
            
        # create time vector -- common to all plots
        time_vec = np.arange(start=0,
                            stop=metadata['max_time']+metadata['etddf_dt'],
                            step=metadata['etddf_dt'])

        # create figure for figure parameter set
        plt.figure()
        legend_str = []
        # create params title
        delta_str = fig[0].split('delta')[1]
        if int(delta_str) > 9:
            delta_str = str(int(delta_str)/10)
        tau_str = fig[2].split('tau')[1]
        if int(tau_str) > 9:
            tau_str = str(int(tau_str)/10)
        params_str = r'$\delta$=' + delta_str + r', $\tau$=' + tau_str

        # color_cnt = 0

        # configure pyplot for using latex
        plt.rc('text', usetex=True)
        plt.rc('font',family='serif')
        plt.grid(True)
        plt.title('Rel. Position RMSE, ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']),size=TITLE_SIZE)
        plt.xlabel('Time [s]',size=LABEL_SIZE)
        plt.ylabel(r'Est error [$m$]',size=LABEL_SIZE)

        for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
            label.set_fontsize(TICK_SIZE)

        # for each loaded data file
        for param_data in data:
            # for each agent generate mse plot
            color_cnt = 0
            for id_ in agent_ids:
                # extract agent data to plot
                rel_mse_data = param_data['results']['etddf_rel_mse'][id_]
                # baseline_mse_data = param_data['results']['baseline_mse'][:,id_]

                for conn in range(0,rel_mse_data.shape[1]):
                    color_str = 'C' + str(color_cnt%10)

                    plt.plot(time_vec[0:-1],np.sqrt(rel_mse_data[:,conn]),color=color_str)

                    legend_str.append(r'${}\rightarrow {}$'.format(id_,metadata['agent_cfg']['conns'][id_][conn]))
                    color_cnt += 1


                # plt.plot(time_vec,np.sqrt(baseline_mse_data),'--',color=color_str)

                # plt.title(r'Agent {} ownship pos MSE: $\delta={}$, $\tau_g={}$, msg drop={}'.format(id_+1,param_data['metadata']['delta_value'],param_data['metadata']['tau_value'],param_data['metadata']['msg_drop_prob_value']))

            # legend_str.append(r'$\delta={}$'.format(param_data['metadata']['delta_value']))
                # legend_str.append('{}'.format(id_))

            print('-----')
            print(str(fig) + ', Abs. pos. -- ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']))
            print('Total possible messages to send: {}'.format(param_data['results']['msgs_total']))
            print('Total messages sent: {}'.format(param_data['results']['msgs_sent']))
            print('CI triggers: {}'.format(param_data['results']['ci_total']))
            print('CI trigger rate: {}'.format(param_data['results']['ci_rate']))

        plt.legend(legend_str,loc='upper left')
        plt.ylim([-1,10])

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
    # figs = [['delta15','drop00','tau50']]
    # figs = [['delta10','drop00','tau50'],['delta10','drop00','tau35'],['delta20','drop00','tau50'],['delta20','drop00','tau35']]
    # figs = [['delta10','drop00','tau35'],['delta10','drop00','tau50'],['delta10','drop00','tau70'],
    #         ['delta15','drop00','tau35'],['delta15','drop00','tau50'],['delta15','drop00','tau70'],
    #         ['delta20','drop00','tau35'],['delta20','drop00','tau50'],['delta20','drop00','tau70'],
    #         ['delta25','drop00','tau35'],['delta25','drop00','tau50'],['delta25','drop00','tau70']]
    # figs = [['delta10','drop00','tau35'],['delta25','drop00','tau70']]
    # figs = [['delta10','drop00','tau50'],['delta20','drop00','tau50']]
    # figs = [['delta25','drop00','tau70']]
    figs = [['delta10','drop00','tau7'],['delta15','drop00','tau7']]

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
        # create params title
        delta_str = fig[0].split('delta')[1]
        if int(delta_str) > 9:
            delta_str = str(int(delta_str)/10)
        tau_str = fig[2].split('tau')[1]
        if int(tau_str) > 9:
            tau_str = str(int(tau_str)/10)
        params_str = r'$\delta$=' + delta_str + r', $\tau$=' + tau_str

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
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.ylim([-7,7])
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,0],'C0')
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,0,0]),2*np.sqrt(etddf_cov_history[id_][:,0,0]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,0],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,0,0]),2*np.sqrt(nav_cov_history[id_][:,0,0]),alpha=0.1,color='C3')
                plt.legend(loc='lower right',ncol=4,labels=['etddf','nav',r'etddf $\pm 2 \sigma$',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'$\eta$ [$m$]',size=LABEL_SIZE)
                plt.title('Agent ' + str(id_) + ', ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Position error',size=TITLE_SIZE)

                plt.subplot(312)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.ylim([-7,7])
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,2])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,2,2]),2*np.sqrt(etddf_cov_history[id_][:,2,2]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,1],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,1,1]),2*np.sqrt(nav_cov_history[id_][:,1,1]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'$\xi$ [$m$]',size=LABEL_SIZE)

                plt.subplot(313)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.ylim([-7,7])
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,4])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,4,4]),2*np.sqrt(etddf_cov_history[id_][:,4,4]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,2],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,2,2]),2*np.sqrt(nav_cov_history[id_][:,2,2]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.xlabel('Time [s]',size=LABEL_SIZE)
                plt.ylabel(r'$d$ [$m$]',size=LABEL_SIZE)

                # Velocity estimate error
                plt.figure()
                plt.subplot(311)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.ylim([-7,7])
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,1],'C0')
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,1,1]),2*np.sqrt(etddf_cov_history[id_][:,1,1]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,3],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,3,3]),2*np.sqrt(nav_cov_history[id_][:,3,3]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.legend(loc='lower right',ncol=4,labels=['etddf','nav',r'etddf $\pm 2 \sigma$',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'$\dot{\eta}$ [$m/s$]',size=LABEL_SIZE)
                plt.title('Agent ' + str(id_) + ', ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Velocity error',size=TITLE_SIZE)

                plt.subplot(312)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.ylim([-7,7])
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,3])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,3,3]),2*np.sqrt(etddf_cov_history[id_][:,3,3]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,4],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,4,4]),2*np.sqrt(nav_cov_history[id_][:,4,4]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.ylabel(r'$\dot{\xi}$ [$m/s$]',size=LABEL_SIZE)

                plt.subplot(313)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.ylim([-7,7])
                plt.plot(time_vec[:-1],etddf_state_error[id_][:,5])
                plt.fill_between(time_vec[:-1],-2*np.sqrt(etddf_cov_history[id_][:,5,5]),2*np.sqrt(etddf_cov_history[id_][:,5,5]),alpha=0.1,color='C0')
                plt.plot(nav_time_vec,nav_state_error[id_][:,5],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,5,5]),2*np.sqrt(nav_cov_history[id_][:,5,5]),alpha=0.1,color='C3')
                # plt.legend(['etddf',r'etddf $\pm 2 \sigma$','nav',r'nav $\pm 2 \sigma$'])
                plt.xlabel('Time [s]',size=LABEL_SIZE)
                plt.ylabel(r'$\dot{d}$ [$m/s$]',size=LABEL_SIZE)

                # Attitude error
                plt.figure()
                plt.subplot(411)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,6],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,6,6]),2*np.sqrt(nav_cov_history[id_][:,6,6]),alpha=0.1,color='C3')
                plt.ylabel(r'$q_0$',size=LABEL_SIZE)
                plt.title('Agent ' + str(id_) + ', ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Attitude error',size=TITLE_SIZE)
                plt.legend(['est error',r'$\pm 2\sigma$'],ncol=2,loc='lower right')

                plt.subplot(412)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,7],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,7,7]),2*np.sqrt(nav_cov_history[id_][:,7,7]),alpha=0.1,color='C3')
                plt.ylabel(r'$q_1$',size=LABEL_SIZE)

                plt.subplot(413)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,8],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,8,8]),2*np.sqrt(nav_cov_history[id_][:,8,8]),alpha=0.1,color='C3')
                plt.ylabel(r'$q_2$',size=LABEL_SIZE)

                plt.subplot(414)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,9],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,9,9]),2*np.sqrt(nav_cov_history[id_][:,9,9]),alpha=0.1,color='C3')
                plt.ylabel(r'$q_3$',size=LABEL_SIZE)
                plt.xlabel('Time [s]',size=LABEL_SIZE)

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
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,euler_angles[:,0]*180/np.pi,'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(euler_cov[:,0,0])*180/np.pi,2*np.sqrt(euler_cov[:,0,0])*180/np.pi,alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'$\phi$ [$deg$]',size=LABEL_SIZE)
                plt.title('Agent ' + str(id_) + ', ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Euler attitude error',size=TITLE_SIZE)

                plt.subplot(312)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,euler_angles[:,1]*180/np.pi,'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(euler_cov[:,1,1])*180/np.pi,2*np.sqrt(euler_cov[:,1,1])*180/np.pi,alpha=0.1,color='C3')
                # plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'$\theta$ [$deg$]',size=LABEL_SIZE)

                plt.subplot(313)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,euler_angles[:,2]*180/np.pi,'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(euler_cov[:,2,2])*180/np.pi,2*np.sqrt(euler_cov[:,2,2])*180/np.pi,alpha=0.1,color='C3')
                # plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.xlabel('Time [s]',size=LABEL_SIZE)
                plt.ylabel(r'$\psi$ [$deg$]',size=LABEL_SIZE)

                # Accelerometer Bias error
                plt.figure()
                plt.subplot(311)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,10],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,10,10]),2*np.sqrt(nav_cov_history[id_][:,10,10]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'X [$m/s/s$]',size=LABEL_SIZE)
                plt.title('Agent ' + str(id_) + ', ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Accel bias error',size=TITLE_SIZE)

                plt.subplot(312)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,11],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,11,11]),2*np.sqrt(nav_cov_history[id_][:,11,11]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Y [$m/s/s$]',size=LABEL_SIZE)

                plt.subplot(313)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,12],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,12,12]),2*np.sqrt(nav_cov_history[id_][:,12,12]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.xlabel('Time [s]',size=LABEL_SIZE)
                plt.ylabel(r'Z [$m/s/s$]',size=LABEL_SIZE)

                # Gyro bias error
                plt.figure()
                plt.subplot(311)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,13],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,13,13]),2*np.sqrt(nav_cov_history[id_][:,13,13]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'X [$rad/s$]',size=LABEL_SIZE)
                plt.title('Agent ' + str(id_) + ', ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']) + ', Gyro bias error',size=TITLE_SIZE)

                plt.subplot(312)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,14],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,14,14]),2*np.sqrt(nav_cov_history[id_][:,14,14]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.ylabel(r'Y [$rad/s$]',size=LABEL_SIZE)

                plt.subplot(313)
                for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                    label.set_fontsize(TICK_SIZE)
                plt.grid(True)
                plt.plot(nav_time_vec,nav_state_error[id_][:,15],'C3--')
                plt.fill_between(nav_time_vec,-2*np.sqrt(nav_cov_history[id_][:,15,15]),2*np.sqrt(nav_cov_history[id_][:,15,15]),alpha=0.1,color='C3')
                plt.legend(['est error',r'$\pm 2 \sigma$'])
                plt.xlabel('Time [s]',size=LABEL_SIZE)
                plt.ylabel(r'Z [$rad/s$]',size=LABEL_SIZE)



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

def trajectory_plots(path,agent_ids,params):
    """
    Creates plot of vehicle trajectories.

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
    # figs = [['delta10','drop00','tau5'],['delta10','drop00','tau7'],['delta20','drop00','tau5'],['delta20','drop00','tau7']]#,
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
    # figs = [['delta15','drop00','tau70']]
    # figs = [['delta10','drop00','tau7'],['delta25','drop00','tau7']]
    # figs = [['delta10','drop00','tau50'],['delta20','drop00','tau50']]
    # figs = [['delta25','drop00','tau7']]
    figs = [['delta10','drop00','tau7'],['delta15','drop00','tau7']]

    # load simulation metadata and get ids of agents to plot
    metadata = load_metadata(path)['cfg']
    if len(agent_ids) == 1 and agent_ids[0] == -1:
        agent_ids = list(range(0,len(metadata['agent_cfg']['conns'])))

    # plot handles
    f = [[] for x in range(0,len(figs))]

    plt.rc('text', usetex=True)
    plt.rc('font',family='serif')

    # for each fig to be created, get data
    for i,fig in enumerate(figs):

        # get all sim data files with desired params
        all_files = os.listdir(path)
        files_to_load = []
        for file in all_files:
            keep_flag = True
            for param in fig:
                if param not in file and params != -1:
                    keep_flag = False
            if keep_flag: files_to_load.append(file)
        
        data = []
        for file in files_to_load:
            data.append(load_sim_data(os.path.join(os.path.abspath(path),file)))

        # create time vector -- common to all plots
        time_vec = np.arange(start=0,
                            stop=metadata['max_time']+metadata['etddf_dt'],
                            step=metadata['etddf_dt'])

        # create figure for figure parameter set
        f[i] = plt.figure()
        plt.grid(True)
        ax = f[i].add_subplot(111, projection='3d')
        legend_str = []
        # create params title
        delta_str = fig[0].split('delta')[1]
        if int(delta_str) > 9:
            delta_str = str(int(delta_str)/10)
        tau_str = fig[2].split('tau')[1]
        if int(tau_str) > 9:
            tau_str = str(int(tau_str)/10)
        params_str = r'$\delta$=' + delta_str + r', $\tau$=' + tau_str

        # color_cnt = 0

        # configure pyplot for using latex
        # plt.rc('text', usetex=True)
        # plt.rc('font',family='serif')
        plt.title('Sample trajectories, ' + params_str + ', GPS agents: ' + str(metadata['agent_cfg']['sensors']['lin_abs_pos']['agents']),size=TITLE_SIZE)
        # plt.xlabel('Time [s]',size=LABEL_SIZE)
        # plt.ylabel(r'Est error [$m$]',size=LABEL_SIZE)
        plt.xlabel(r'$\eta$ [$m$]',size=LABEL_SIZE)
        plt.ylabel(r'$\xi$ [$m$]',size=LABEL_SIZE)
        ax.set_zlabel(r'$d$ [$m$]',size=LABEL_SIZE)

        for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
            label.set_fontsize(TICK_SIZE)

        # for each loaded data file
        # for param_data in data:
        #     # for each agent generate mse plot
        #     color_cnt = 0
        #     for id_ in agent_ids:
        #         # extract agent data to plot
        #         rel_mse_data = param_data['results']['etddf_rel_mse'][id_]

        print(len(data))

        # for sample trajectory, we just need one plot, and one trajectory from one monte carlo sim
        for id_ in agent_ids:
            true_pos = data[0]['results']['nav_true_states'][1][id_]
            print(true_pos.shape)
            
            
            # ax = Axes3D(fig1)
            # plt.plot(gt_data[:,0], gt_data[:,1])
            # ax.plot(gt_data[:,0],gt_data[:,1],gt_data[:,2])
            # plt.plot(est[:,0],est[:,1])
            ax.plot(true_pos[0,:-8],true_pos[1,:-8],true_pos[2,:-8])
            ax.scatter(true_pos[0,-9],true_pos[1,-9],true_pos[2,-9],marker='>',label='_nolegend_')
            # ax.plot(generated_measurements['GPS'][:,0],generated_measurements['GPS'][:,1],generated_measurements['GPS'][:,2],'x')
            # plt.title('Ground Truth 3D Position')
            # plt.xlabel('X Position [m]')
            # plt.ylabel('Y Position [m]')
            # ax.set_zlabel('Z Position [m]')
            # plt.legend(['ground truth','ins estimate','gps measurements'])
            # ax.set_xlim([-100,100])
            # ax.set_ylim([-100,100])
            # ax.set_zlim([-100,100])

            legend_str.append('{}'.format(id_))

        plt.legend(legend_str,loc='center left')


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
    parser.add_argument('-r','--rel-mse',dest='rel_mse_flag',action='store_true',
                    help='plot relative mean-squared-errors (relMSE)')
    parser.add_argument('-j','--traj',dest='traj_flag',action='store_true',
                    help='plot vehicle trajectory')
    parser.add_argument('-p','--plot-params',dest='params',action='store',nargs='*',type=float,
                    help='specifiy which simulations parameters you want to plot. Use -1 for all.')
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

    if args.rel_mse_flag:
        rel_mse_plots(args.dir_path, agents)

    if args.traj_flag:
        trajectory_plots(args.dir_path, agents, args.params)

    plt.show()
    # if args.data_usage:
        # print_data_usage(args.dir_path, agents)