#!/usr/bin/env python

from __future__ import division

"""
Monte Carlo simulation framework for integrated navigation filtering and 
event-triggered decentralized data fusion.
"""

import os
import sys
import argparse
import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from scipy.integrate import solve_ivp

from strapdown_ins import StrapdownINS, IMU, GPS, Compass, Depth, DVL, Magnetometer

# from etddf.sim import SimInstance
from etddf.agent import Agent
from etddf.filters.kf import KF
from etddf.filters.etkf import ETKF
from etddf.dynamics import *
from etddf.helpers.config_handling import load_config
from etddf.helpers.msg_handling import MeasurementMsg, StateMsg
from etddf.helpers.data_handling import package_results, save_sim_data, make_data_directory, write_metadata_file
from etddf.helpers.data_viz import mse_plots, time_trace_plots
from etddf.quantization import Quantizer, covar_diagonalize
from etddf.covar_intersect import covar_intersect, gen_sim_transform

import pudb; pudb.set_trace()

G_ACCEL = 9.80665 # gravitational acceleration [m/s/s]

class NavSimInstance:
    """
    Run one simulation instance with the specified parameters. Combines ETDDF and navigation filtering simulations.

    Keyword arguments:
 
        delta           -- event-triggering threshold, allowed values=[0,inf)  
        tau             -- covariance intersection threshold per state, allowed values=[0,inf)  
        msg_drop_prob   -- probability that a message will be dropped in transit, values=[0,1]  
        max_time        -- simulation run time [sec], default=20 sec  
        dt              -- simulation time step [sec], default=0.1 sec  
        use_adaptive_tau -- boolean flag for using covarinace intersection adaptive thresholding  
        fixed_rng       -- value to seed random number generator  
        process_noise   -- structure of process noise values for every agent for testing  
        sensor_noise    -- structure of sensor noies values for every agent for testing  

    Returns:
    
        sim_data        -- simulation data structure containing results for current sim  
    """

    def __init__(self,delta,tau,msg_drop_prob,baseline_cfg,nav_baseline_cfg,
                    agent_cfg,nav_agent_cfg,max_time=20,nav_dt=0.01,etddf_dt=0.1,
                    use_adaptive_tau=True,fixed_rng=None,process_noise=False,
                    sensor_noise=False,quantization_flag=False,diagonalization_flag=False):

        self.max_time = max_time
        self.etddf_dt = etddf_dt
        self.nav_dt = nav_dt
        self.sim_time = 0
        self.sim_time_step = 0
        self.num_agents = len(agent_cfg['conns'])

        self.connections = agent_cfg['conns']

        self.etddf_dynamics = agent_cfg['dynamics_fxn']
        self.etddf_sensors = agent_cfg['sensors']
        self.nav_sensors = nav_agent_cfg['sensors']

        self.delta = delta
        self.tau_state_goal = tau
        self.tau = tau*0.75
        self.msg_drop_prob = msg_drop_prob
        self.use_adaptive_tau = use_adaptive_tau

        # navigation etddf fusion
        self.nav_ci = nav_agent_cfg['nav_etddf_ci']

        # vector for keeping track of etddf updates in nav time
        self.etddf_update_time = []

        # data compression flags and object
        self.quantization = quantization_flag
        self.diagonalization = diagonalization_flag
        # if self.quantization:
        #     self.quantizer = Quantizer(quantizer_fxn='x2')

        self.fixed_rng = fixed_rng
        if self.fixed_rng is not False:
            np.random.seed(self.fixed_rng)
            print('Fixing random number generator with seed: {}'.format(self.fixed_rng))


        # load matlab data for verification
        # self.sensor_noise_data = scipy.io.loadmat('../matlab/sensor_data_6agents.mat')
        # self.x_start_data = self.sensor_noise_data['x_start_data']
        # self.w_data = self.sensor_noise_data['w_data']
        # self.v_data = self.sensor_noise_data['v_data']
        # self.v_rel_data = self.sensor_noise_data['v_rel_data']

        self.all_msgs = []

        # generate ground truth starting positions
        x_true_vec_etddf, x_true_vec_nav = self.init_ground_truth()

        # create true states array for simulation
        # state order: position, velocity, acceleration, attitude, angular rates, accel bias, gyro bias
        self.true_states = np.zeros((self.num_agents,21,int(self.max_time/self.nav_dt)+10))
        for i in range(0,self.num_agents):
            self.true_states[i,:,0] = np.concatenate((np.take(x_true_vec_nav[i,:],[0,1,2,3,4,5]),np.zeros((3,)),self.quat2euler(np.take(x_true_vec_nav[i,:],[6,7,8,9])),np.zeros((3,)),np.take(x_true_vec_nav[i,:],[10,11,12,13,14,15])))

        # create baseline filter and agents from config for centralized baseline
        self.baseline_filter = self.init_etddf_baseline(x_true_vec_etddf,
                                                    baseline_cfg['dynamics_fxn'],
                                                    {},
                                                    baseline_cfg['sensors'])

        # create nav baseline filters, that will get gps aiding
        self.nav_baselines = self.init_nav_baseline(x_true_vec_nav,nav_baseline_cfg['sensors'])

        # initialize agents (includes navigation filters)
        self.agents = self.init_agents(x_true_vec_etddf,
                                        agent_cfg['dynamics_fxn'],
                                        {},
                                        agent_cfg['sensors'])

        # initialize nav filters for agents
        self.init_nav_agents(x_true_vec_nav)
        
        print('Initialized baseline and agents')

    def init_ground_truth(self):
        """
        Create ground truth initial position, velocity, and attitude for each agent's ETDDF filters:
        6 states -- [posx velx posy vely posz velz]
        and nav filters
        16 states -- [posx posy posz velx vely velz quat0 quat1 quat2 quat3 biasax biasay biasaz biasgx biasgy biasgz]
        """
        x_true_vec_etddf = np.zeros(6*self.num_agents)
        x_true_vec_nav = np.zeros((self.num_agents,16))
        for i in range(0,self.num_agents):
            # etddf initial states
            start_noise = np.random.normal([0,0,0,0,0,0],[5,0.5,5,0.2,5,0.2])
            # start_noise[1] = 0; start_noise[3] = 0; start_noise[5] = 0
            # start_noise = self.x_start_data[0,i]
            x_true_etddf = np.array(((i+1)*30,0,(i+1)*30,0,5,0)).T + start_noise
            x_true_vec_etddf[6*i:6*i+6] = x_true_etddf

            # nav filter initial states, uses etddf position and velocity
            x_true_posvel = np.array([x_true_etddf[0],x_true_etddf[2],x_true_etddf[4],x_true_etddf[1],x_true_etddf[3],x_true_etddf[5]])
            x_true_nav_att = np.array([1,0,0,0])
            x_true_nav_bias = np.array([0,0,0,0,0,0])
            x_true_vec_nav[i,:] = np.concatenate((x_true_posvel,x_true_nav_att,x_true_nav_bias))

        return x_true_vec_etddf, x_true_vec_nav

    def init_etddf_baseline(self,x_true_vec,dynamics_fxn='lin_ncv',dynamics_fxn_params={},sensors={}):
        """
        Create baseline centralized filter for comparison.

        Inputs:

            x_true_vec  -- initial true starting positions for every agent
            dynamics    -- name of dynamics fxn to be used
            sensors     -- dictionary of sensors and associated parameters

        Returns:

            baseline_filter -- instance of baseline centralized filter
        """
        # get sensor noise params
        R_abs = sensors['lin_abs_pos']['noise']
        R_rel = sensors['lin_rel_range']['noise']

        # get dynamics and process noise matrices
        F_full, G_full, Q_full = globals()[dynamics_fxn](self.etddf_dt,self.num_agents)

        # create initial estimate w/ ground truth
        x0_full = x_true_vec
        # create initial covariance
        P0_full = 100*np.eye(6*self.num_agents)

        # create filter instance
        baseline_filter = KF(F_full,G_full,0,0,Q_full,R_abs,R_rel,x0_full,P0_full,0)

        # add mse history
        baseline_filter.mse_history = [[] for x in range(0,self.num_agents)]

        return baseline_filter

    def init_nav_baseline(self,x_true_vec,nav_sensors):
        """
        Create baseline navigation filters for all agents, as if all agents were able to
        receive gps.
        """
        nav_filters = []

        for i in range(0,self.num_agents):
            # create sensor instances
            sensor_dict = {}
            for s in nav_sensors:
                if s == 'IMU':
                    imu = IMU()
                    sensor_dict['IMU'] = imu
                elif s == 'GPS':
                    gps = GPS()
                    sensor_dict['GPS'] = gps
                elif s == 'Compass':
                    compass = Compass()
                    sensor_dict['Compass'] = compass
            nf = StrapdownINS(sensors=sensor_dict,dt=self.nav_dt,init_est=x_true_vec[i,:])
            nav_filters.append(nf)

        return nav_filters

    def init_agents(self,x_true_vec,dynamics_fxn='lin_ncv',dynamics_fxn_params={},sensors={}):
        """
        Create agents, including associated local filters and common information filters.

        Inputs:

            x_true_vec  -- initial true starting positions for every agent
            dynamics    -- name of dynamics fxn to be used
            sensors     -- dictionary of sensors and associated parameters

        Returns:

            agents -- list of Agent instances
        """
        R_abs = sensors['lin_abs_pos']['noise']
        R_rel = sensors['lin_rel_range']['noise']

        # initialize list of agents
        agents = []

        for i in range(0,self.num_agents):

            agent_id = deepcopy(i)
            ids = sorted(deepcopy(self.connections[agent_id]))
            ids.append(agent_id)

            # build list of distance one and distance two neighbors for each agent
            # each agent gets full list of connections
            neighbor_conn_ids = []
            for j in range(0,len(self.connections[agent_id])):
                for k in range(0,len(self.connections[self.connections[agent_id][j]])):
                    if not self.connections[self.connections[agent_id][j]][k] in neighbor_conn_ids:
                        neighbor_conn_ids += self.connections[self.connections[agent_id][j]]

                    # remove agent's own id from list of neighbors
                    if agent_id in neighbor_conn_ids:
                        neighbor_conn_ids.remove(agent_id)

            # combine with direct connection ids and sort
            ids = list(set(sorted(ids + neighbor_conn_ids)))

            # divide out direct measurement connections and all connections
            connections_new = list(set(sorted(neighbor_conn_ids + self.connections[agent_id])))
            meas_connections = self.connections[agent_id]
            self.meas_connections = meas_connections
            self.agent_connections = connections_new

            est_state_length = len(ids)

            # find the connections, and therefore intersecting states of all connections, used for similarity transforms in CI updates
            self.neighbor_connections = {}
            self.neighbor_meas_connections = {}
            # loop through all connections of each of self's connections
            for ii in self.connections[agent_id]:
                if ii is not agent_id:
                    neighbor_agent_id = deepcopy(ii) # TODO this should be deepcopy(ii)
                    ids_new = sorted(deepcopy(self.connections[neighbor_agent_id]))
                    ids_new.append(neighbor_agent_id)

                    # build list of distance one and distance two neighbors for each agent
                    # each agent gets full list of connections
                    neighbor_conn_ids = []
                    for jj in range(0,len(self.connections[neighbor_agent_id])):
                        for kk in range(0,len(self.connections[self.connections[neighbor_agent_id][jj]])):
                            if not self.connections[self.connections[neighbor_agent_id][jj]][kk] in neighbor_conn_ids:
                                neighbor_conn_ids += deepcopy(self.connections[self.connections[neighbor_agent_id][jj]])

                            # remove agent's own id from list of neighbors
                            if neighbor_agent_id in neighbor_conn_ids:
                                neighbor_conn_ids.remove(neighbor_agent_id)

                    # combine with direct connection ids and sort
                    ids_new = list(set(sorted(ids_new + neighbor_conn_ids)))

                    # divide out direct measurement connections and all connections
                    neighbor_connections_new = list(set(sorted(neighbor_conn_ids + deepcopy(self.connections[neighbor_agent_id]))))
                    meas_connections_new = deepcopy(self.connections[neighbor_agent_id])
                    self.neighbor_meas_connections[ii] = deepcopy(meas_connections_new)
                    self.neighbor_connections[ii] = deepcopy(neighbor_connections_new)

            # construct local estimate
            # TODO: remove hardcoded 6
            n = (est_state_length)*6
            F,G,Q = globals()[dynamics_fxn](self.etddf_dt,est_state_length,**dynamics_fxn_params)

            # sort all connections
            # ids = sorted([agent_id] + connections_new)
            # create initial state estimate by grabbing relevant ground truth from full ground truth vector
            x0 = np.array([])
            for j in range(0,len(ids)):
                if j == agent_id:
                    x = x_true_vec[ids[j]*6:ids[j]*6+6]
                else:
                    x = np.array((0,0,0,0,0,0)).transpose()
                x0 = np.hstack( (x0,x) )

            P0 = 5000*np.eye(6*est_state_length)

            local_filter = ETKF(F,G,0,0,Q,np.array(R_abs),np.array(R_rel),
                                x0.reshape((F.shape[0],1)),P0,self.delta,
                                agent_id,connections_new,-1)

            # construct common information estimates
            common_estimates = []
            for j in range(0,len(meas_connections)):
                
                # find unique states between direct connections
                # inter_states = set(meas_connections).intersection(self.connections[self.connections[i][j]])
                unique_states = set(meas_connections+self.connections[self.connections[agent_id][j]])
                comm_ids = list(unique_states)
                x0_comm = np.array([])
                for k in range(0,len(comm_ids)):
                    if k == agent_id:
                        x = x_true_vec[comm_ids[k]*6:comm_ids[k]*6+6]
                    else:
                        x = np.array((0,0,0,0,0,0)).transpose()
                    x0_comm = np.hstack( (x0_comm,x) )
                
                # create comm info filter initial covariance
                P0_comm = 5000*np.eye(6*len(comm_ids))

                # generate dynamics
                F_comm, G_comm, Q_comm = globals()[dynamics_fxn](self.etddf_dt,len(comm_ids),**dynamics_fxn_params)

                # remove agent id from comm ids
                if agent_id in comm_ids:
                    comm_ids.remove(agent_id)

                # create common information filter
                comm_filter = ETKF(F_comm,G_comm,0,0,Q_comm,np.array(R_abs),np.array(R_rel),
                                    x0_comm.reshape((F_comm.shape[0],1)),P0_comm,self.delta,
                                    agent_id,comm_ids,meas_connections[j])

                common_estimates.append(comm_filter)

            # create agent instance
            new_agent = Agent(agent_id,connections_new,meas_connections,neighbor_conn_ids,
                                local_filter,common_estimates,x_true_vec[6*i:6*i+6],
                                0,len(x0)*self.tau_state_goal,len(x0)*self.tau,
                                self.use_adaptive_tau,self.quantization,self.diagonalization)

            agents.append(new_agent)

        return agents

    def init_nav_agents(self,x_true_vec_nav):
        """
        Post-hoc adds nav filters to all agents.
        """
        # parent constructor initialzies etddf portion of agents
        for a in self.agents:

            sensor_dict = {}
            for s,s_agents in self.nav_sensors.items():
                print(s,s_agents)
                if s == 'IMU' and (a.agent_id in s_agents or s_agents[0] == 'all'):
                    imu = IMU()
                    sensor_dict['IMU'] = imu
                elif s == 'GPS' and (a.agent_id in s_agents or s_agents[0] == 'all'):
                    gps = GPS()
                    sensor_dict['GPS'] = gps
                elif s == 'Compass' and (a.agent_id in s_agents or s_agents[0] == 'all'):
                    compass = Compass()
                    sensor_dict['Compass'] = compass

            # extract intitial estimate from local ETDDF filter, to also use in nav filter for position and velocity
            _,a_idx = a.get_location(a.agent_id)
            init_etddf_est = a.local_filter.x[a_idx]

            # construct nav filter inital estimate
            # init_nav_est = np.array([init_etddf_est[0,0],
            #                             init_etddf_est[2,0],
            #                             init_etddf_est[4,0],
            #                             init_etddf_est[1,0],
            #                             init_etddf_est[3,0],
            #                             init_etddf_est[5,0],
            #                             1,0,0,0,0,0,0,0,0,0])
            init_nav_est = x_true_vec_nav[a.agent_id,:]

            # create navigation filter instance
            nf = StrapdownINS(sensors=sensor_dict,dt=self.nav_dt,init_est=init_nav_est)
            a.nav_filter = nf

    def propagate_true_state(self,prop_state,control_fnxs,prop_time,eval_steps=None):
        """
        Propagate true state of vehicle with nonlinear odes and desired
        second order dynamics control.

        Parameters
        ----------
        state
            num_agents x 15 numpy array with true state of all vehicles in simulation
        control_fxns
            fxn handles for determining control on NED body x,y,z accelerations and angular rates
        prop_time
            propagation timeframe
        eval_steps : default None
            array of time points at which solution should be returned. used in scipy solve_ivp
        
        Returns
        -------
        prop_state
            num_agents x 15 numpy array with propagated true state of all vehicles
        """
        # extract control fxns --> control will be same for all vehicles, besides noise, if present
        x_accel = control_fnxs[0]
        y_accel = control_fnxs[1]
        z_accel = control_fnxs[2]
        roll_rate = control_fnxs[3]
        pitch_rate = control_fnxs[4]
        yaw_rate = control_fnxs[5]

        # rate of change function for use in solve_ivp
        def dydt(t,y):
            roll,pitch,yaw = y[6:9]
            xvel,yvel,zvel = y[3:6]
            p,q,r = roll_rate(t),pitch_rate(t),yaw_rate(t)
            xacc,yacc,zacc = np.dot(self.ypr_rotation_b2r(roll,pitch,yaw),np.array([x_accel(t),y_accel(t),z_accel(t)]))
            return [xvel,yvel,zvel,xacc,yacc,zacc,p,q,r]

        # prop_state = np.zeros((15,))

        # integrate IVP w/ initial conditions 
        y0 = np.array([prop_state[0],prop_state[1],prop_state[2],prop_state[3],prop_state[4],prop_state[5],prop_state[9],prop_state[10],prop_state[11]])
        soln = solve_ivp(dydt,prop_time,y0)
        soln = np.transpose(soln.y)

        prop_state[0:6] = soln[-1,0:6]
        prop_state[6] = x_accel(prop_time[1])
        prop_state[7] = y_accel(prop_time[1])
        prop_state[8] = z_accel(prop_time[1]) #+ G_ACCEL 
        prop_state[9:12] = soln[-1,6:9]
        prop_state[12] = roll_rate(prop_time[1])
        prop_state[13] = pitch_rate(prop_time[1])
        prop_state[14] = yaw_rate(prop_time[1])

        return prop_state

    def update(self,nav_sensors,etddf_sensors,etddf_dynamics='lin_ncv'):
        """
        Move forward one timestep in simulation. Assumes nav filters run more quickly than etddf,
        so updates happen at nav filter rate nav_dt.
        """
        # generate nav filter measurements and update filters & baseline filters
        # true_states = np.zeros((self.num_agents,15))

        # save gps measurements for consistency across nav and etddf filters
        gps_meas = np.zeros((self.num_agents,3))
        baseline_gps_meas = np.zeros((self.num_agents,3))

        for i in range(0,self.num_agents):
            # generate vehicle control
            x_accel = lambda t: 1.0*np.sin(t*np.pi/self.max_time) #+ np.random.normal(0,0.01)
            y_accel = lambda t: 0.0*np.sin(t*3*np.pi/self.max_time) #+ np.random.normal(0,0.01)
            z_accel = lambda t: 0.0*np.sin(t*3*np.pi/self.max_time) #+ np.random.normal(0,0.01)
            # generate yaw rates
            roll_rate = lambda t: 0.0*np.sin(t*2*np.pi/self.max_time) #+ np.random.normal(0,0.01)
            pitch_rate = lambda t: 0.0*np.sin(t*2*np.pi/self.max_time) #+ np.random.normal(0,0.01)
            yaw_rate = lambda t: 0.4#*np.sin(t*4*np.pi/self.max_time) #+ np.random.normal(0,0.01)

            control_fxns = [x_accel,y_accel,z_accel,roll_rate,pitch_rate,yaw_rate]

            # propagate true state
            updated_true_states = self.propagate_true_state(self.true_states[i,0:15,self.sim_time_step],control_fxns,[self.sim_time,self.sim_time+self.nav_dt])
            self.true_states[i,0:15,self.sim_time_step+1] = updated_true_states

            # rotate inertial gravity vector into body frame
            body_rotation = self.ypr_rotation_r2b(self.true_states[i,6,self.sim_time_step+1],
                                                self.true_states[i,7,self.sim_time_step+1],
                                                self.true_states[i,8,self.sim_time_step+1])
            body_grav = np.dot(body_rotation,np.array([0,0,G_ACCEL]))
            # add gravity to true state for generating measurements (we don't propagate with it because it is constant in our reference frame)
            true_state_wgrav = self.true_states[i,:,self.sim_time_step+1]
            true_state_wgrav[6:9] += body_grav
            # true_state_wgrav[8] += G_ACCEL
            
            # generate IMU measurement for nav filter and baseline filter
            # accel_state = np.array([self.true_states[i,6,self.sim_time_step+1],self.true_states[i,7,self.sim_time_step+1],self.true_states[i,8,self.sim_time_step+1],self.true_states[i,12,self.sim_time_step+1],self.true_states[i,13,self.sim_time_step+1],self.true_states[i,14,self.sim_time_step+1]])
            accel_state = np.array([true_state_wgrav[6],true_state_wgrav[7],true_state_wgrav[8],true_state_wgrav[12],true_state_wgrav[13],true_state_wgrav[14]])
            accel_meas,gyro_meas,_,_,accel_bias,gyro_bias = self.agents[i].nav_filter.sensors['IMU'].gen_measurement_from_accel(accel_state)
            imu_meas = np.reshape(np.concatenate((accel_meas,gyro_meas)),(6,))

            # update bias states
            self.true_states[i,15:18,self.sim_time_step+1] = accel_bias
            self.true_states[i,18:21,self.sim_time_step+1] = gyro_bias

            # accel_state = np.array([self.true_states[i,6,self.sim_time_step+1],self.true_states[i,7,self.sim_time_step+1],self.true_states[i,8,self.sim_time_step+1],self.true_states[i,12,self.sim_time_step+1],self.true_states[i,13,self.sim_time_step+1],self.true_states[i,14,self.sim_time_step+1]])
            accel_state = np.array([true_state_wgrav[6],true_state_wgrav[7],true_state_wgrav[8],true_state_wgrav[12],true_state_wgrav[13],true_state_wgrav[14]])
            accel_meas,gyro_meas,_,_,accel_bias,gyro_bias = self.nav_baselines[i].sensors['IMU'].gen_measurement_from_accel(accel_state)
            baseline_imu_meas = np.reshape(np.concatenate((accel_meas,gyro_meas)),(6,))
            
            # propagate coupled filter
            self.agents[i].nav_filter.propagate(imu_meas)
            # propagate baseline filter
            self.nav_baselines[i].propagate(baseline_imu_meas)

            # check for sensors
            for s_name,s in self.agents[i].nav_filter.sensors.items():
                if np.mod(self.sim_time_step,(1/(s.rate*self.nav_dt))) == 0 and s_name != 'IMU':
                    meas_state = np.array([self.true_states[i,0,self.sim_time_step+1],
                                            self.true_states[i,1,self.sim_time_step+1],
                                            self.true_states[i,2,self.sim_time_step+1],
                                            self.true_states[i,3,self.sim_time_step+1],
                                            self.true_states[i,4,self.sim_time_step+1],
                                            self.true_states[i,5,self.sim_time_step+1],
                                            self.true_states[i,9,self.sim_time_step+1],
                                            self.true_states[i,10,self.sim_time_step+1],
                                            self.true_states[i,11,self.sim_time_step+1],
                                            self.true_states[i,0,self.sim_time_step+1],
                                            self.true_states[i,0,self.sim_time_step+1],
                                            0])
                    meas = s.gen_measurement(meas_state)
                    # if self.sim_time_step == 900:
                    #     pudb.set_trace()
                    # sensor_measurements[s_name][sensor_measurements_cnt[s_name],:] = meas
                    # sensor_measurements_cnt[s_name] += 1

                    if s_name == 'GPS':
                        gps_meas[i,:] = meas

                    self.agents[i].nav_filter.update(meas,s_name)

            for s_name,s in self.nav_baselines[i].sensors.items():
                if np.mod(self.sim_time_step,(1/(s.rate*self.nav_dt))) == 0 and s_name != 'IMU':
                    meas_state = np.array([self.true_states[i,0,self.sim_time_step+1],
                                            self.true_states[i,1,self.sim_time_step+1],
                                            self.true_states[i,2,self.sim_time_step+1],
                                            self.true_states[i,3,self.sim_time_step+1],
                                            self.true_states[i,4,self.sim_time_step+1],
                                            self.true_states[i,5,self.sim_time_step+1],
                                            self.true_states[i,9,self.sim_time_step+1],
                                            self.true_states[i,10,self.sim_time_step+1],
                                            self.true_states[i,11,self.sim_time_step+1],
                                            self.true_states[i,0,self.sim_time_step+1],
                                            self.true_states[i,0,self.sim_time_step+1],
                                            0])
                    meas = s.gen_measurement(meas_state)
                    # sensor_measurements[s_name][sensor_measurements_cnt[s_name],:] = meas
                    # sensor_measurements_cnt[s_name] += 1

                    if s_name == 'GPS':
                        baseline_gps_meas[i,:] = meas

                    self.nav_baselines[i].update(meas,s_name)

            # self.agents[i].nav_filter.state_history.append(self.agents[i].nav_filter.x)
            # self.agents[i].nav_filter.cov_history.append(self.agents[i].nav_filter.P)

            self.agents[i].true_state.append(np.take(self.true_states[i,:,self.sim_time_step+1],[0,3,1,4,2,5]))

        # check if it's time to update etddf filters
        # if np.mod(self.sim_time,self.etddf_dt) == 0:
        if np.mod(self.sim_time_step,self.etddf_dt/self.nav_dt) == 0:
            # add nav time to update vector
            self.etddf_update_time.append(self.sim_time_step)

            # get dynamics abnd sensor noise
            F_local, G_local, _ = globals()[etddf_dynamics](self.etddf_dt,1)
            R_abs = etddf_sensors['lin_abs_pos']['noise']
            R_rel = etddf_sensors['lin_rel_range']['noise']

            # initialize msg inbox and ci inbox
            inbox = []
            ci_inbox = []
            for i in range(0,self.num_agents):
                inbox.append([])
                ci_inbox.append([])

            # generate control input
            # agent_control_input = np.array( ((2*np.cos(0.75*self.sim_time)),(2*np.sin(0.75*self.sim_time)),(0)) )
            all_control_input = np.zeros((self.num_agents*3))
            for i in range(0,self.num_agents):
                agent_control_input = np.dot(self.ypr_rotation_b2r(self.true_states[i,9,self.sim_time_step+1],self.true_states[i,10,self.sim_time_step+1],self.true_states[i,11,self.sim_time_step+1]),self.true_states[i,6:9,self.sim_time_step+1])
                all_control_input[i*3:i*3+3] = agent_control_input
            # all_control_input = np.tile(agent_control_input,self.num_agents)

            # print('agent crtl input: {}'.format(agent_control_input))

            # propagate baseline estimate
            self.baseline_filter.predict(all_control_input)

            # agent updates -- ground truth, local thresholding, and processing
            for j in range(0,self.num_agents):
                msgs = []

                # propagate agent true states
                # w = self.w_data[0,j][:,self.sim_time_step] # loaded from data file
                # print('w: {}'.format(w))
                w = np.random.multivariate_normal([0,0,0,0,0,0],Q_local_true).transpose()
                # self.agents[j].true_state.append(np.dot(F_local,self.agents[j].true_state[-1]) 
                #     + np.dot(G_local,agent_control_input) + w)
                # self.agents[j].true_state.append(np.take(self.true_states[j,:,self.sim_time_step+1],[0,3,1,4,2,5]))

                # generate measurements
                if self.agents[j].agent_id in etddf_sensors['lin_abs_pos']['agents']:
                    # simulate measurement noise
                    # v = self.v_data[0,j][:,self.sim_time_step] # loaded from data file
                    # print('v_abs: {}'.format(v))
                    v = np.random.multivariate_normal([0,0,0],R_abs)
                    # create measurement with agent true state and simmed noise
                    # use same gps meas as nav filter
                    # y_abs = np.dot(H_local,self.agents[j].true_state[-1]) + v
                    y_abs = gps_meas[j,:]
                    
                    # generate message structure -> note dest will change in thresholding
                    y_abs_msg = MeasurementMsg(self.agents[j].agent_id,
                                                self.agents[j].agent_id,
                                                self.agents[j].agent_id,
                                                [1,1,1],'abs',y_abs)
                    # add msg to queue
                    msgs.append(y_abs_msg)

                    # comm drop simulation
                    if np.random.binomial(1,1-self.msg_drop_prob):
                        self.baseline_filter.update(y_abs,'abs',self.agents[j].agent_id,
                                                    self.agents[j].agent_id)

                for k in range(0,len(self.agents[j].meas_connections)):
                    # simulate measurement noise
                    # v_rel = self.v_rel_data[0,j][:,self.sim_time_step]
                    # print('v_rel: {}'.format(v_rel))
                    v_rel = np.random.multivariate_normal([0,0,0],R_rel)
                    # create measurement with agent and target true states and simmed noise
                    y_rel = np.dot(H_rel,np.hstack( (self.agents[j].true_state[-1],
                                self.agents[self.agents[j].meas_connections[k]].true_state[-1]) )) + v_rel
                    # generate message structure -> note dest will change in thresholding
                    y_rel_msg = MeasurementMsg(self.agents[j].agent_id,
                                                self.agents[j].meas_connections[k],
                                                self.agents[j].meas_connections[k],
                                                [1,1,1],'rel',y_rel)
                    # add msg to queue
                    msgs.append(y_rel_msg)

                    # comms drop simulation
                    if np.random.binomial(1,1-self.msg_drop_prob):
                        self.baseline_filter.update(y_rel,'rel',self.agents[j].agent_id,
                                                    self.agents[j].meas_connections[k])

                # locally process measurements
                # agent_control_input = control_input[2*j:2*j+1,self.sim_time_step]
                # outgoing = self.agents[j].process_local_measurements(agent_control_input,msgs)
                outgoing = self.agents[j].process_local_measurements(all_control_input[j*3:j*3+3],msgs)

                # add outgoing measurements to agent inboxes
                for k in outgoing:
                    dest = k.dest
                    inbox[dest].append(k)
                    self.all_msgs.append(k)

            # agent update -- process received messages
            for j in range(0,self.num_agents):
                self.agents[j].process_received_measurements(inbox[j])

            # covariance intersection
            # this loop is a proxy for a service request and response framework.
            # The agent that triggers CI creates a request in the form of a StateMsg message
            # populated with its info, and sends it to its direct (distance-one) connections.
            # The connections popluate the remaining empty fields of the received message with 
            # their own information, and responds with a new message of the same type with their info.
            # Everyone adds these messages to their inbox to be processed.
            for j, agent in enumerate(self.agents):
                
            #     # check covariance trace for triggering CI
                agent.ci_trigger_rate = agent.ci_trigger_cnt / ((self.sim_time_step*(self.nav_dt/self.etddf_dt))-1)
                if np.trace(agent.local_filter.P) > agent.tau and agent.ci_trigger_rate < 0.2:
                # if self.sim_time_step*(self.nav_dt/self.etddf_dt) % 5 == 0:
                    agent.ci_trigger_cnt += 1
                    # agent.ci_trigger_rate = agent.ci_trigger_cnt / ((self.sim_time_step*(self.nav_dt/self.etddf_dt))-1)

                    for conn_id in agent.meas_connections:
                        
                        # generate message
                        msg_a = agent.gen_ci_message(conn_id,list(self.agents[conn_id].connections))
                        msg_b = self.agents[conn_id].gen_ci_message(agent.agent_id,list(agent.connections))

                        # # compress state messages
                        # if self.quantization and self.diagonalization:

                        #     # create element types list: assumes position, velocity alternating structure
                        #     element_types = []
                        #     for el_idx in range(0,msg_a.est_cov.shape[0]):
                        #         if el_idx % 2 == 0: element_types.append('position')
                        #         else: element_types.append('velocity')

                        #     # first diagonalize
                        #     cova_diag = covar_diagonalize(msg_a.est_cov)
                        #     covb_diag = covar_diagonalize(msg_b.est_cov)

                        #     # then quantize
                        #     bits_a = self.quantizer.state2quant(msg_a.state_est, cova_diag, element_types, diag_only=True)
                        #     bits_b = self.quantizer.state2quant(msg_b.state_est, covb_diag, element_types, diag_only=True)

                        #     # then decompress
                        #     meana_quant, cova_quant = self.quantizer.quant2state(bits_a[0], 2*cova_diag.shape[0], element_types, diag_only=True)
                        #     meanb_quant, covb_quant = self.quantizer.quant2state(bits_b[0], 2*covb_diag.shape[0], element_types, diag_only=True)

                        #     assert(cova_quant.shape == msg_a.est_cov.shape)

                        #     # add back to state messages
                        #     meana_quant = np.reshape(meana_quant,msg_a.state_est.shape)
                        #     msg_a.state_est = meana_quant
                        #     msg_a.est_cov = cova_quant
                        #     meanb_quant = np.reshape(meanb_quant,msg_b.state_est.shape)
                        #     msg_b.state_est = meanb_quant
                        #     msg_b.est_cov = covb_quant

                        # elif self.quantization:

                        #     # create element types list: assumes position, velocity alternating structure
                        #     element_types = []
                        #     for el_idx in range(0,msg_a.est_cov.shape[0]):
                        #         if el_idx % 2 == 0: element_types.append('position')
                        #         else: element_types.append('velocity')

                        #     # quantize
                        #     bits_a = self.quantizer.state2quant(msg_a.state_est, msg_a.est_cov, element_types)
                        #     bits_b = self.quantizer.state2quant(msg_b.state_est, msg_b.est_cov, element_types)

                        #     # then decompress
                        #     meana_quant, cova_quant = self.quantizer.quant2state(bits_a[0], int(msg_a.est_cov.shape[0] + (msg_a.est_cov.shape[0]**2 + msg_a.est_cov.shape[0])/2), element_types)
                        #     meanb_quant, covb_quant = self.quantizer.quant2state(bits_b[0], int(msg_b.est_cov.shape[0] + (msg_b.est_cov.shape[0]**2 + msg_b.est_cov.shape[0])/2), element_types)

                        #     # add back to state messages
                        #     meana_quant = np.reshape(meana_quant,msg_a.state_est.shape)
                        #     msg_a.state_est = meana_quant
                        #     msg_a.est_cov = cova_quant
                        #     meanb_quant = np.reshape(meanb_quant,msg_b.state_est.shape)
                        #     msg_b.state_est = meanb_quant
                        #     msg_b.est_cov = covb_quant

                        # elif self.diagonalization:
                        #     # diagonalize
                        #     cova_diag = covar_diagonalize(msg_a.est_cov)
                        #     covb_diag = covar_diagonalize(msg_b.est_cov)

                        #     msg_a.est_cov = cova_diag
                        #     msg_b.est_cov = covb_diag

                        # add messages to ci inbox
                        ci_inbox[conn_id].append(deepcopy(msg_a))
                        ci_inbox[agent.agent_id].append(deepcopy(msg_b))

            # # process inbox messages
            for j, msg_list in enumerate(ci_inbox):
                self.agents[j].process_ci_messages(msg_list)

            # perform CI between nav and etddf instances
            if self.nav_ci:
                for j, agent in enumerate(self.agents):
                    # grab state estimate from local filter
                    xa = deepcopy(agent.local_filter.x)
                    Pa = deepcopy(agent.local_filter.P)

                    # nav filter is "connected" only to local filter
                    # nav filter state ordering is [x y z xvel yvel zvel]
                    b_connections = [agent.agent_id]
                    b_id = agent.agent_id
                    xbT = np.take(deepcopy(self.agents[j].nav_filter.x),[0,3,1,4,2,5,6,7,8,9,10,11,12,13,14,15])
                    PbT = deepcopy(self.agents[j].nav_filter.P)[np.ix_([0,3,1,4,2,5,6,7,8,9,10,11,12,13,14,15],[0,3,1,4,2,5,6,7,8,9,10,11,12,13,14,15])]
                    xbTred = np.reshape(xbT[0:6],(6,1))
                    PbTred = PbT[0:6,0:6]

                    # construct transform for etddf instance
                    Ta, il_a, inter = gen_sim_transform(agent.agent_id,list(agent.connections),
                                                        b_id,list(b_connections),num_states=6)

                    # compute reduced, transformed state estimate
                    xaT = np.dot(inv(Ta),xa)
                    xaTred = xaT[0:il_a]
                    PaT = np.dot(inv(Ta),np.dot(Pa,Ta))
                    PaTred_grid = np.ix_(np.arange(0,il_a),np.arange(0,il_a))
                    PaTred = PaT[PaTred_grid]

                    # if self.sim_time_step == 1950:
                        # pudb.set_trace()

                    # perform covariance intersection with reduced estimates
                    alpha = np.ones((PaTred.shape[0],1))
                    xc, Pc = covar_intersect(xaTred,xbTred,PaTred,PbTred,alpha)

                    # compute information delta for conditional update
                    invD_a = inv(Pc) - inv(PaTred)
                    invDd_a = np.dot(inv(Pc),xc) - np.dot(inv(PaTred),xaTred)

                    invD_b = inv(Pc) - inv(PbTred)
                    invDd_b = np.dot(inv(Pc),xc) - np.dot(inv(PbTred),xbTred)

                    # conditional gaussian update
                    if (PaT.shape[0]-Pc.shape[0] == 0) or (PaT.shape[1]-Pc.shape[1] == 0):
                        cond_cov_a = invD_a
                        cond_mean_a = invDd_a
                    else:
                        cond_cov_a_row1 = np.hstack( (invD_a,np.zeros((Pc.shape[0],PaT.shape[1]-Pc.shape[1]))) )
                        cond_cov_a_row2 = np.hstack( (np.zeros((PaT.shape[0]-Pc.shape[0],Pc.shape[1])),np.zeros((PaT.shape[0]-Pc.shape[0],PaT.shape[0]-Pc.shape[0]))) )
                        cond_cov_a = np.vstack( (cond_cov_a_row1,cond_cov_a_row2) )
                        cond_mean_a = np.vstack( (invDd_a,np.zeros((PaT.shape[0]-Pc.shape[0],1))) )

                    Va = inv(inv(PaT) + cond_cov_a)
                    va = np.dot(Va,np.dot(inv(PaT),xaT) + cond_mean_a)

                    # transform back to normal state order
                    xa = np.dot(Ta,va)
                    Pa = np.dot(Ta,np.dot(Va,inv(Ta)))

                    # update local estimates
                    agent.local_filter.x = deepcopy(xa)
                    agent.local_filter.P = deepcopy(Pa)

                    # do it all again for nav estimate side
                    if (PbT.shape[0]-Pc.shape[0] == 0) or (PbT.shape[1]-Pc.shape[1] == 0):
                        cond_cov_b = invD_b
                        cond_mean_b = invDd_b
                    else:
                        cond_cov_b_row1 = np.hstack( (invD_b,np.zeros((Pc.shape[0],PbT.shape[1]-Pc.shape[1]))) )
                        cond_cov_b_row2 = np.hstack( (np.zeros((PbT.shape[0]-Pc.shape[0],Pc.shape[1])),np.zeros((PbT.shape[0]-Pc.shape[0],PbT.shape[0]-Pc.shape[0]))) )
                        cond_cov_b = np.vstack( (cond_cov_b_row1,cond_cov_b_row2) )
                        cond_mean_b = np.vstack( (invDd_b,np.zeros((PbT.shape[0]-Pc.shape[0],1))) )

                    Vb = inv(inv(PbT) + cond_cov_b)
                    vb = np.dot(Vb,np.dot(inv(PbT),np.reshape(xbT,(xbT.shape[0],1))) + cond_mean_b)

                    # transform back to normal state order
                    xb = np.take(vb,[0,2,4,1,3,5,6,7,8,9,10,11,12,13,14,15])
                    Pb = Vb[np.ix_([0,2,4,1,3,5,6,7,8,9,10,11,12,13,14,15],[0,2,4,1,3,5,6,7,8,9,10,11,12,13,14,15])]

                    # renormalize quaternions
                    xb[6:10] = xb[6:10]/np.linalg.norm(xb[6:10])

                    agent.nav_filter.x = deepcopy(xb)
                    agent.nav_filter.P = deepcopy(Pb)

                    # update common estimates
                    # for i, filter_ in enumerate(agent.common_estimates):
                    #     if filter_.meas_connection == b_id:
                    #         filter_.x = deepcopy(xc)
                    #         filter_.P = deepcopy(Pc)
                    #         agent.connection_tau_rates[i] = b_rate

            # update state history and mse 
            for j, agent in enumerate(self.agents):
                agent.local_filter.state_history.append(deepcopy(agent.local_filter.x))
                agent.local_filter.cov_history.append(deepcopy(agent.local_filter.P))

                for filter_ in agent.common_estimates:
                    filter_.state_history.append(deepcopy(filter_.x))
                    filter_.cov_history.append(deepcopy(filter_.P))

                # record agent (position) MSE and baseline MSE
                _,idx = agent.get_location(agent.agent_id)
                agent_mse = np.linalg.norm(np.take(agent.local_filter.x,[idx[0],idx[2],idx[4]]) - np.take(agent.true_state[-1],[0,2,4]),ord=2)**2 
                agent.mse_history.append(agent_mse)

                # relative position mse
                for k in range(0,len(agent.meas_connections)):
                    # get est location
                    _,rel_idx = agent.get_location(agent.meas_connections[k])
                    agent_rel_mse = np.linalg.norm( (np.take(agent.local_filter.x,[idx[0],idx[2],idx[4]]) - np.take(agent.local_filter.x,[rel_idx[0],rel_idx[2],rel_idx[4]])) - \
                                    (np.take(agent.true_state[-1],[0,2,4]) - np.take(self.agents[agent.meas_connections[k]].true_state[-1],[0,2,4])),ord=2)**2
                    agent.rel_mse_history[k].append(agent_rel_mse)

                baseline_agent_mse = np.linalg.norm(np.take(self.baseline_filter.x,[j*6,(j*6)+2,(j*6)+4]) - np.take(agent.true_state[-1],[0,2,4]),ord=2)**2 
                self.baseline_filter.mse_history[j].append(baseline_agent_mse)

            # update baseline est and cov histories
            self.baseline_filter.state_history.append(deepcopy(self.baseline_filter.x))
            self.baseline_filter.cov_history.append(deepcopy(self.baseline_filter.P))

        for j,agent in enumerate(self.agents):
            # if self.sim_time_step == 900:
            #     # print('break')
            #     pudb.set_trace()
            nav_agent_mse = np.linalg.norm(agent.nav_filter.x[0:3] - np.take(agent.true_state[-1],[0,2,4]),ord=2)**2
            agent.nav_filter.state_history.append(deepcopy(agent.nav_filter.x))
            agent.nav_filter.cov_history.append(deepcopy(agent.nav_filter.P))
            try:
                agent.nav_mse_history.append(nav_agent_mse)
            except AttributeError:
                agent.nav_mse_history = [nav_agent_mse]

            nav_baseline_agent_mse = np.linalg.norm(self.nav_baselines[j].x[0:3] - np.take(agent.true_state[-1],[0,2,4]),ord=2)**2
            try:
                self.nav_baselines[j].mse_history.append(nav_baseline_agent_mse)
            except AttributeError:
                self.nav_baselines[j].mse_history = [nav_baseline_agent_mse]

    def run_sim(self,print_strs=None,print_status=False):
        """
        Run simulation with sim instance parameters, and package results data.
        Called externally after initialization of sim instance.

        Inputs:

            print_strs -- list of status strings to be printed to terminal, e.g. sim params

        Returns:

            sim_data -- packaged simulation data to be processed
        """
        print_strs.append([])

        # sim update loop
        while self.sim_time < self.max_time+self.nav_dt:
            # update printed status messages
            sim_time_str = 'Sim time: {} of {} sec, {}% complete'.format(
                self.sim_time,self.max_time,100*(self.sim_time/self.max_time))
            print_strs[3] = sim_time_str

            if self.sim_time_step % 10 == 0 and print_status:
                self.print_status(print_strs)

            # update agents
            self.update(self.nav_sensors,self.etddf_sensors,self.etddf_dynamics)

            # increment simulation time
            self.sim_time += self.nav_dt
            self.sim_time_step += 1

        # create empty numpy arrays to store results -> rows=time, columns=agents
        mse_results_array = np.empty((int(self.max_time/(self.etddf_dt))+1,self.num_agents))
        rel_mse_array = [np.empty((int(self.max_time/(self.etddf_dt))+1,len(agent.meas_connections))) for i,agent in enumerate(self.agents)]
        local_filter_history = np.empty(())
        baseline_mse_array = np.empty((int(self.max_time/(self.etddf_dt))+1,self.num_agents))

        baseline_state_history = np.array(self.baseline_filter.state_history)
        baseline_cov_history = np.array(self.baseline_filter.cov_history)

        # nav filter results
        nav_mse_results_array = np.empty((self.sim_time_step,self.num_agents))
        local_filter_history = np.empty(())
        nav_baseline_mse_array = np.empty((self.sim_time_step,self.num_agents))

        nav_baseline_state_history = np.array(self.baseline_filter.state_history)
        nav_baseline_cov_history = np.array(self.baseline_filter.cov_history)

        # etddf message passing statistics
        agent_msgs_total = []
        agent_msgs_sent = []
        agent_ci_total = []
        agent_ci_rate = []

        # all etddf filter histories
        agent_state_histories = []
        agent_cov_histories = []
        agent_true_states = []
        agent_state_error = []
        agent_cov_error = []
        baseline_state_error = []

        # all nav filter histories
        nav_agent_state_histories = []
        nav_agent_cov_histories = []
        nav_agent_true_states = []
        nav_agent_state_error = []
        nav_agent_cov_error = []

        for i,a in enumerate(self.agents):
            # populate metrics for etddf
            mse_results_array[:,i] = np.array(a.mse_history)
            baseline_mse_array[:,i] = np.array(self.baseline_filter.mse_history[i])
            # metrics for nav
            nav_mse_results_array[:,i] = np.squeeze(np.array(a.nav_mse_history))
            nav_baseline_mse_array[:,i] = np.array(self.nav_baselines[i].mse_history)

            agent_state_histories.append(np.array(a.local_filter.state_history))
            agent_cov_histories.append(np.array(a.local_filter.cov_history))
            agent_true_states.append(np.array(a.true_state))

            # relative mse
            for j,conn in enumerate(a.meas_connections):
                rel_mse_array[i][:,j] = np.array(a.rel_mse_history[j])
        
            state_error = []
            cov_error = []
            for j in range(0,len(a.local_filter.state_history)-1):
                id_loc, id_idx = a.get_location(a.agent_id)
                try:
                    state_error.append(np.squeeze(np.take(a.local_filter.state_history[j],[id_idx])) - a.true_state[self.etddf_update_time[j]])
                except IndexError:
                    pudb.set_trace()
                cov_error.append(a.local_filter.cov_history[j][np.ix_(id_idx,id_idx)])
                # assert(state_error[-1].shape == (6,1))
                # pudb.set_trace()
            agent_state_error.append(np.array(state_error))
            agent_cov_error.append(np.array(cov_error))
            # agent_state_error.append(np.array(a.local_filter.state_history)-np.array(a.true_state))

            nav_agent_state_histories.append(np.array(a.nav_filter.state_history))
            nav_agent_cov_histories.append(np.array(a.nav_filter.cov_history))
            nav_agent_true_states.append(np.array(self.true_states[a.agent_id,:,:]))

            nav_state_error = []
            nav_cov_error = []
            for j in range(0,len(a.nav_filter.state_history)):
                gt_quat = self.euler2quat(self.true_states[a.agent_id,9:12,j])
                truth = np.concatenate((np.take(self.true_states[a.agent_id,:,j],[0,1,2,3,4,5]),gt_quat,np.take(self.true_states[a.agent_id,:,j],[15,16,17,18,19,20])))
                nav_state_error.append(a.nav_filter.state_history[j] - truth)
                # nav_cov_error.append(a.nav_filter.cov_history[j][id_idx,id_idx])
            nav_agent_state_error.append(np.array(nav_state_error))

            # nav_agent_state_error.append(np.array(a.nav_filter.state_history)-np.take(self.true_states[a.agent_id,:,:],[0,1,2,3,4,5,9,10,11,15,16,17,18,19,20]))

            agent_msgs_total.append(a.total_msgs)
            agent_msgs_sent.append(a.msgs_sent)
            agent_ci_total.append(a.ci_trigger_cnt)
            agent_ci_rate.append(a.ci_trigger_rate)

            # populate metrics for nav filtering



            # loc,idx = a.get_location(a.agent_id)
            # agent_state_error.append(np.array(a.local_filter.state_history[idx])-np.array(a.true_state))
            # baseline_state_error.append(np.array(baseline_state_history[6*a.agent_id:6*a.agent_id+6])-np.array(a.true_state))
            
        # print ci_process_worst_case_time for each agent
        # for i,a in enumerate(self.agents):
            # print('Agent {} worst case time: {}'.format(i,a.ci_process_worst_case_time))

        # package simulation results
        # res = package_results()
        # results_dict = {'baseline': self.baseline_filter, 'agents': self.agents}
        results_dict = {'etddf_agent_mse': mse_results_array,
                        'etddf_agent_rel_mse': rel_mse_array,
                        'etddf_agent_state_histories': agent_state_histories,
                        'etddf_agent_cov_histories': agent_cov_histories,
                        'etddf_agent_true_states': agent_true_states,
                        'etddf_agent_state_error': agent_state_error,
                        'etddf_agent_cov_error': agent_cov_error,
                        'etddf_baseline_mse': baseline_mse_array,
                        'etddf_baseline_state_history': baseline_state_history,
                        'etddf_baseline_cov_history': baseline_cov_history,
                        # 'baseline_state_error': baseline_state_error,
                        'agent_msgs_total': agent_msgs_total,
                        'agent_msgs_sent': agent_msgs_sent,
                        'agent_ci_total': agent_ci_total,
                        'agent_ci_rate': agent_ci_rate,
                        'nav_agent_mse': nav_mse_results_array,
                        'nav_agent_state_histories': nav_agent_state_histories,
                        'nav_agent_cov_histories': nav_agent_cov_histories,
                        'nav_agent_true_states': nav_agent_true_states,
                        'nav_agent_state_error': nav_agent_state_error,
                        'nav_baseline_mse': nav_baseline_mse_array,
                        'nav_baseline_state_history': nav_baseline_state_history,
                        'nav_baseline_cov_history': nav_baseline_cov_history}
                        
        # create metadata dictionary
        metadata_dict = {'max_time': self.max_time, 
                        'etddf_dt': self.etddf_dt,
                        'nav_dt': self.nav_dt,
                        'connections': self.connections,
                        'num_agents': self.num_agents,
                        'delta': self.delta,
                        'tau_goal': self.tau_state_goal,
                        'msg_drop_prob': self.msg_drop_prob,
                        'dynamics': self.etddf_dynamics,
                        'etddf_sensors': self.etddf_sensors,
                        'nav_sensors': self.nav_sensors}

        return {'metadata': metadata_dict, 'results': results_dict}

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

    def print_status(self,print_strs):
        """
        Print simulation status, formatted nicely w/ input strs.

        Inputs:

            print_strs -- list of strings to be printed to console
        """
        os.system('clear')
        print('------------------------------------------------------')
        for str_ in print_strs:
            print(str_)
        print('------------------------------------------------------')

# global valyes for baseline KF, and agent KF initialization
# TODO: move to helper or config
H_local = np.array( ((1,0,0,0,0,0),
                    (0,0,1,0,0,0),
                    (0,0,0,0,1,0)) )
H_rel = np.array( ((1,0,0,0,0,0,-1,0,0,0,0,0),
                    (0,0,1,0,0,0,0,0,-1,0,0,0),
                    (0,0,0,0,1,0,0,0,0,0,-1,0)) )
Q_local_true = np.array( ((0.0003,0.005,0,0,0,0),
                            (0.005,0.1,0,0,0,0),
                            (0,0,0.0003,0.005,0,0),
                            (0,0,0.005,0.1,0,0),
                            (0,0,0,0,0.0003,0.005),
                            (0,0,0,0,0.005,0.1)) )


# main driver function

def main(plot=False,cfg_path=None,save_path=None,print_status=False):
    """
    Main driver function. Called when running sim.py directly or as a module.

    Inputs:

        plot [optional] -- flag to determine if sim results will be plotted
        cfg_path [optional] -- string path to config file (full path)
        save_path [optional] --  string path to save location for sim results (full path)

    Returns:

        none
    """

    # load sim config
    if cfg_path is None:
        cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                        '../config/config.yaml'))
    cfg = load_config(cfg_path)

    # specify path for saving data
    if save_path is None:
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                        '../data/'))

    # make new data directory for sim files
    save_path = make_data_directory(save_path)

    # extract config params
    num_mc_sim = cfg['num_mc_sims']
    delta_values = cfg['delta_values']
    tau_values = cfg['tau_values']
    msg_drop_prob_values = cfg['msg_drop_prob_values']

    # initilize results container
    results = []

    # sim counter
    sim_cnt = 1
    total_sims = len(delta_values)*len(tau_values)*len(msg_drop_prob_values)*num_mc_sim

    for i in delta_values:
        for j in tau_values:
            for k in msg_drop_prob_values:

                # numpy array for monte carlo averaged results for etddf filters
                etddf_mc_mse_results = np.empty((int(cfg['max_time']/cfg['etddf_dt'] + 1),len(cfg['agent_cfg']['conns']),num_mc_sim))
                etddf_mc_baseline_mse_results = np.empty((int(cfg['max_time']/cfg['etddf_dt'] + 1),len(cfg['agent_cfg']['conns']),num_mc_sim))
                etddf_mc_rel_mse_results = [np.empty((int(cfg['max_time']/cfg['etddf_dt'] + 1),len(cfg['agent_cfg']['conns'][x]),num_mc_sim)) for x in range(0,len(cfg['agent_cfg']['conns']))]
                etddf_state_results = []
                etddf_cov_results = []
                etddf_baseline_results = []
                etddf_true_states = []
                etddf_state_error = []
                etddf_cov_error = []

                # monte carlo averaged nav filter results
                nav_mc_mse_results = np.empty((int(cfg['max_time']/cfg['nav_dt'] + 1),len(cfg['agent_cfg']['conns']),num_mc_sim))
                nav_mc_baseline_mse_results = np.empty((int(cfg['max_time']/cfg['nav_dt'] + 1),len(cfg['agent_cfg']['conns']),num_mc_sim))
                nav_state_results = []
                nav_cov_results = []
                nav_baseline_results = []
                nav_true_states = []
                nav_state_error = []

                mc_msgs_total = np.empty((len(cfg['agent_cfg']['conns']),num_mc_sim))
                mc_msgs_sent = np.empty((len(cfg['agent_cfg']['conns']),num_mc_sim))
                mc_ci_total = np.empty((len(cfg['agent_cfg']['conns']),num_mc_sim))
                mc_ci_rate = np.empty((len(cfg['agent_cfg']['conns']),num_mc_sim))

                for m in range(1,num_mc_sim+1):
                    # create simulation status strings to be printed
                    sim_print_str = 'Initializing simulation {} of {}'.format(sim_cnt,total_sims)
                    param_print_str = 'Params: delta={},\t tau={}, \t msg drop prob={}'.format(i,j,k)
                    mc_print_str = 'Monte Carlo sim {} of {}'.format(m,num_mc_sim)

                    # create sim instance w/ sim params
                    sim = NavSimInstance(delta=i,tau=j,msg_drop_prob=k,
                                        baseline_cfg=cfg['baseline_cfg'],
                                        nav_baseline_cfg=cfg['nav_baseline_cfg'],
                                        agent_cfg=cfg['agent_cfg'],
                                        nav_agent_cfg=cfg['nav_agent_cfg'],
                                        max_time=cfg['max_time'],
                                        etddf_dt=cfg['etddf_dt'],
                                        nav_dt=cfg['nav_dt'],
                                        use_adaptive_tau=cfg['use_adaptive_tau'],
                                        fixed_rng=cfg['fixed_rng'],
                                        process_noise=False,
                                        sensor_noise=False,
                                        quantization_flag=cfg['quantization'],
                                        diagonalization_flag=cfg['diagonalization'])
                    # run simulation
                    res = sim.run_sim([sim_print_str,param_print_str,mc_print_str],print_status)
                    # add results to results container
                    # results.append(res)
                    etddf_mc_mse_results[:,:,m-1] = res['results']['etddf_agent_mse']
                    etddf_mc_baseline_mse_results[:,:,m-1] = res['results']['etddf_baseline_mse']
                    nav_mc_mse_results[:,:,m-1] = res['results']['nav_agent_mse']
                    nav_mc_baseline_mse_results[:,:,m-1] = res['results']['nav_baseline_mse']

                    for ii in range(0,len(cfg['agent_cfg']['conns'])):
                        etddf_mc_rel_mse_results[ii][:,:,m-1] = res['results']['etddf_agent_rel_mse'][ii]

                    etddf_state_error.append(res['results']['etddf_agent_state_error'])
                    etddf_cov_results.append(res['results']['etddf_agent_cov_histories'])
                    etddf_cov_error.append(res['results']['etddf_agent_cov_error'])
                    nav_state_error.append(res['results']['nav_agent_state_error'])
                    nav_cov_results.append(res['results']['nav_agent_cov_histories'])

                    etddf_true_states.append(res['results']['etddf_agent_true_states'])
                    nav_true_states.append(res['results']['nav_agent_true_states'])

                    mc_msgs_total[:,m-1] = res['results']['agent_msgs_total']
                    mc_msgs_sent[:,m-1] = res['results']['agent_msgs_sent']
                    mc_ci_total[:,m-1] = res['results']['agent_ci_total']
                    mc_ci_rate[:,m-1] = res['results']['agent_ci_rate']

                    sim_cnt += 1

                etddf_mc_avg_mse_results = np.mean(etddf_mc_mse_results,axis=2)
                etddf_mc_std_mse_results = np.std(etddf_mc_mse_results,axis=2)
                etddf_mc_avg_baseline_mse_results = np.mean(etddf_mc_baseline_mse_results,axis=2)
                etddf_mc_std_baseline_mse_results = np.std(etddf_mc_baseline_mse_results,axis=2)
                etddf_mc_avg_rel_mse_results = [np.mean(etddf_mc_rel_mse_results[x],axis=2) for x in range(0,len(etddf_mc_rel_mse_results))]
                etddf_mc_std_rel_mse_results = [np.std(etddf_mc_rel_mse_results[x],axis=2) for x in range(0,len(etddf_mc_rel_mse_results))]

                nav_mc_avg_mse_results = np.mean(nav_mc_mse_results,axis=2)
                nav_mc_std_mse_results = np.std(nav_mc_mse_results,axis=2)
                nav_mc_avg_baseline_mse_results = np.mean(nav_mc_baseline_mse_results,axis=2)
                nav_mc_std_baseline_mse_results = np.std(nav_mc_baseline_mse_results,axis=2)

                # compute monte carlo sim averaged state error and covariance for ettdf instances
                etddf_state_error_mc_avg = [etddf_state_error[0][x] for x in range(0,len(etddf_state_error[0]))]
                etddf_cov_histories_mc_avg = [etddf_cov_results[0][x] for x in range(0,len(etddf_cov_results[0]))]
                etddf_cov_error_mc_avg = [etddf_cov_error[0][x] for x in range(0,len(etddf_cov_error[0]))]
                for ii in range(1,len(etddf_state_error)):
                    for jj in range(0,len(etddf_state_error[ii])):
                        etddf_state_error_mc_avg[jj] += etddf_state_error[ii][jj]
                        etddf_cov_histories_mc_avg[jj] += etddf_cov_results[ii][jj]
                        etddf_cov_error_mc_avg[jj] += etddf_cov_error[ii][jj]

                etddf_state_error_mc_avg = [etddf_state_error_mc_avg[x]/num_mc_sim for x in range(0,len(etddf_state_error_mc_avg))]
                etddf_cov_histories_mc_avg = [etddf_cov_histories_mc_avg[x]/num_mc_sim for x in range(0,len(etddf_cov_histories_mc_avg))]
                etddf_cov_error_mc_avg = [etddf_cov_error_mc_avg[x]/num_mc_sim for x in range(0,len(etddf_cov_error_mc_avg))]

                # compute monte carlo sim averaged state error and covariance for nav filter instances
                nav_state_error_mc_avg = [nav_state_error[0][x] for x in range(0,len(nav_state_error[0]))]
                nav_cov_histories_mc_avg = [nav_cov_results[0][x] for x in range(0,len(nav_cov_results[0]))]
                for ii in range(1,len(nav_state_error)):
                    for jj in range(0,len(nav_state_error[ii])):
                        nav_state_error_mc_avg[jj] += nav_state_error[ii][jj]
                        nav_cov_histories_mc_avg[jj] += nav_cov_results[ii][jj]

                nav_state_error_mc_avg = [nav_state_error_mc_avg[x]/num_mc_sim for x in range(0,len(nav_state_error_mc_avg))]
                nav_cov_histories_mc_avg = [nav_cov_histories_mc_avg[x]/num_mc_sim for x in range(0,len(nav_cov_histories_mc_avg))]

                mc_avg_msgs_total = np.mean(mc_msgs_total,axis=1)
                mc_avg_msgs_sent = np.mean(mc_msgs_sent,axis=1)
                mc_avg_ci_total = np.mean(mc_ci_total,axis=1)
                mc_avg_ci_rate = np.mean(mc_ci_rate,axis=1)

                mc_std_msgs_total = np.std(mc_msgs_total,axis=1)
                mc_std_msgs_sent = np.std(mc_msgs_sent,axis=1)
                mc_std_ci_total = np.std(mc_ci_total,axis=1)
                mc_std_ci_rate = np.std(mc_ci_rate,axis=1)

                results = {'etddf_mse': etddf_mc_avg_mse_results,
                            'etddf_rel_mse': etddf_mc_avg_rel_mse_results,
                            'etddf_baseline_mse': etddf_mc_avg_baseline_mse_results,
                            'etddf_state_error': etddf_state_error_mc_avg,
                            'etddf_cov_error': etddf_cov_error_mc_avg,
                            'etddf_state_history': etddf_state_results,
                            'etddf_cov_history': etddf_cov_histories_mc_avg,
                            'etddf_true_states': etddf_true_states,
                            'nav_mse': nav_mc_avg_mse_results,
                            'nav_baseline_mse': nav_mc_avg_baseline_mse_results,
                            'nav_state_error': nav_state_error_mc_avg,
                            'nav_state_history': nav_state_results,
                            'nav_cov_history': nav_cov_histories_mc_avg,
                            'nav_true_states': nav_true_states,
                            'msgs_total': mc_avg_msgs_total,
                            'msgs_sent': mc_avg_msgs_sent,
                            'ci_total': mc_avg_ci_total,
                            'ci_rate': mc_avg_ci_rate,
                            'etddf_mse_std': etddf_mc_std_mse_results,
                            'etddf_rel_mse_std': etddf_mc_std_rel_mse_results,
                            'etddf_baseline_mse_std': etddf_mc_std_baseline_mse_results,
                            'nav_mse_std': nav_mc_std_mse_results,
                            'nav_mse_baseline_std': nav_mc_std_baseline_mse_results,
                            'msgs_total_std': mc_std_msgs_total,
                            'msgs_sent_std': mc_std_msgs_sent,
                            'ci_total_std': mc_std_ci_total,
                            'ci_rate_std': mc_std_ci_rate}

                # create metadata dictionary
                metadata_dict = {'num_mc_sim': num_mc_sim,
                                'delta_value': (i,delta_values),
                                'tau_value': (j,tau_values),
                                'msg_drop_prob_value': (k,msg_drop_prob_values)}

                # create filename
                file_name = 'delta{}_tau{}_drop{}_mc{}'.format(i,j,k,num_mc_sim).replace('.','')

                # save data to pickle file
                save_sim_data(metadata_dict,results,save_path,file_name)

    # write metadata file for data
    sim_metadata = {'cfg': cfg}
    write_metadata_file(save_path,sim_metadata)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a set of simulations using the ET-DDF framework.')
    parser.add_argument('-p','--plot',dest='plot_flag',action='store_true',
                            help='plot the simulation results at the end (not suggested unless you are only running one sim')
    parser.add_argument('-c','--config-path', dest='config_path',action='store',
                            help='specify the (full) path to a sim config file.')
    parser.add_argument('-s','--save-path', dest='save_path',action='store',
                            help='specify the (full) path to the location where sim data will be saved.')
    parser.add_argument('-t','--print-status',dest='print_flag',action='store_true',
                            help='print simulation status to the terminal')
    args = parser.parse_args()

    # run the sim driver with command line args
    main(plot=args.plot_flag,cfg_path=args.config_path,save_path=args.save_path,print_status=args.print_flag)