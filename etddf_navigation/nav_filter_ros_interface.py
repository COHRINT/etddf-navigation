#!/usr/bin/env python

"""
ROS nav filter interface to Gazebo simulation environment.
"""

import sys
import rospy
import numpy as np

from sensor_msgs.msg import Imu, MagneticField, FluidPressure
from nav_msgs.msg import Odometry
# from etddf_ros.msg import gpsMeasurement

from strapdown_ins import StrapdownINS

class NavFilterROSInterface:
    """
    Subscribes to sensor topics in ROS framework, and sends measurements
    to nav filter.
    """
    def __init__(self,sensor_topics,filtering=False,numpy_filename=None):
        rospy.init_node('nf_interface')

        # if we want to actively filter data, else just collect and save
        self.filtering = filtering
        if self.filtering:
            self.nav_filter = StrapdownINS('',0.1)
        else:
            self.imu_data = np.empty((1,6))
            self.gps_data = np.empty((1,3))
            self.compass_data = np.empty((1,1))
            self.depth_data = np.empty((1,1))
            self.gt_data = np.empty((1,13))
            # self.gt_twist_data = np.empty((6,))

        self.imu_sub = rospy.Subscriber(sensor_topics['imu'],Imu,self.imu_callback)
        # self.gps_sub = rospy.Subscriber(gps_topic,Gps,self.gps_callback)
        # self.compass_sub = rospy.Subscriber(compass_topic,Compass,self.compass_callback)
        # self.depth_sub = rospy.Subscriber(depth_topic,Depth,self.depth_callback)VC
        self.gt_sub = rospy.Subscriber(sensor_topics['gt'],Odometry,self.gt_callback)

        rospy.spin()
        print('shutting down')
        if numpy_filename is not None:
            np.save(numpy_filename + '_imu_data.npy',self.imu_data)
            np.save(numpy_filename + '_ground_truth_data.npy',self.gt_data)

    def imu_callback(self,msg):

        # update nav filter or save data
        if self.filtering:
            # self.nav_filter.propagate()
            pass
        else:
            lin_accel_x = msg.linear_acceleration.x
            lin_accel_y = msg.linear_acceleration.y
            lin_accel_z = msg.linear_acceleration.z
            ang_vel_x = msg.angular_velocity.x
            ang_vel_y = msg.angular_velocity.y
            ang_vel_z = msg.angular_velocity.z
            meas = np.array([lin_accel_x,lin_accel_y,lin_accel_z,ang_vel_x,ang_vel_y,ang_vel_z])
            meas = np.atleast_2d(meas)
            self.imu_data = np.concatenate((self.imu_data,meas),axis=0)

    def gps_callback(self,msg):
        pass

    def compass_callback(self,msg):
        pass

    def depth_callback(self,msg):
        pass

    def gt_callback(self,msg):
        
        # save data
        pose_position_x = msg.pose.pose.position.x
        pose_position_y = msg.pose.pose.position.y
        pose_position_z = msg.pose.pose.position.z
        pose_orientation_x = msg.pose.pose.orientation.x # quaternion
        pose_orientation_y = msg.pose.pose.orientation.y
        pose_orientation_z = msg.pose.pose.orientation.z
        pose_orientation_w = msg.pose.pose.orientation.w
        twist_linear_velocity_x = msg.twist.twist.linear.x
        twist_linear_velocity_y = msg.twist.twist.linear.y
        twist_linear_velocity_z = msg.twist.twist.linear.z
        twist_ang_velocity_x = msg.twist.twist.angular.x # roll, pitch, yaw rates
        twist_ang_velocity_y = msg.twist.twist.angular.y
        twist_ang_velocity_z = msg.twist.twist.angular.z

        meas = np.array([pose_position_x,
                            pose_position_y,
                            pose_position_z,
                            pose_orientation_x,
                            pose_orientation_y,
                            pose_orientation_z,
                            pose_orientation_w,
                            twist_linear_velocity_x,
                            twist_linear_velocity_y,
                            twist_linear_velocity_z,
                            twist_ang_velocity_x,
                            twist_ang_velocity_y,
                            twist_ang_velocity_z])

        meas = np.atleast_2d(meas)

        self.gt_data = np.concatenate((self.gt_data,meas),axis=0)


if __name__ == "__main__":
    numpy_filename = sys.argv[1]
    imu_topic = '/bluerov2_0/imu'
    gt_topic = '/bluerov2_0/pose_gt'
    sensor_topics = {'imu': imu_topic, 'gt': gt_topic}
    NavFilterROSInterface(sensor_topics,filtering=False,numpy_filename=numpy_filename)