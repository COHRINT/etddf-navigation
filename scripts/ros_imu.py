#!/usr/bin/env python

"""
Simulated IMU measurements in ROS framework.
"""

import os
import sys
import rospy
import numpy as np

from sensor_msgs.msg import Imu

class ROSImu:
    """
    """
    def __init__(self,imu_gt_topic,pub_topic,config_location):

        rospy.init_node('simulated_imu')

        self.pub = rospy.Publisher(pub_topic,Imu,queue_size=10)
        rospy.Subscriber(imu_gt_topic,Imu,self.msg_cb)

        # load sensor config

        # set sensor params
        self.rate = rospy.get_param('rate')

        self.accel_noise = rospy.get_param('accel_noise')
        self.accel_bias_noise = rospy.get_param('accel_bias')
        self.accel_bias_tc = rospy.get_param('accel_bias_tc')

        self.gyro_noise = rospy.get_param('gyro_noise')
        self.gyro_bias_noise = rospy.get_param('gyro_bias')
        self.gyro_bias_tc = rospy.get_param('gyro_bias_tc')

        self.last_accel_state = np.zeros((3,))
        self.last_gyro_state = np.zeros((3,))
        
        self.last_accel_bias = np.zeros((3,))
        self.last_gyro_bias = np.zeros((3,))

        rospy.spin()

    def generate_measurement(self,accel_x_gt,accel_y_gt,accel_z_gt,gyro_x_gt,gyro_y_gt,gyro_z_gt):
        """
        Generates a simulated imu measurement from ground truth data by adding
        white noise and bias froma 1st order Gauss-Markov process.

        Assumes all accel and gyro axis are independent, and ave same noise params.
        """
        accel_gt = np.array([accel_x_gt,accel_y_gt,accel_z_gt])
        gyro_gt = np.array([gyro_x_gt,gyro_y_gt,gyro_z_gt])

        accel_bias = self.last_accel_bias + np.exp(-(1/self.rate)/self.accel_bias_tc)*np.random.multivariate_normal([0,0,0],np.eye(3)*self.accel_bias_noise)
        accel_noise = np.random.multivariate_normal([0,0,0],self.accel_noise)
        accel_meas = accel_gt + accel_bias + accel_noise

        self.last_accel_bias = accel_bias
        self.last_accel_state = accel_gt

        gyro_bias = self.last_gyro_bias + np.exp(-(1/self.rate)/self.gyro_bias_tc)*np.random.multivariate_normal([0,0,0],np.eye(3)*self.gyro_bias_noise)
        gyro_noise = np.random.multivariate_normal([0,0,0],self.gyro_noise)
        gyro_meas = gyro_gt + gyro_bias + gyro_noise

        self.last_gyro_bias = gyro_bias
        self.last_gyro_state = gyro_gt

        return accel_meas[0], accel_meas[1], accel_meas[2], gyro_meas[0], gyro_meas[1],gyro_meas[2]

    def publish_msg(self,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z):
        """
        Create new IMU message with simulated values.
        """
        msg = Imu()

        msg.header.stamp = rospy.Time.now()
        msg.linear_acceleration.x = accel_x
        msg.linear_acceleration.y = accel_y
        msg.linear_acceleration.z = accel_z
        msg.angular_velocity.x = gyro_x
        msg.angular_velocity.y = gyro_y
        msg.angular_velocity.z = gyro_z

        self.pub.publish(msg)

    def msg_cb(self,msg):
        """
        Callback for groundtruth imu data.
        """
        # extract data from msg
        accel_x_gt = msg.linear_acceleration.x
        accel_y_gt = msg.linear_acceleration.y
        accel_z_gt = msg.linear_acceleration.z
        gyro_x_gt = msg.angular_velocity.x
        gyro_y_gt = msg.angular_velocity.y
        gyro_z_gt = msg.angular_velocity.z

        # generate measurement with noise and bias
        ax,ay,az,gx,gy,gz = self.generate_measurement(accel_x_gt,accel_y_gt,accel_z_gt,gyro_x_gt,gyro_y_gt,gyro_z_gt)

        # publish new measurement
        self.publish_msg(ax,ay,az,gx,gy,gz)

if __name__ == "__main__":
    gt_topic = sys.argv[1]
    pub_topic = sys.argv[2]
    if len(sys.argv) > 3:
        config_location = sys.argv[3]
    else:
        config_location = None

    r = ROSImu(gt_topic,pub_topic,config_location)