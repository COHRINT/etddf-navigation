#!/usr/bin/env python

from __future__ import division

import numpy as np

"""
Utility functions for converting between euler angles and quaternions.
"""

def euler2quat(angles,deg=False):
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

def ENU2NED(x):
    """
    Convert input vector from ENU coordinates to NED coordinates.
    """
    x_out = np.empty_like(x)
    x_out[0] = x[1]
    x_out[1] = x[0]
    x_out[2] = -x[2]
    return x_out

def NED2ENU(x):
    """
    Convert input vector from NED to ENU coordinates.
    """
    x_out = np.empty_like(x)
    x_out[0] = x[1]
    x_out[1] = x[0]
    x_out[2] = -x[2]
    return x_out