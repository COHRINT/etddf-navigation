% Computing derivatives of quaternion for quaternion to euler covariance

clear; close all; clc;

syms q0 q1 q2 q3

phi = atan(2*(q0*q1 + q2*q3)/(1-2*(q1^2 + q2^2)));
theta = asin(2*(q0*q2 - q3*q1));
psi = atan(2*(q0*q3+q1*q2)/(1-2*(q2^2+q3^2)));

dphi_dq0 = diff(phi,q0)
dphi_dq1 = diff(phi,q1)
dphi_dq2 = diff(phi,q2)
dphi_dq3 = diff(phi,q3)

dtheta_dq0 = diff(theta,q0)
dtheta_dq1 = diff(theta,q1)
dtheta_dq2 = diff(theta,q2)
dtheta_dq3 = diff(theta,q3)

dpsi_dq0 = diff(psi,q0)
dpsi_dq1 = diff(psi,q1)
dpsi_dq2 = diff(psi,q2)
dpsi_dq3 = diff(psi,q3)

% save('compass_meas_mat_derivs',dpsi_dq0,dpsi_dq1,dpsi_dq2,dpsi_dq3)