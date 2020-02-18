% Computing derivatives of quaternion for compass measurement matrix

clear; close all; clc;

syms q0 q1 q2 q3

psi = atan(2*(q0*q3+q1*q2)/(1-2*(q2^2+q3^2)));

dpsi_dq0 = diff(psi,q0)
dpsi_dq1 = diff(psi,q1)
dpsi_dq2 = diff(psi,q2)
dpsi_dq3 = diff(psi,q3)

save('compass_meas_mat_derivs',dpsi_dq0,dpsi_dq1,dpsi_dq2,dpsi_dq3)