from math import *
import numpy as np

def kalman_filter(last_pose, last_covariance, u, z, c, m):

    alpha = np.array([1, 1, 1, 1])
    theta = last_pose["th_deg"] * (pi / 180)
    vt = u(0)
    wt = u(1)

    dt = 0.001

    

    G13 = -(vt / wt) * cos(theta) + (vt / wt) * cos(theta + wt*dt)
    G23 = -(vt / wt) * sin(theta) + (vt / wt) * sin(theta + wt*dt)
    G = np.matrix([[1, 0, G13], [0, 1, G23], [0, 0, 1]])

    Vt11 = (-sin(theta) + sin(theta + wt*dt))/wt
    Vt21 = (cos(theta) - cos(theta + wt*dt))/wt
    Vt12 = (vt*(sin(theta) - sin(theta + wt*dt))/wt**2) + (vt * cos(theta + wt*dt) * dt)/wt
    Vt22 = -(vt*(cos(theta) - cos(theta + wt*dt))/wt**2) + (vt * sin(theta + wt*dt) * dt)/wt
    Vt = np.matrix([[Vt11, Vt12], [Vt21, Vt22], [0, dt]])

    a1 = alpha[0]
    a2 = alpha[1]
    a3 = alpha[2]
    a4 = alpha[3]
    M = np.matrix([[a1*vt**2 + a2*wt**2, 0], [0, a3*vt**2 + a4*wt**2]])

    estimated_pose = last_pose + np.array([G23, -G13, wt*dt])
    estimated_covariance = G * last_covariance * np.transpose(G) + Vt * M * np.transpose(Vt)