# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:55:42 2021
@author: Faiza
"""
import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import random
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from numpy import save
import os
import cv2
from PIL import Image
import matplotlib.ticker as mtick
import sympy as sym
import math
from time import process_time
import time
import timeit
import datetime
import multiprocessing as mp
from multiprocessing import Pool
from profilehooks import profile
import copy
from collections import deque
import pandas as pd
def getNodesVolumes(section_radius, section_length, node_length):
    radius_section_start = section_radius[0:-1]  # wo die section beginnt
    radius_section_end = section_radius[1:] # dort wo die section endet
    slope = radius_section_end - radius_section_start / section_length  # steigung in der section
    slope_left_node_part = slope[0:-1]  # slope_leftpart_node
    slope_right_node_part = slope[1:]  # slope_rightpart_node
    radius_at_mid_node = radius_section_start[1:]



    radius_node0_end = radius_section_start[0] + (node_length / 2) * slope_left_node_part[0]
    radius_node_start = radius_at_mid_node - (node_length / 2) * slope_left_node_part
    radius_node_end = radius_at_mid_node + (node_length / 2) * slope_right_node_part
    V_node_left_part = 1 / 3 * np.pi * (node_length / 2) * np.square(radius_node_start + radius_at_mid_node)
    V_node_right_part = 1 / 3 * np.pi * (node_length / 2) * np.square(radius_node_end + radius_at_mid_node)
    V_node0_left = 1 / 3 * np.pi * (node_length / 2) * np.square(
        radius_node0_end + radius_section_start[0])
    return radius_node0_end, radius_node_start, radius_node_end, V_node0_left, V_node_left_part, V_node_right_part



def get_stenosis_c_sec(N_sec,L, c, s_grad):
    x0 = -L*100/2
    xf = L*100/2
    nx = N_sec  # make this an odd number, so that we will have a segmentation at the midpoint
    distance = 0  # for removal of segmentations surrounding the midsection

    midsection_percentage = s_grad  # midsection area as a percentage of the inlet area
    radius_inlet = c*100

    sigma = 0.010  # controls the spread of the stenosis/aneurysm
    mu = 0.0  # controls the center of the stenosis/aneurysm

    x = np.linspace(x0, xf, nx)
    mid_index = math.floor(len(x) / 2.0)
    x = np.concatenate((x[0:mid_index - distance], np.array([x[mid_index]]), x[mid_index + 1 + distance:]))

    area_midsection = (midsection_percentage / 100.0) * np.pi * radius_inlet ** 2.0
    radius_midsection = np.sqrt(area_midsection / np.pi)
    A = -1.0 * (radius_midsection - radius_inlet) * np.sqrt(2.0 * np.pi * sigma ** 0.5) / np.exp(
        -1.0 * ((0.0 - mu) ** 2 / 2 / sigma ** 0.5))  # A = 4.856674471372556

    stenosis_r_cm = radius_inlet- A / np.sqrt(2.0 * np.pi * sigma ** 0.5) * np.exp(-1.0 * ((x - mu) ** 2 / 2 / sigma ** 0.5))

    return stenosis_r_cm/100

def get_expansion_c_sec(N_sec,L, c, s_grad):
    x0 = 0 * 100
    xf = L * 100
    nx = N_sec  # make this an odd number, so that we will have a segmentation at the midpoint

    x = np.linspace(x0, xf, nx)

    distance = 0  # for removal of segmentations surrounding the midsection
    r_inlet = c * 100
    expansion_percentage = s_grad  # midsection area as a percentage of the inlet area

    A_inlet = np.pi * (r_inlet) ** 2
    A_outlet = (expansion_percentage / 100) * A_inlet
    r_outlet = np.sqrt(A_outlet / np.pi)
    m = (r_outlet - r_inlet) / (xf - x0)
    print("m: ", m)
    r = m * x + r_inlet
    last_radius = m * (r[-1] - r[-2]) + r[-1]
    return r / 100, last_radius/100

class Model_Flow_Efficient:
    def __init__(self,pdic):
        self.am1_0_vol = 1
        self.am1_0_pd = 1

        self.inp_ent = pdic['inp']
        self.inp_ext = pdic['out']

        self.N_sec = pdic['N_sec']
        self.N_nodes = pdic['N_nodes']
        self.samples = pdic['samples']

        self.u_e = pdic['u_e']
        self.l = pdic['L'] / self.N_sec
        self.length_node = self.l*10**-3
        self.c_sec = pdic['c'] * np.ones(self.N_sec)


        if pdic['stenosis'] == True:
            self.c_sec = get_stenosis_c_sec(self.N_sec,pdic['L'], pdic['c'], pdic['grad'])
        if pdic['expansion'] == True:
            self.c_sec, self.last_radius = get_expansion_c_sec(self.N_sec, pdic['L'], pdic['c'], pdic['grad'])

        self.c_sec_end = np.hstack((self.c_sec[1:], self.last_radius))

        self.h = pdic['h']

        self.poi_rat = pdic['poi_rat']
        self.k1 = pdic['k1']
        self.k2 = pdic['k2']
        self.k3 = pdic['k3']

        self.beta_1 = pdic['beta_1']
        self.beta_2 = pdic['beta_2']

        self.lambda_1 = pdic['lame1']
        self.lambda_2 = pdic['lame2']

        if pdic['FEM'] == True:
            self.k_st = np.array([(self.poi_rat / ((1 - 2 * self.poi_rat) * (1 + self.poi_rat))) * (self.l / np.pi) * (
                        self.k1 * (np.exp(self.k2 * (self.c_sec[i]))) + self.k3 ) for i in range(self.N_sec)])
            self.k_st = self.k_st[0]*np.ones(self.N_sec)
            self.k_c_st = np.array([0.5 * (
                        self.k1 * (np.exp(self.k2 * (self.c_sec[i])))+ self.k3) * (np.pi * np.square(self.c_sec[i]) / self.l) * (1 / (1 + self.poi_rat)) for i in range(self.N_sec)])
            self.k_c_st = self.k_c_st[0]*np.ones(self.N_sec)
        elif pdic['stiff_wall'] == True:
            self.k_c_st = pdic['kc_stiff']*np.ones(self.N_sec)
            self.k_st = pdic['k_stiff']*np.ones(self.N_sec)
        else:
            self.k_st = (pdic['beta_1'] * self.lambda_1 * self.l * self.h / (np.pi * self.c)) * np.ones(self.N_sec)
            self.k_c_st = (pdic['beta_2'] * self.lambda_2 * np.pi * self.c * self.h / (self.l)) * np.ones(self.N_sec)  # coupling spring constant



        self.geo_dissipation = pdic['geo_dissipation']
        self.vis_dissipation = pdic['vis_dissipation']

        self.m_st = np.multiply(np.pi * pdic['rho_st'] * self.c_sec,  pdic['h'] * self.l * np.ones(self.N_sec))
        self.d_st = pdic['d_st'] * np.ones(self.N_sec)
        self.d_c_st = pdic['d_c_st'] * np.ones(self.N_sec)  # coupling damping coefficient of structure
        self.section_radius_complete = np.hstack((self.c_sec, self.last_radius))
        radius_node0_end, radius_node_start, radius_node_end, self.volume_node0, self.volume_node_leftpart, self.volume_node_rightpart = getNodesVolumes(
            self.section_radius_complete, self.l, self.length_node)
        volume_nodes_complete = np.hstack((self.volume_node0, self.volume_node_leftpart + self.volume_node_rightpart))
        self.m_n = np.multiply(pdic['rho_f'], volume_nodes_complete)
        #self.m_n = np.multiply(pdic['rho_f'] * np.pi * np.square(self.c_sec),  self.l * 10 ** -3 * np.ones(self.N_nodes))
       # self.m_n[0] = self.m_n[0] /2
        self.r_sec = pdic['rho_f']
        self.vf = pdic['vf']
        self.b_st = pdic['b_st']


        self.dyn_pressure = np.empty((self.N_nodes, int(self.samples)))
        self.dyn_pressure_mmHg = np.empty((self.N_nodes, int(self.samples)))
        self.stat_pressure = np.empty((self.N_nodes, int(self.samples)))
        self.stat_pressure_mmHg = np.empty((self.N_nodes, int(self.samples)))
        self.r_n_save = np.empty((self.N_nodes, int(self.samples)))
        self.section_mom_save = np.empty((self.N_nodes, int(self.samples)))
        self.wall_mom_save = np.empty((self.N_nodes, int(self.samples)))
        self.radius_sec_save = np.empty((self.N_nodes, int(self.samples)))
        self.mass_node_save = np.empty((self.N_nodes, int(self.samples)))
        self.Volume_node_save = np.empty((self.N_nodes, int(self.samples)))
        self.total_pressure = np.empty((self.N_nodes, int(self.samples)))
        self.total_pressure_mmHg = np.empty((self.N_nodes, int(self.samples)))


        self.mmHgFactor = pdic['mmHgFactor']

        self.H_st = np.empty((int(self.samples)))
        self.H_fl = np.empty((int(self.samples)))
        self.E_fl = np.empty((int(self.samples)))
        self.inp_val = np.empty((int(self.samples), 2))
        self.out_val = np.empty((int(self.samples), 2))
        self.A_sec = np.empty((self.N_sec, int(self.samples)))
        self.V_sec = np.empty((self.N_sec, int(self.samples)))
        self.A_n = np.empty((self.N_nodes, int(self.samples)))
        self.A_c = np.empty((self.N_sec, int(self.samples)))
        self.A_circ = np.empty((self.N_sec, int(self.samples)))
        self.p_d = np.empty((self.N_nodes, int(self.samples)))

        self.H_q_st = np.empty((self.N_sec, int(self.samples)))
        self.H_i_st = np.empty((self.N_sec, int(self.samples)))
        self.H_i_sec = np.empty((self.N_sec, int(self.samples)))
        self.H_r_n = np.empty((self.N_nodes, int(self.samples)))

        self.L = pdic['L']

        self.geo_dissipation_factor =pdic['geo_factor']
        # *self.l / self.c_sec
        self.vis_dissipation_factor = pdic['vis_factor']*2*self.l / self.c_sec

    def PHModel_old(self, t, x):
        u_p = np.concatenate(([float(self.inp_ent(t))], [self.inp_ext(t)]))
        x = x.reshape(-1, )
        q_st = x[0:self.N_sec]
        i_st = x[self.N_sec:2 * self.N_sec]
        i_sec = x[2 * self.N_sec:3 * self.N_sec]
        r_n = x[3 * self.N_sec:]



        A_sec = np.multiply(np.pi * self.c_sec,(q_st + self.c_sec))

        alpha_1 = A_sec[1:-1] / (A_sec[1:-1] + A_sec[2:])  # check
        alpha_2 = 1 - (A_sec[0:-2] / (A_sec[0:-2] + A_sec[1:-1]))  # check

        V_sec = self.l * A_sec[1:-1] - (self.m_n[2:] / r_n[2:]) * alpha_1 - (
                    self.m_n[1:-1] / r_n[1:-1]) * alpha_2  # check
        V_sec = np.concatenate((np.reshape(
            self.l * A_sec[0] - self.m_n[1] * (1 / r_n[1]) * (A_sec[0] / (A_sec[0] + A_sec[1])) - self.m_n[0] * (
                    1 / r_n[0]) * float(self.am1_0_vol), (1,)), V_sec, np.reshape(
            A_sec[-1] * self.l - self.m_n[-1] * (1 / r_n[-1]) * (1 - (A_sec[-2] / (A_sec[-2] + A_sec[-1]))),
            (1,))))  # check

        A_n = np.hstack((np.reshape(self.m_n[0] / (r_n[0] * (self.c_sec[0] + q_st[0] )), (1,)),
                         self.m_n[1:] / (r_n[1:] * (2 * self.c_sec[1:] + q_st[0:-1] + q_st[1:]))))  # check

        A_c = V_sec / (q_st + self.c_sec)  # check

        alpha = A_sec[0:-1] / (A_sec[0:-1] + A_sec[1:])

        p_d = np.concatenate((np.reshape(
            (1 * self.r_sec / 2) * np.square((u_p[0] / A_sec[0])) * self.am1_0_pd - (
                    1 / (self.r_sec * 2)) * np.square((i_sec[0] / V_sec[0])) * self.am1_0_pd,
            (1,)),
                              (1 / (2 * self.r_sec)) * np.square(i_sec[0:-1] / V_sec[0:-1]) * alpha - (
                                      1 / (2 * self.r_sec)) * np.square(i_sec[1:] / V_sec[1:]) * (
                                      np.ones(len(alpha)) - alpha)))  # check

        row1 = np.concatenate(
            (np.reshape([self.k_st[0] + self.k_c_st[0], -self.k_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.k_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.k_st[1:-1]) + np.diag(self.k_c_st[0:-2]) + np.diag(self.k_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.k_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.k_c_st[-2], self.k_st[-1] + self.k_c_st[-2]], (1, 2))),
            axis=1)

        Cspring = np.concatenate((row1, matrix, row2), axis=0)
        C_A_n = np.concatenate(
            (np.concatenate((np.diag(A_n[0:-1]), np.zeros((self.N_sec - 1, 1))), axis=1) + np.concatenate(
                (np.zeros((self.N_sec - 1, 1)), np.diag(A_n[1:])), axis=1),
             np.hstack((np.zeros(self.N_sec - 1), A_n[-1])).reshape(1, -1)), axis=0)
        H_q_st = np.dot(Cspring, q_st)  + A_c * (
                    1 / (2 * self.r_sec)) * np.square(i_sec / V_sec) + np.dot(C_A_n, p_d)

        H_i_st = i_st / self.m_st

        H_i_sec = i_sec / (self.r_sec * V_sec)

        log_A = np.array([self.b_st * np.log(r_n[i]/ self.r_sec) for i in range(0, self.N_sec, 1)])

        H_r_n = (self.m_n / (np.square(r_n))) * (log_A + p_d)

        C1 = np.identity(self.N_sec)
        C2 = np.concatenate(
            (np.zeros((1, self.N_sec)),
             np.concatenate((np.identity(self.N_sec - 1), np.zeros((self.N_sec - 1, 1))), axis=1)))
        C1_star = np.reshape(np.concatenate(([1], np.zeros(self.N_sec - 1))), (1, self.N_sec))
        C1_star_help = np.reshape(np.zeros(self.N_sec), (1, self.N_sec))
        C2_star = np.reshape(np.concatenate((np.zeros(self.N_sec - 1), [1])), (1, self.N_sec))

        row1 = np.concatenate(
            (np.reshape([self.d_st[0] + self.d_c_st[0], -self.d_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.d_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.d_st[1:-1]) + np.diag(self.d_c_st[0:-2]) + np.diag(self.d_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.d_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.d_c_st[-2], self.d_st[-1] + self.d_c_st[-2]], (1, 2))),
            axis=1)

        R_s = np.concatenate((row1, matrix, row2), axis=0)

        loss_g = np.zeros((self.N_sec))
        for k in range(0, self.N_sec - 1, 1):
            if A_sec[k] <= A_sec[k + 1]:
                angle = np.arctan((self.c_sec[k + 1] - self.c_sec[k]) / self.l)
                if math.degrees(angle) <= 45:
                    loss_g[k] = np.sin(angle/2) * (1 - (A_sec[k] / A_sec[k + 1])) ** 2
                else:
                    loss_g[k] = (1 - (A_sec[k] / A_sec[k + 1])) ** 2
            else:
                loss_g[k] = 0.5 * (1 - (A_sec[k + 1] / A_sec[k]))
        loss_g[-1] = 1
        loss_g = np.multiply(self.geo_dissipation_factor, loss_g)

        if self.geo_dissipation == False:
            loss_g = loss_g * 0

        loss_vis = np.array(
            [self.vf / (2*(self.c_sec[i] + q_st[i]) * np.abs(H_i_sec[i]) * self.r_sec) if H_i_sec[i] != 0 else 0 for i
             in range(0, self.N_sec, 1)])
        loss_vis = np.multiply(loss_vis, self.vis_dissipation_factor)

        if self.vis_dissipation == False:
            loss_vis = loss_vis * 0

        R_f_geo = np.diag(A_sec * loss_g * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f_vis = np.diag(A_sec * loss_vis * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f = R_f_geo + R_f_vis




        theta_pi = np.diag(A_sec)
        theta_rho = np.diag(np.square(r_n) / self.m_n)
        g_An = np.diag(A_n[1:])
        psi_pi = np.diag(A_c / 2)
        varphi_pi = np.diag(i_sec / (self.c_sec + q_st))

        g_An_1 = np.concatenate((g_An, np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1))), axis=1)
        g_An_2 = np.concatenate((np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1)), g_An), axis=1)

        varphi_rho = np.dot(theta_rho, np.concatenate(
            (np.hstack(([A_n[0]], np.zeros(self.N_sec - 1))).reshape(1, -1), g_An_1 + g_An_2)))
        phi = np.dot(theta_pi, np.dot(C1.T - C2.T, theta_rho.T))
        theta = np.dot(theta_pi, np.concatenate((C1_star_help.T, -C2_star.T), axis=1))
        varphi = varphi_rho + np.dot(theta_rho, np.dot((C2 + C1), psi_pi))
        psi = np.dot(np.reshape(np.concatenate((C1_star_help.T, C2_star.T)), (2, self.N_sec)), psi_pi)
        C = np.identity(self.N_sec)

        gamma = np.concatenate((np.dot(theta_rho.T, C1_star.T), np.zeros((self.N_sec, 1))), axis=1)

        J_R_1 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.identity(self.N_sec),
                                np.zeros((self.N_sec, self.N_sec)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        J_R_2 = np.concatenate((-np.identity(self.N_sec), -R_s, np.dot(C.T, varphi_pi.T), np.dot(C.T, varphi.T)),
                               axis=1)
        J_R_3 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi_pi, C), -R_f, phi), axis=1)
        J_R_4 = np.concatenate(
            (np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi, C), -phi.T, np.zeros((self.N_sec, self.N_sec))),
            axis=1)
        J_R = np.concatenate((J_R_1, J_R_2, J_R_3, J_R_4))

        g_P_1 = np.concatenate((np.zeros((self.N_sec, 2)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_2 = np.concatenate((np.dot(C.T, psi.T), np.identity(self.N_sec)), axis=1)
        g_P_3 = np.concatenate((theta, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_4 = np.concatenate((gamma, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P = np.concatenate((g_P_1, g_P_2, g_P_3, g_P_4))
        x_vec = np.concatenate((H_q_st, H_i_st, H_i_sec, H_r_n))
        u = np.concatenate((u_p, self.u_e))
        dxdt = np.dot(J_R, x_vec) + np.dot(g_P, u)
        return dxdt
    def PHModel(self, t, x):
        u_p = np.concatenate(([float(self.inp_ent(t))], [self.inp_ext(t)]))
        x = x.reshape(-1, )
        q_st = x[0:self.N_sec]
        i_st = x[self.N_sec:2 * self.N_sec]
        i_sec = x[2 * self.N_sec:3 * self.N_sec]
        r_n = x[3 * self.N_sec:]

        A_sec = np.multiply(np.pi * self.c_sec, (q_st + self.c_sec))
        A_cross_start = np.multiply(np.pi * (self.c_sec + q_st), (q_st + self.c_sec))

        radius_section_start = self.c_sec + q_st
        radius_section_end = np.hstack((self.c_sec[1:], self.last_radius))
        section_radius_complete = np.hstack((self.c_sec + q_st, self.last_radius + q_st[-1]))
        radius_node0_end, radius_node_start, radius_node_end, V_node0, V_node_left_part, V_node_right_part = getNodesVolumes(
            section_radius_complete, self.l, self.length_node)
        l_node = self.length_node
        r_node_end = radius_node_end
        r_node_start = radius_node_start
        r_node0_end = radius_node0_end
        V_node0_left = V_node0

        V_sec = 1 / 3 * np.pi * self.l * np.square(
            radius_section_start[1:-1] + radius_section_end[1:-1]) - V_node_left_part[1:] - V_node_right_part[0:-1]
        V_sec = np.concatenate((np.reshape(
            1 / 3 * np.pi * self.l * np.square(radius_section_start[0] + radius_section_end[0]) - V_node0_left -
            V_node_left_part[0], (1,)), V_sec, np.reshape(
            1 / 3 * np.pi * self.l * np.square(radius_section_start[-1] + radius_section_end[-1]) - V_node_right_part[
                -1],
            (1,))))  # check

        A_n_old = np.hstack((np.reshape(self.m_n[0] / (r_n[0] * (self.c_sec[0] + q_st[0])), (1,)),
                             self.m_n[1:] / (r_n[1:] * (2 * self.c_sec[1:] + q_st[0:-1] + q_st[1:]))))  # check

        A_n = (self.m_n[1:] * (3 / l_node) * np.sqrt(np.square(r_node_end - r_node_start) + np.square(l_node))) / (
                    r_n[1:] * (r_node_start + r_node_end))
        A_n0 = (self.m_n[0] * (3 * 2 / l_node) * np.sqrt(
            np.square(radius_section_start[0] - r_node0_end) + np.square(l_node / 2))) / (
                           r_n[0] * (radius_section_start[0] + r_node0_end))
        A_n = np.hstack((A_n0, A_n))

        A_circ = (radius_section_start +radius_section_end)*np.pi*np.sqrt(np.square(radius_section_end-radius_section_start) +np.square(self.l))

        alpha_2 = V_node_right_part[0:-1] / (V_node_right_part[0:-1] + V_node_left_part[1:])
        alpha_1 = V_node_left_part[1:] / (V_node_right_part[0:-1] + V_node_left_part[1:])
        A_c_old = np.hstack(
            (np.reshape(A_circ[0] - A_n[0], (1,)), A_circ[1:-1] - A_n[1:-1] * alpha_2 - A_n[2:] * alpha_1,
             np.reshape(A_circ[-1] - A_n[-1] * alpha_2[-1], (-1,))))  # check
        A_c = np.hstack((np.reshape(
            A_circ[0] - A_n[0] - A_n[1] * (V_node_left_part[0] / (V_node_right_part[0] + V_node_left_part[0])), (1,)),
                         A_circ[1:-1] - A_n[1:-1] * (
                                 V_node_right_part[0:-1] / (V_node_right_part[0:-1] + V_node_left_part[1:])) - A_n[
                                                                                                               2:] * (
                                 V_node_left_part[1:] / (V_node_right_part[0:-1] + V_node_left_part[1:])),
                         np.reshape(A_circ[-1] - A_n[-1] * (
                                 V_node_right_part[-1] / (V_node_right_part[-1] + V_node_left_part[-1])),
                                    (-1,))))  # check
        alpha = V_node_left_part / (V_node_right_part + V_node_left_part)

        ##hier ist ein Fehler##
        p_d = np.concatenate((np.reshape(
            (1 * self.r_sec / 2) * np.square((u_p[0] / A_cross_start[0])) * float(self.am1_0_pd) - (
                    1 / (self.r_sec * 2)) * np.square((i_sec[0] / V_sec[0])) * float(self.am1_0_pd),
            (1,)),
                              (1 / (2 * self.r_sec)) * np.square(i_sec[0:-1] / V_sec[0:-1]) * alpha - (
                                      1 / (2 * self.r_sec)) * np.square(i_sec[1:] / V_sec[1:]) * (
                                      np.ones(len(alpha)) - alpha)))  # check
        row1 = np.concatenate(
            (np.reshape([self.k_st[0] + self.k_c_st[0], -self.k_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.k_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.k_st[1:-1]) + np.diag(self.k_c_st[0:-2]) + np.diag(self.k_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.k_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.k_c_st[-2], self.k_st[-1] + self.k_c_st[-2]], (1, 2))),
            axis=1)

        Cspring = np.concatenate((row1, matrix, row2), axis=0)
        C_A_n = np.concatenate(
            (np.concatenate((np.diag(A_n[0:-1]), np.zeros((self.N_sec - 1, 1))), axis=1) + np.concatenate(
                (np.zeros((self.N_sec - 1, 1)), np.diag(A_n[1:])), axis=1),
             np.hstack((np.zeros(self.N_sec - 1), A_n[-1])).reshape(1, -1)), axis=0)
        H_q_st = np.dot(Cspring, q_st) + A_c * (
                1 / (2 * self.r_sec)) * np.square(i_sec / V_sec) + np.dot(C_A_n, p_d)

        H_i_st = i_st / self.m_st

        H_i_sec = i_sec / (self.r_sec * V_sec)

        log_A = np.array([self.b_st * np.log(r_n[i] / self.r_sec) for i in range(0, self.N_sec, 1)])

        H_r_n = (self.m_n / (np.square(r_n))) * (log_A + p_d)

        C1 = np.identity(self.N_sec)
        C2 = np.concatenate(
            (np.zeros((1, self.N_sec)),
             np.concatenate((np.identity(self.N_sec - 1), np.zeros((self.N_sec - 1, 1))), axis=1)))
        C1_star = np.reshape(np.concatenate(([1], np.zeros(self.N_sec - 1))), (1, self.N_sec))
        C1_star_help = np.reshape(np.zeros(self.N_sec), (1, self.N_sec))
        C2_star = np.reshape(np.concatenate((np.zeros(self.N_sec - 1), [1])), (1, self.N_sec))

        row1 = np.concatenate(
            (np.reshape([self.d_st[0] + self.d_c_st[0], -self.d_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.d_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.d_st[1:-1]) + np.diag(self.d_c_st[0:-2]) + np.diag(self.d_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.d_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.d_c_st[-2], self.d_st[-1] + self.d_c_st[-2]], (1, 2))),
            axis=1)

        R_s = np.concatenate((row1, matrix, row2), axis=0)

        loss_g = np.zeros((self.N_sec))
        for k in range(0, self.N_sec - 1, 1):
            if A_sec[k] <= A_sec[k + 1]:
                angle = np.arctan((self.c_sec[k + 1] - self.c_sec[k]) / self.l)
                if math.degrees(angle) <= 45:
                    loss_g[k] = np.sin(angle / 2) * (1 - (A_sec[k] / A_sec[k + 1])) ** 2
                else:
                    loss_g[k] = (1 - (A_sec[k] / A_sec[k + 1])) ** 2
            else:
                loss_g[k] = 0.5 * (1 - (A_sec[k + 1] / A_sec[k]))
        loss_g[-1] = 1
        loss_g = np.multiply(self.geo_dissipation_factor, loss_g)

        if self.geo_dissipation == False:
            loss_g = loss_g * 0

        loss_vis = np.array(
            [self.vf / (2 * (self.c_sec[i] + q_st[i]) * np.abs(H_i_sec[i]) * self.r_sec) if H_i_sec[i] != 0 else 0 for i
             in range(0, self.N_sec, 1)])
        loss_vis = np.multiply(loss_vis, self.vis_dissipation_factor)

        if self.vis_dissipation == False:
            loss_vis = loss_vis * 0

        R_f_geo = np.diag(A_sec * loss_g * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f_vis = np.diag(A_sec * loss_vis * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f = R_f_geo + R_f_vis

        theta_pi = np.diag(A_sec)
        theta_rho = np.diag(np.square(r_n) / self.m_n)
        g_An = np.diag(A_n[1:])
        psi_pi = np.diag(A_c / 2)
        varphi_pi = np.diag(i_sec / (self.c_sec + q_st))

        g_An_1 = np.concatenate((g_An, np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1))), axis=1)
        g_An_2 = np.concatenate((np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1)), g_An), axis=1)

        varphi_rho = np.dot(theta_rho, np.concatenate(
            (np.hstack(([A_n[0]], np.zeros(self.N_sec - 1))).reshape(1, -1), g_An_1 + g_An_2)))
        phi = np.dot(theta_pi, np.dot(C1.T - C2.T, theta_rho.T))
        theta = np.dot(theta_pi, np.concatenate((C1_star_help.T, -C2_star.T), axis=1))
        varphi = varphi_rho + np.dot(theta_rho, np.dot((C2 + C1), psi_pi))
        psi = np.dot(np.reshape(np.concatenate((C1_star_help.T, C2_star.T)), (2, self.N_sec)), psi_pi)
        C = np.identity(self.N_sec)

        gamma = np.concatenate((np.dot(theta_rho.T, C1_star.T), np.zeros((self.N_sec, 1))), axis=1)

        J_R_1 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.identity(self.N_sec),
                                np.zeros((self.N_sec, self.N_sec)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        J_R_2 = np.concatenate((-np.identity(self.N_sec), -R_s, np.dot(C.T, varphi_pi.T), np.dot(C.T, varphi.T)),
                               axis=1)
        J_R_3 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi_pi, C), -R_f, phi), axis=1)
        J_R_4 = np.concatenate(
            (np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi, C), -phi.T, np.zeros((self.N_sec, self.N_sec))),
            axis=1)
        J_R = np.concatenate((J_R_1, J_R_2, J_R_3, J_R_4))

        g_P_1 = np.concatenate((np.zeros((self.N_sec, 2)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_2 = np.concatenate((np.dot(C.T, psi.T), np.identity(self.N_sec)), axis=1)
        g_P_3 = np.concatenate((theta, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_4 = np.concatenate((gamma, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P = np.concatenate((g_P_1, g_P_2, g_P_3, g_P_4))
        x_vec = np.concatenate((H_q_st, H_i_st, H_i_sec, H_r_n))

        u = np.concatenate((u_p, self.u_e))
        dxdt = np.dot(J_R, x_vec) + np.dot(g_P, u)
        return dxdt
    def Jacobi(self, t, x):
        return 0

    def get_IntermediateResults_old(self, t, x, i):
        u_p = np.concatenate(([float(self.inp_ent(t))], [self.inp_ext(t)]))
        x = x.reshape(-1, )
        q_st = x[0:self.N_sec]
        i_st = x[self.N_sec:2 * self.N_sec]
        i_sec = x[2 * self.N_sec:3 * self.N_sec]
        r_n = x[3 * self.N_sec:]

        A_sec = np.multiply(np.pi * self.c_sec, (q_st + self.c_sec))

        alpha_1 = A_sec[1:-1] / (A_sec[1:-1] + A_sec[2:])  # check
        alpha_2 = 1 - (A_sec[0:-2] / (A_sec[0:-2] + A_sec[1:-1]))  # check

        V_sec = self.l * A_sec[1:-1] - (self.m_n[2:] / r_n[2:]) * alpha_1 - (
                self.m_n[1:-1] / r_n[1:-1]) * alpha_2  # check
        V_sec = np.concatenate((np.reshape(
            self.l * A_sec[0] - self.m_n[1] * (1 / r_n[1]) * (A_sec[0] / (A_sec[0] + A_sec[1])) - self.m_n[0] * (
                    1 / r_n[0]) * float(self.am1_0_vol), (1,)), V_sec, np.reshape(
            A_sec[-1] * self.l - self.m_n[-1] * (1 / r_n[-1]) * (1 - (A_sec[-2] / (A_sec[-2] + A_sec[-1]))),
            (1,))))  # check

        A_n = np.hstack((np.reshape(self.m_n[0] / (r_n[0] * (self.c_sec[0] + q_st[0])), (1,)),
                         self.m_n[1:] / (r_n[1:] * (2 * self.c_sec[1:] + q_st[0:-1] + q_st[1:]))))  # check

        A_c = V_sec / (q_st + self.c_sec)  # check

        alpha = A_sec[0:-1] / (A_sec[0:-1] + A_sec[1:])

        p_d = np.concatenate((np.reshape(
            (1 * self.r_sec / 2) * np.square((u_p[0] / A_sec[0])) * self.am1_0_pd - (
                    1 / (self.r_sec * 2)) * np.square((i_sec[0] / V_sec[0])) * self.am1_0_pd,
            (1,)),
                              (1 / (2 * self.r_sec)) * np.square(i_sec[0:-1] / V_sec[0:-1]) * alpha - (
                                      1 / (2 * self.r_sec)) * np.square(i_sec[1:] / V_sec[1:]) * (
                                      np.ones(len(alpha)) - alpha)))  # check

        row1 = np.concatenate(
            (np.reshape([self.k_st[0] + self.k_c_st[0], -self.k_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.k_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.k_st[1:-1]) + np.diag(self.k_c_st[0:-2]) + np.diag(self.k_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.k_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.k_c_st[-2], self.k_st[-1] + self.k_c_st[-2]], (1, 2))),
            axis=1)

        Cspring = np.concatenate((row1, matrix, row2), axis=0)
        C_A_n = np.concatenate(
            (np.concatenate((np.diag(A_n[0:-1]), np.zeros((self.N_sec - 1, 1))), axis=1) + np.concatenate(
                (np.zeros((self.N_sec - 1, 1)), np.diag(A_n[1:])), axis=1),
             np.hstack((np.zeros(self.N_sec - 1), A_n[-1])).reshape(1, -1)), axis=0)
        H_q_st = np.dot(Cspring, q_st) + A_c * (
                1 / (2 * self.r_sec)) * np.square(i_sec / V_sec) + np.dot(C_A_n, p_d)

        H_i_st = i_st / self.m_st

        H_i_sec = i_sec / (self.r_sec * V_sec)

        log_A = np.array([self.b_st * np.log(r_n[i] / self.r_sec) for i in range(0, self.N_sec, 1)])

        H_r_n = (self.m_n / (np.square(r_n))) * (log_A + p_d)

        C1 = np.identity(self.N_sec)
        C2 = np.concatenate(
            (np.zeros((1, self.N_sec)),
             np.concatenate((np.identity(self.N_sec - 1), np.zeros((self.N_sec - 1, 1))), axis=1)))
        C1_star = np.reshape(np.concatenate(([1], np.zeros(self.N_sec - 1))), (1, self.N_sec))
        C1_star_help = np.reshape(np.zeros(self.N_sec), (1, self.N_sec))
        C2_star = np.reshape(np.concatenate((np.zeros(self.N_sec - 1), [1])), (1, self.N_sec))

        row1 = np.concatenate(
            (np.reshape([self.d_st[0] + self.d_c_st[0], -self.d_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.d_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.d_st[1:-1]) + np.diag(self.d_c_st[0:-2]) + np.diag(self.d_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.d_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.d_c_st[-2], self.d_st[-1] + self.d_c_st[-2]], (1, 2))),
            axis=1)

        R_s = np.concatenate((row1, matrix, row2), axis=0)

        loss_g = np.zeros((self.N_sec))
        for k in range(0, self.N_sec - 1, 1):
            if A_sec[k] <= A_sec[k + 1]:
                angle = np.arctan((self.c_sec[k + 1] - self.c_sec[k]) / self.l)
                if math.degrees(angle) <= 45:
                    loss_g[k] = np.sin(angle / 2) * (1 - (A_sec[k] / A_sec[k + 1])) ** 2
                else:
                    loss_g[k] = (1 - (A_sec[k] / A_sec[k + 1])) ** 2
            else:
                loss_g[k] = 0.5 * (1 - (A_sec[k + 1] / A_sec[k]))
        loss_g[-1] = 1
        loss_g = np.multiply(self.geo_dissipation_factor, loss_g)

        if self.geo_dissipation == False:
            loss_g = loss_g * 0

        loss_vis = np.array(
            [self.vf / (2 * (self.c_sec[i] + q_st[i]) * np.abs(H_i_sec[i]) * self.r_sec) if H_i_sec[i] != 0 else 0 for i
             in range(0, self.N_sec, 1)])
        loss_vis = np.multiply(loss_vis, self.vis_dissipation_factor)

        if self.vis_dissipation == False:
            loss_vis = loss_vis * 0

        R_f_geo = np.diag(A_sec * loss_g * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f_vis = np.diag(A_sec * loss_vis * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f = R_f_geo + R_f_vis

        print("angle: ", math.degrees(np.arctan((self.c_sec[1] - self.c_sec[0]) / self.l)))
        print("geometrical loss: ", loss_g[0])
        print("viscous loss: ", loss_vis[0], " ", loss_vis[int(self.N_sec/2)], " ", loss_vis[-1])
        print("geometrical total loss: ", R_f_geo[0,0])
        print("viscous total loss: ", R_f_vis[int(self.N_sec/2), int(self.N_sec/2)])
        print("total loss: ", R_f[int(self.N_sec/2), int(self.N_sec/2)])

        theta_pi = np.diag(A_sec)
        theta_rho = np.diag(np.square(r_n) / self.m_n)
        g_An = np.diag(A_n[1:])
        psi_pi = np.diag(A_c / 2)
        varphi_pi = np.diag(i_sec / (self.c_sec + q_st))

        g_An_1 = np.concatenate((g_An, np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1))), axis=1)
        g_An_2 = np.concatenate((np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1)), g_An), axis=1)

        varphi_rho = np.dot(theta_rho, np.concatenate(
            (np.hstack(([A_n[0]], np.zeros(self.N_sec - 1))).reshape(1, -1), g_An_1 + g_An_2)))
        phi = np.dot(theta_pi, np.dot(C1.T - C2.T, theta_rho.T))
        theta = np.dot(theta_pi, np.concatenate((C1_star_help.T, -C2_star.T), axis=1))
        varphi = varphi_rho + np.dot(theta_rho, np.dot((C2 + C1), psi_pi))
        psi = np.dot(np.reshape(np.concatenate((C1_star_help.T, C2_star.T)), (2, self.N_sec)), psi_pi)
        C = np.identity(self.N_sec)

        gamma = np.concatenate((np.dot(theta_rho.T, C1_star.T), np.zeros((self.N_sec, 1))), axis=1)

        J_R_1 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.identity(self.N_sec),
                                np.zeros((self.N_sec, self.N_sec)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        J_R_2 = np.concatenate((-np.identity(self.N_sec), -R_s, np.dot(C.T, varphi_pi.T), np.dot(C.T, varphi.T)),
                               axis=1)
        J_R_3 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi_pi, C), -R_f, phi), axis=1)
        J_R_4 = np.concatenate(
            (np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi, C), -phi.T, np.zeros((self.N_sec, self.N_sec))),
            axis=1)
        J_R = np.concatenate((J_R_1, J_R_2, J_R_3, J_R_4))

        g_P_1 = np.concatenate((np.zeros((self.N_sec, 2)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_2 = np.concatenate((np.dot(C.T, psi.T), np.identity(self.N_sec)), axis=1)
        g_P_3 = np.concatenate((theta, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_4 = np.concatenate((gamma, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P = np.concatenate((g_P_1, g_P_2, g_P_3, g_P_4))
        x_vec = np.concatenate((H_q_st, H_i_st, H_i_sec, H_r_n))
        u = np.concatenate((u_p, self.u_e))
        dxdt = np.dot(J_R, x_vec) + np.dot(g_P, u)

        y = np.dot(g_P.T, x_vec)



        self.A_sec[:,i] = A_sec

        self.V_sec[:,i] = V_sec
        self.A_n[:,i] = A_n
        self.A_c[:,i] = A_c

        self.H_q_st[:,i] = H_q_st
        self.H_i_st[:,i] = H_i_st
        self.H_i_sec[:,i] = H_i_sec
        self.H_r_n[:,i] = H_r_n


        self.stat_pressure[:, i] = log_A
        self.stat_pressure_mmHg[:, i] = log_A/self.mmHgFactor
        self.dyn_pressure[:,i] = p_d
        self.dyn_pressure_mmHg[:, i] = p_d/self.mmHgFactor
        self.total_pressure[:,i] = log_A + p_d
        self.total_pressure_mmHg[:, i] = (log_A/self.mmHgFactor)+(p_d/self.mmHgFactor)
        self.inp_val[i, 0] = float(self.inp_ent(t))
        self.inp_val[i, 1] = float(self.inp_ext(t))
        self.out_val[i, 0] = y[0]
        self.out_val[i, 1] = y[1]


        H_st = 0
        for j in range(0, self.N_sec - 1):
            H_st = H_st + 0.5 * (i_st[j] ** 2 / self.m_st[j]) + 0.5 * self.k_st[j] * q_st[j] ** 2 + 0.5 * self.k_c_st[
                j] * (q_st[j] - q_st[j + 1]) ** 2
        H_st = H_st + 0.5 * (i_st[self.N_sec - 1] ** 2 / self.m_st[self.N_sec - 1]) + 0.5 * self.k_st[self.N_sec - 1] * \
               q_st[self.N_sec - 1] ** 2
        self.H_st[i] = H_st

        H_fl = 0
        for j in range(0, self.N_sec):
            H_fl = H_fl + 0.5 * (i_sec[j] ** 2 / (self.r_sec * V_sec[j]))
        self.H_fl[i] = H_fl

        E_fl = 0
        for j in range(0, self.N_nodes):
            E_fl = E_fl + self.m_n[j] * self.b_st * (
                        (r_n[j] - self.r_sec * (1 + np.log(r_n[j] / self.r_sec))) / (r_n[j] * self.r_sec))
        self.E_fl[i] = E_fl

        return (t,float(self.inp_ent(t)))
    def get_IntermediateResults(self, t, x, i):
        u_p = np.concatenate(([float(self.inp_ent(t))], [self.inp_ext(t)]))
        x = x.reshape(-1, )
        q_st = x[0:self.N_sec]
        i_st = x[self.N_sec:2 * self.N_sec]
        i_sec = x[2 * self.N_sec:3 * self.N_sec]
        r_n = x[3 * self.N_sec:]

        A_sec = np.multiply(np.pi * self.c_sec, (q_st + self.c_sec))
        A_cross_start = np.multiply(np.pi * (self.c_sec + q_st), (q_st + self.c_sec))

        radius_section_start = self.c_sec + q_st
        radius_section_end = np.hstack((self.c_sec[1:], self.last_radius))
        section_radius_complete = np.hstack((self.c_sec + q_st, self.last_radius + q_st[-1]))
        radius_node0_end, radius_node_start, radius_node_end, V_node0, V_node_left_part, V_node_right_part = getNodesVolumes(
            section_radius_complete, self.l, self.length_node)
        l_node = self.length_node
        r_node_end = radius_node_end
        r_node_start = radius_node_start
        r_node0_end = radius_node0_end
        V_node0_left = V_node0
        volume_node_complete = np.hstack((V_node0, V_node_left_part+V_node_right_part))
        V_sec = 1 / 3 * np.pi * self.l * np.square(
            radius_section_start[1:-1] + radius_section_end[1:-1]) - V_node_left_part[1:] - V_node_right_part[0:-1]
        V_sec = np.concatenate((np.reshape(
            1 / 3 * np.pi * self.l * np.square(radius_section_start[0] + radius_section_end[0]) - V_node0_left -
            V_node_left_part[0], (1,)), V_sec, np.reshape(
            1 / 3 * np.pi * self.l * np.square(radius_section_start[-1] + radius_section_end[-1]) - V_node_right_part[
                -1],
            (1,))))  # check


        A_n_old = np.hstack((np.reshape(self.m_n[0] / (r_n[0] * (self.c_sec[0] + q_st[0])), (1,)),
                             self.m_n[1:] / (r_n[1:] * (2 * self.c_sec[1:] + q_st[0:-1] + q_st[1:]))))  # check

        A_n = (self.m_n[1:]*(3/l_node)*np.sqrt(np.square(r_node_end-r_node_start)+np.square(l_node)))/(r_n[1:]*(r_node_start + r_node_end))
        A_n0 = (self.m_n[0]*(3*2/l_node)*np.sqrt(np.square(radius_section_start[0]-r_node0_end)+np.square(l_node/2)))/(r_n[0]*(radius_section_start[0] + r_node0_end))
        A_n = np.hstack((A_n0, A_n))


        A_circ = (radius_section_start +radius_section_end)*np.pi*np.sqrt(np.square(radius_section_end-radius_section_start) +np.square(self.l))

        alpha_2 = V_node_right_part[0:-1] / (V_node_right_part[0:-1] + V_node_left_part[1:])
        alpha_1 = V_node_left_part[1:] / (V_node_right_part[0:-1] + V_node_left_part[1:])
        A_c_old = np.hstack((np.reshape(A_circ[0] - A_n[0], (1,)), A_circ[1:-1] - A_n[1:-1] * alpha_2 - A_n[2:] * alpha_1,
                         np.reshape(A_circ[-1] - A_n[-1] * alpha_2[-1], (-1,))))  # check
        A_c = np.hstack((np.reshape(
            A_cross_start[0] - A_n[0] - A_n[1] * (V_node_left_part[0] / (V_node_right_part[0] + V_node_left_part[0])), (1,)),
                         A_circ[1:-1] - A_n[1:-1] * (
                                     V_node_right_part[0:-1] / (V_node_right_part[0:-1] + V_node_left_part[1:])) - A_n[
                                                                                                                   2:] * (
                                     V_node_left_part[1:] / (V_node_right_part[0:-1] + V_node_left_part[1:])),
                         np.reshape(A_circ[-1] - A_n[-1] * (
                                     V_node_right_part[-1] / (V_node_right_part[-1] + V_node_left_part[-1])),
                                    (-1,))))  # check
        A_circ = A_c_old
        alpha = V_node_left_part / (V_node_right_part + V_node_left_part)

        ##hier ist ein Fehler##
        p_d = np.concatenate((np.reshape(
            (1 * self.r_sec / 2) * np.square((u_p[0] / A_cross_start[0])) * float(self.am1_0_pd) - (
                    1 / (self.r_sec * 2)) * np.square((i_sec[0] / V_sec[0])) * float(self.am1_0_pd),
            (1,)),
                              (1 / (2 * self.r_sec)) * np.square(i_sec[0:-1] / V_sec[0:-1]) * alpha - (
                                      1 / (2 * self.r_sec)) * np.square(i_sec[1:] / V_sec[1:]) * (
                                      np.ones(len(alpha)) - alpha)))  # check
        row1 = np.concatenate(
            (np.reshape([self.k_st[0] + self.k_c_st[0], -self.k_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.k_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.k_st[1:-1]) + np.diag(self.k_c_st[0:-2]) + np.diag(self.k_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.k_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.k_c_st[-2], self.k_st[-1] + self.k_c_st[-2]], (1, 2))),
            axis=1)

        Cspring = np.concatenate((row1, matrix, row2), axis=0)
        C_A_n = np.concatenate(
            (np.concatenate((np.diag(A_n[0:-1]), np.zeros((self.N_sec - 1, 1))), axis=1) + np.concatenate(
                (np.zeros((self.N_sec - 1, 1)), np.diag(A_n[1:])), axis=1),
             np.hstack((np.zeros(self.N_sec - 1), A_n[-1])).reshape(1, -1)), axis=0)
        H_q_st = np.dot(Cspring, q_st) + A_c * (
                1 / (2 * self.r_sec)) * np.square(i_sec / V_sec) + np.dot(C_A_n, p_d)

        H_i_st = i_st / self.m_st

        H_i_sec = i_sec / (self.r_sec * V_sec)

        log_A = np.array([self.b_st * np.log(r_n[i] / self.r_sec) for i in range(0, self.N_sec, 1)])

        H_r_n = (self.m_n / (np.square(r_n))) * (log_A + p_d)

        C1 = np.identity(self.N_sec)
        C2 = np.concatenate(
            (np.zeros((1, self.N_sec)),
             np.concatenate((np.identity(self.N_sec - 1), np.zeros((self.N_sec - 1, 1))), axis=1)))
        C1_star = np.reshape(np.concatenate(([1], np.zeros(self.N_sec - 1))), (1, self.N_sec))
        C1_star_help = np.reshape(np.zeros(self.N_sec), (1, self.N_sec))
        C2_star = np.reshape(np.concatenate((np.zeros(self.N_sec - 1), [1])), (1, self.N_sec))

        row1 = np.concatenate(
            (np.reshape([self.d_st[0] + self.d_c_st[0], -self.d_c_st[0]], (1, 2)), np.zeros((1, self.N_sec - 2))),
            axis=1)
        matrix = np.concatenate((-np.diag(self.d_c_st[0:-2]), np.zeros((self.N_sec - 2, 2))), axis=1) + np.concatenate(
            (np.zeros(
                (self.N_sec - 2, 1)),
             np.diag(self.d_st[1:-1]) + np.diag(self.d_c_st[0:-2]) + np.diag(self.d_c_st[1:-1]), np.zeros(
                (self.N_sec - 2, 1))), axis=1) + np.concatenate(
            (np.zeros((self.N_sec - 2, 2)), -np.diag(self.d_c_st[1:-1])), axis=1)
        row2 = np.concatenate(
            (np.zeros((1, self.N_sec - 2)), np.reshape([-self.d_c_st[-2], self.d_st[-1] + self.d_c_st[-2]], (1, 2))),
            axis=1)

        R_s = np.concatenate((row1, matrix, row2), axis=0)

        loss_g = np.zeros((self.N_sec))
        for k in range(0, self.N_sec - 1, 1):
            if A_sec[k] <= A_sec[k + 1]:
                angle = np.arctan((self.c_sec[k + 1] - self.c_sec[k]) / self.l)
                if math.degrees(angle) <= 45:
                    loss_g[k] = np.sin(angle / 2) * (1 - (A_sec[k] / A_sec[k + 1])) ** 2
                else:
                    loss_g[k] = (1 - (A_sec[k] / A_sec[k + 1])) ** 2
            else:
                loss_g[k] = 0.5 * (1 - (A_sec[k + 1] / A_sec[k]))
        loss_g[-1] = 1
        loss_g = np.multiply(self.geo_dissipation_factor, loss_g)

        if self.geo_dissipation == False:
            loss_g = loss_g * 0

        loss_vis = np.array(
            [self.vf / (2 * (self.c_sec[i] + q_st[i]) * np.abs(H_i_sec[i]) * self.r_sec) if H_i_sec[i] != 0 else 0 for i
             in range(0, self.N_sec, 1)])
        loss_vis = np.multiply(loss_vis, self.vis_dissipation_factor)

        if self.vis_dissipation == False:
            loss_vis = loss_vis * 0

        R_f_geo = np.diag(A_sec * loss_g * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f_vis = np.diag(A_sec * loss_vis * (1 * self.r_sec / 2) * (np.abs(i_sec / (self.r_sec * V_sec))))
        R_f = R_f_geo + R_f_vis

        theta_pi = np.diag(A_sec)
        theta_rho = np.diag(np.square(r_n) / self.m_n)
        g_An = np.diag(A_n[1:])
        psi_pi = np.diag(A_c / 2)
        varphi_pi = np.diag(i_sec / (self.c_sec + q_st))

        g_An_1 = np.concatenate((g_An, np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1))), axis=1)
        g_An_2 = np.concatenate((np.reshape(np.zeros(self.N_sec - 1), (self.N_sec - 1, 1)), g_An), axis=1)

        varphi_rho = np.dot(theta_rho, np.concatenate(
            (np.hstack(([A_n[0]], np.zeros(self.N_sec - 1))).reshape(1, -1), g_An_1 + g_An_2)))
        phi = np.dot(theta_pi, np.dot(C1.T - C2.T, theta_rho.T))
        theta = np.dot(theta_pi, np.concatenate((C1_star_help.T, -C2_star.T), axis=1))
        varphi = varphi_rho + np.dot(theta_rho, np.dot((C2 + C1), psi_pi))
        psi = np.dot(np.reshape(np.concatenate((C1_star_help.T, C2_star.T)), (2, self.N_sec)), psi_pi)
        C = np.identity(self.N_sec)

        gamma = np.concatenate((np.dot(theta_rho.T, C1_star.T), np.zeros((self.N_sec, 1))), axis=1)

        J_R_1 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.identity(self.N_sec),
                                np.zeros((self.N_sec, self.N_sec)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        J_R_2 = np.concatenate((-np.identity(self.N_sec), -R_s, np.dot(C.T, varphi_pi.T), np.dot(C.T, varphi.T)),
                               axis=1)
        J_R_3 = np.concatenate((np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi_pi, C), -R_f, phi), axis=1)
        J_R_4 = np.concatenate(
            (np.zeros((self.N_sec, self.N_sec)), np.dot(-varphi, C), -phi.T, np.zeros((self.N_sec, self.N_sec))),
            axis=1)
        J_R = np.concatenate((J_R_1, J_R_2, J_R_3, J_R_4))

        g_P_1 = np.concatenate((np.zeros((self.N_sec, 2)), np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_2 = np.concatenate((np.dot(C.T, psi.T), np.identity(self.N_sec)), axis=1)
        g_P_3 = np.concatenate((theta, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P_4 = np.concatenate((gamma, np.zeros((self.N_sec, self.N_sec))), axis=1)
        g_P = np.concatenate((g_P_1, g_P_2, g_P_3, g_P_4))
        x_vec = np.concatenate((H_q_st, H_i_st, H_i_sec, H_r_n))
        u = np.concatenate((u_p, self.u_e))
        dxdt = np.dot(J_R, x_vec) + np.dot(g_P, u)
        y = np.dot(g_P.T, x_vec)

        self.A_sec[:, i] = A_sec

        self.V_sec[:, i] = V_sec
        self.A_n[:, i] = A_n
        self.A_c[:, i] = A_c
        self.A_circ[:, i] = A_circ

        self.H_q_st[:, i] = H_q_st
        self.H_i_st[:, i] = H_i_st
        self.H_i_sec[:, i] = H_i_sec
        self.H_r_n[:, i] = H_r_n

        self.stat_pressure[:, i] = log_A
        self.stat_pressure_mmHg[:, i] = log_A / self.mmHgFactor
        self.r_n_save[:, i] = r_n
        self.section_mom_save[:, i] = i_sec
        self.wall_mom_save[:, i] = i_st
        self.mass_node_save[:, i] = self.m_n
        self.radius_sec_save[:, i] = q_st
        self.Volume_node_save[:, i] = volume_node_complete
        self.dyn_pressure[:, i] = p_d
        self.dyn_pressure_mmHg[:, i] = p_d / self.mmHgFactor
        self.total_pressure[:, i] = log_A + p_d
        self.total_pressure_mmHg[:, i] = (log_A / self.mmHgFactor) + (p_d / self.mmHgFactor)
        print('der Fluss ist:', float(self.inp_ent(t)))
        self.inp_val[i, 0] = float(self.inp_ent(t))
        self.inp_val[i, 1] = float(self.inp_ext(t))
        self.out_val[i, 0] = y[0]
        self.out_val[i, 1] = y[1]

        H_st = 0
        for j in range(0, self.N_sec - 1):
            H_st = H_st + 0.5 * (i_st[j] ** 2 / self.m_st[j]) + 0.5 * self.k_st[j] * q_st[j] ** 2 + 0.5 * self.k_c_st[
                j] * (q_st[j] - q_st[j + 1]) ** 2
        H_st = H_st + 0.5 * (i_st[self.N_sec - 1] ** 2 / self.m_st[self.N_sec - 1]) + 0.5 * self.k_st[self.N_sec - 1] * \
               q_st[self.N_sec - 1] ** 2
        self.H_st[i] = H_st

        H_fl = 0
        for j in range(0, self.N_sec):
            H_fl = H_fl + 0.5 * (i_sec[j] ** 2 / (self.r_sec * V_sec[j]))
        self.H_fl[i] = H_fl

        E_fl = 0
        for j in range(0, self.N_nodes):
            E_fl = E_fl + self.m_n[j] * self.b_st * (
                    (r_n[j] - self.r_sec * (1 + np.log(r_n[j] / self.r_sec))) / (r_n[j] * self.r_sec))
        self.E_fl[i] = E_fl

        return (t, float(self.inp_ent(t)))


