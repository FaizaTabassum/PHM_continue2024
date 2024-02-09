# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:17:24 2021
@author: Faiza
"""
import copy

import numpy
import numpy as np
from numpy import save
import functions as func
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from profilehooks import profile
import matplotlib.pyplot as plt
import pandas as pd



class constraints:
    def __init__(self, path_load, path_save, cycles, tsample, tsim, tinp, inp_max, out_max, compare):
        self.path_load = path_load
        self.path_save = path_save
        self.tsample = tsample
        self.tsim_ = tsim
        self.tinp = tinp
        self.inp_max = inp_max
        self.out_max = out_max
        self.compare = compare

    @property
    def inp_const(self):
        inp_max = self.inp_max
        return lambda t: (inp_max/2)*(1-np.cos(np.pi*t/self.tinp)) if t < self.tinp else inp_max

    @property
    def inp_data(self):
        Q_input = np.load(self.path_load + "/Q_i_p.npy", allow_pickle=True)[()]
        Q_i = interp1d(Q_input[:,0], Q_input[:,1])
        upper_bound = np.max(Q_input[:,0])
        def f(x):
            if x <= upper_bound:
                return Q_i(x)
            return 0
        return f

    @property
    def out_const(self):
        out = lambda t: self.out_max
        return out

    @property
    def ts(self):
        return self.tsample

    @property
    def tsim(self):
        return self.tsim_

    @property
    def samples(self):
        return int(self.tsim / self.tsample)

    def save(self, sol, add):
        save(self.path_save + str(add), sol)

    def niter(self, sol):
        steps = np.array(sol.steps[1:]) - np.array(sol.steps[0:-1])
        print('mean-std: ', numpy.mean(steps) - numpy.std(steps))
        print('mean: ', numpy.mean(steps))
        print('mean+std: ', numpy.mean(steps) + numpy.std(steps))

@profile
def main():
    N_sec = 33
    N_nodes = 33
    time_plot = 0.5
    sample_time = 1e-03
    path_save = r'C:\Users\Faiza\Desktop\PH_simvascular\1D_porthamiltonian_FSI\analytical_solution'
    path_load = r''
    inp_max = 125*10**-6 #max flow input at inlet
    out_max = 0 #max pressure at outlet
    tinp = 3*10**-3

    cons = constraints(path_load,path_save, 1, sample_time, time_plot, tinp, inp_max, out_max, False)

    t_ev = np.ndarray.tolist(np.linspace(0, time_plot, int(cons.samples)))

    parameter = {
        'stenosis': False,
        'expansion': True,
        'grad': 120,
        'FEM': True,
        'geo_dissipation': True,
        'geo_factor': 0,
        'vis_factor': 16,
        'vis_dissipation': True,
        'inp': cons.inp_const,
        'out': cons.out_const,
        'L': 5 * 10 ** -2,
        'c': 1* 10 ** -2,
        'h': 3 * 10 ** -4,

        'N_sec': N_sec,
        'N_nodes': N_nodes,
        't_ev': t_ev,
        'samples': cons.samples,

        'beta_1': 1.1,
        'beta_2': 8.5 * 10 ** (-5),
        'lame1': 1.7 * 10 ** 6,
        'lame2': 5.75 * 10 ** 5,
        'rho_f': 1.06 * 10 ** 3,
        'vf': 0.004,
        'rho_st': 1.1 * 10 ** 3,
        'poi_rat': 0.4,
        'E_mod': 3 * 10 ** 5,
        'd_st':0,
        'd_c_st': 0,
        'b_st': 2.15 * 10 ** 9,
        'u_e': np.zeros(N_sec),
        'k1': 1*10**10,
        'k2': -20,
        'k3': 1*10**9,
        'mmHgFactor': 133.322368421,
        'stiff_wall': True
    }



    PHFSI = func.Model_Flow_Efficient(parameter) #this is for flow, if you want to test Pressure, you need to substitute Flow by Pressure
    T = [0, time_plot]
    x_init = np.concatenate((np.zeros(3*N_sec), parameter['rho_f']*np.ones(N_nodes))).reshape(-1,)
    sol = solve_ivp(
        fun=lambda t, x0: PHFSI.PHModel(t, x0),
        obj=PHFSI,
        t_span=T,
        t_eval=t_ev,
        min_step=sample_time,
        max_step=sample_time,
        y0=x_init,
        first_step=None,
        hmax=sample_time,
        hmin=sample_time,
        rtol=10 ** (-3),
        atol=10 ** (-6),
        dense_output=False,
        method='BDF',
        vectorized=False
    )
    print(PHFSI.stat_pressure[0,-1])
    print(PHFSI.stat_pressure[1, -1])
    plt.figure()
    plt.title('stat pressure')
    plt.plot(PHFSI.stat_pressure[:,-1])
    plt.figure()
    plt.title('Fluss')
    plt.plot(PHFSI.inp_val[0, :])
    # plt.figure()
    # plt.title('A_n')
    # plt.plot(PHFSI.A_n)
    plt.figure()
    plt.title('density node')
    plt.plot(PHFSI.r_n_save[:,-1])
    plt.figure()
    plt.title('structure velocity ')
    plt.plot(PHFSI.H_i_st[:, -1])
    # plt.figure()
    # plt.title('fluid momentum')
    # plt.plot(PHFSI.section_mom_save[:, -1])
    # plt.figure()
    # plt.title('mass node')
    # plt.plot(PHFSI.mass_node_save[:, -1])
    # plt.figure()
    # plt.title('volume node')
    # plt.plot(PHFSI.Volume_node_save[:, -1])
    # plt.figure()
    # plt.title('radius section')
    # plt.plot(PHFSI.radius_sec_save[:, -1])
    plt.show()
    stop = True


if __name__ == '__main__':
    main()