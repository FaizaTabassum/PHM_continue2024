
import numpy as np
import matplotlib.pyplot as plt


def interpolate_to_n_points(array, line, n_points):
    import scipy.interpolate
    interpolant = scipy.interpolate.interp1d(line, array)
    return interpolant(np.linspace(0, 1, n_points))



def load_p(dir):
    import os.path
    p = np.loadtxt(dir+'_p.csv', delimiter=",", skiprows=1)[1:, 0]
    return p / 10

def load_p_vel(dir):
    import os.path
    vz = np.loadtxt(dir+'_vel.csv', delimiter=",")[:]
    p = np.loadtxt(dir+'_p.csv', delimiter=",", skiprows=1)[1:, 0]
    line = np.loadtxt(dir + '_path.csv', delimiter=",")[:]

    # p cgs -> p mks (Pa): divide by 10
    # vz axial cgs -> v avg mks: divide by 100 (cgs -> mks), divide by 2 (vmax w/ parabolic flow profile -> vavg)
    return p / 10, vz / (2 * 100), line


def get_p_vel(dir, n_points):
    """Gets the pressure and velocity that are comparable to PH simulation results"""
    p, vel, line = load_p_vel(dir)
    return p, vel, line
def get_p(dir, n_points):
    """Gets the pressure and velocity that are comparable to PH simulation results"""
    p= load_p(dir)
    return p


def get_p_vel_PH(dir, timepoint):
    arr = np.load(dir, allow_pickle=True)[()]
    return arr[:, timepoint], arr[:, timepoint]


def compare_diff_p_ph_sv(description, path_ph, path_sv, n_sections):
    p_sv, vel_sv = get_p_vel(path_sv, n_sections)
    p_ph, vel_ph= get_p_vel_PH(path_ph, 499)

    plt.plot(np.diff(p_sv), label= "diffsv "+description)
    plt.plot(np.diff(p_ph), label="diffph " + description)


def compare_p_ph_sv(description, path_ph, path_sv, n_sections, factor, color, l):
    #zaxis, p_sv, vel_sv = get_p_vel(path_sv, n_sections)
    p_sv= get_p(path_sv, n_sections)
    p_ph, vel_ph= get_p_vel_PH(path_ph, 499)
    p_ph = np.append(p_ph, 0)
    if description=='angle = 0°':
        plt.plot(np.linspace(0, l, len(p_sv)), factor*p_sv, color = color,linestyle = 'dashed', label="sv, " + description)
        plt.plot(np.linspace(0, l, n_sections+1), factor*p_ph, color = color, label="ph, " + description)
    else:
        plt.plot(np.linspace(0, l, len(p_sv)), -factor * p_sv, color=color, linestyle='dashed',label="sv, " + description)
        plt.plot(np.linspace(0, l, n_sections + 1), factor * p_ph, color=color, label="ph, " + description)

def compare_p_quotient_ph_sv(description, path_ph, path_sv, n_sections):
    p_sv, vel_sv = get_p_vel(path_sv, n_sections)
    p_ph, vel_ph= get_p_vel_PH(path_ph, 499)
    plt.plot(p_sv/p_ph, label="psv/pph " + description)


def compare_vel_ph_sv(description, path_ph, path_sv, n_sections):
    p_sv, vel_sv = get_p_vel(path_sv, n_sections)
    p_ph, vel_ph, A = get_p_vel_PH(path_ph, 499)
    plt.title("vel " + description)
    plt.plot(vel_sv, label="sv")
    plt.plot(vel_ph, label="ph")

N = [31, 61, 91]


R = [1, 0.002, 0.004]
L = [5, 0.04, 0.05]
G = [0]
Q = [125]
title = ['angle = 2°', 'angle = 4°', 'angle = 6°', 'angle = 8°', 'angle = 10°']
title = ['0.0001mL', '0.001mL', '0.01mL', '0.1mL']
ph_results_path=r'C:\Users\Faiza\Desktop\PHM_continue2024\results'
sv_results_path =r'C:\Users\Faiza\Desktop\PHM_continue2024\results'


case = 'grad'
R_constant = 1
G_constant = 0
N_constant = 31
N_constant_sv = 31
L_constant = 5
Q_constant = 125
entities = 1
color = ["black", "blue", "green", "red", "orange", "gray"]

factor =10
r = 'r'

for case in ['grad']:
    plt.figure()
    for i in range(entities):
        if case == 'grad':
            ph_tocompare_path = f'{ph_results_path}\{r}{str(R_constant)}l{str(L_constant)}g{str(G[i])}f{str(Q_constant)}.npy'
            # sv_tocompare_path = f'{sv_results_path}\{r}{str(R_constant)}l{str(L_constant)}g{str(G[i])}f{str(Q_constant)}'
            sv_tocompare_path = f'{sv_results_path}\straight_20cmlong'
            compare_p_ph_sv(title[i], ph_tocompare_path, sv_tocompare_path, N_constant, factor, color[i], L_constant)
    plt.legend()
    if factor== 0.0075006156130264:
        plt.ylabel('pressure [mmHg]')
    else:
        plt.ylabel('pressure [Ba]')
    plt.xlabel('axial tube position [cm]')
    plt.show()


# def to_hex(t):
#     return '#' + ''.join(['{:02x}'.format(int(255*x)) for x in t[:3]])
#
# GEOMETRY_STEN = "stenosis_grad"
# GEOMETRY_straight = "straight_rad"
# GEOMETRY_stenosis_rad = "stenosis_rad"
#
# n_sections = 31
# p_sv_50_sten, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_STEN][0], n_sections)
# p_ph_50_sten, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_STEN][0], 499)
# p_sv_40_sten, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_STEN][1], n_sections)
# p_ph_40_sten, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_STEN][1], 499)
# p_sv_30_sten, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_STEN][2], n_sections)
# p_ph_30_sten, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_STEN][2], 499)
#
#
# GEOMETRY_straight = "straight_rad"
# S1 = "100"
# n_sections = 31
# p_sv_50_stra, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_straight][0], n_sections)
# p_ph_50_stra, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_straight][0], 499)
# p_sv_40_stra, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_straight][1], n_sections)
# p_ph_40_stra, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_straight][1], 499)
# p_sv_30_stra, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_straight][2], n_sections)
# p_ph_30_stra, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_straight][2], 499)
#
#
# p_sv_50_sten_rad, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_stenosis_rad][0], n_sections)
# p_ph_50_sten_rad, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_stenosis_rad][0], 499)
# p_sv_40_sten_rad, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_stenosis_rad][1], n_sections)
# p_ph_40_sten_rad, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_stenosis_rad][1], 499)
# p_sv_30_sten_rad, vel_sv = get_p_vel(sv_results_paths[GEOMETRY_stenosis_rad][2], n_sections)
# p_ph_30_sten_rad, vel_ph, A = get_p_vel_PH(ph_results_paths[GEOMETRY_stenosis_rad][2], 499)
#
#
#
# cmap1 = cm.get_cmap('Oranges', 10)
# cmap2 = cm.get_cmap('Blues', 10)
# cmap3 = cm.get_cmap('Greens', 10)
# sol = np.load(r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\analysis\analytical_solution'+"/solQ.npy", allow_pickle=True)[()]
# fig, axs = plt.subplots(2, 2, figsize = (5.8, 5.8 / (4 / 3)), constrained_layout=True)
# axs[0,0].plot(np.linspace(0, 0.03, 31), sol.stat_p[:, -1], color = "b", label = "pressure")
# axs[0,0].set_ylabel("pressure [Pa]")
# axs[0,0].set_xlabel("axial vessel position [m]")
# ax1 = axs[0,0].twinx()
# ax1.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
# ax1.plot(np.linspace(0, 0.03, 31), sol.A_sec[:, -1] * sol.H_i_sec[:, -1], color = "r", label = "flow")
# ax1.set_ylabel("flow [m$^3$/s]")
# axs[0,0].legend(loc = 'upper left')
# ax1.legend(loc = 'upper right')
#
# axs[0,1].plot(np.linspace(0, 0.03, 31), p_sv_50_sten, label = "SV 50%", color = to_hex(cmap1(9)), linestyle = (0, (1, 1)))
# axs[0,1].plot(np.linspace(0, 0.03, 31), p_sv_40_sten, label = "SV 60%", color = to_hex(cmap1(6)), linestyle = (0, (2, 3)))
# axs[0,1].plot(np.linspace(0, 0.03, 31), p_sv_30_sten, label = "SV 70%", color = to_hex(cmap1(3)), linestyle = (0, (3, 5)))
# axs[0,1].plot(np.linspace(0, 0.03, 31), p_ph_50_sten, label = "PH 50%", color = to_hex(cmap1(9)), alpha = 0.6)
# axs[0,1].plot(np.linspace(0, 0.03, 31), p_ph_40_sten, label = "PH 60%", color = to_hex(cmap1(6)),  alpha = 0.6)
# axs[0,1].plot(np.linspace(0, 0.03, 31), p_ph_30_sten, label = "PH 70%", color = to_hex(cmap1(3)),  alpha = 0.6)
# axs[0,1].set_ylabel("pressure [Pa]")
# axs[0,1].set_xlabel("axial vessel position [m]")
# axs[0,1].legend(loc = 'upper right')
#
# axs[1,0].plot(np.linspace(0, 0.03, 31), p_sv_50_stra, label = "SV 1mm", color = to_hex(cmap2(9)), linestyle = (0, (1, 1)))
# axs[1,0].plot(np.linspace(0, 0.03, 31), p_sv_40_stra, label = "SV 2mm", color = to_hex(cmap2(6)), linestyle = (0, (2, 3)))
# axs[1,0].plot(np.linspace(0, 0.03, 31), p_sv_30_stra, label = "SV 3mm", color = to_hex(cmap2(3)), linestyle = (0, (3, 5)))
# axs[1,0].plot(np.linspace(0, 0.03, 31), p_ph_50_stra, label = "PH 1mm", color = to_hex(cmap2(9)), alpha = 0.6)
# axs[1,0].plot(np.linspace(0, 0.03, 31), p_ph_40_stra, label = "PH 2mm", color = to_hex(cmap2(6)),  alpha = 0.6)
# axs[1,0].plot(np.linspace(0, 0.03, 31), p_ph_30_stra, label = "PH 3mm", color = to_hex(cmap2(3)),  alpha = 0.6)
# axs[1,0].set_ylabel("pressure [Pa]")
# axs[1,0].set_xlabel("axial vessel position [m]")
# axs[1,0].legend(loc = 'upper right')
#
# axs[1,1].plot(np.linspace(0, 0.03, 31), p_sv_50_sten_rad, label = "SV 1mm", color = to_hex(cmap3(9)), linestyle = (0, (1, 1)))
# axs[1,1].plot(np.linspace(0, 0.03, 31), p_sv_40_sten_rad, label = "SV 2mm", color = to_hex(cmap3(6)), linestyle = (0, (2, 3)))
# axs[1,1].plot(np.linspace(0, 0.03, 31), p_sv_30_sten_rad, label = "SV 3mm", color = to_hex(cmap3(3)), linestyle = (0, (3, 5)))
# axs[1,1].plot(np.linspace(0, 0.03, 31), p_ph_50_sten_rad, label = "PH 1mm", color = to_hex(cmap3(9)), alpha = 0.6)
# axs[1,1].plot(np.linspace(0, 0.03, 31), p_ph_40_sten_rad, label = "PH 2mm", color = to_hex(cmap3(6)),  alpha = 0.6)
# axs[1,1].plot(np.linspace(0, 0.03, 31), p_ph_30_sten_rad, label = "PH 3mm", color = to_hex(cmap3(3)),  alpha = 0.6)
# axs[1,1].set_ylabel("pressure [Pa]")
# axs[1,1].set_xlabel("axial vessel position [m]")
# axs[1,1].legend(loc = 'upper right')
# plt.show()
# #plt.savefig(r'C:\Users\Faiza\Documents\WCB_AbstractSubmission' + '/comparison_sv_ph.pdf')
#
#
#
#
#
# # compare_ph_sv(
# #     '0',
# #     ph_results_paths[GEOMETRY][0],
# #     sv_results_paths[GEOMETRY][0],
# #     N1,
# # )
# #
# # compare_ph_sv(
# #     '1',
# #     ph_results_paths[GEOMETRY][1],
# #     sv_results_paths[GEOMETRY][1],
# #     N1,
# # )
# #
# # compare_ph_sv(
# #     '2',
# #     ph_results_paths[GEOMETRY][2],
# #     sv_results_paths[GEOMETRY][2],
# #     N1,
# # )
# # plt.legend()
#
# # compare_ph_sv_vel(
# #        'r=0.1',
# #        path_ph_c1,
# #        sv_results_paths[GEOMETRY]["0.1"],
# # )
# #
# # compare_ph_sv_vel(
# #        'r=0.2',
# #        path_ph_c2,
# #        sv_results_paths[GEOMETRY]["0.2"],
# # )
# #
# # compare_ph_sv_vel(
# #        'r=0.3',
# #        path_ph_c3,
# #        sv_results_paths[GEOMETRY]["0.3"],
# # )
#
# # p_sv_01, v_sv_01 = get_p_vel(path_sv, 31)
# # p_ph_01, v_ph_01 = get_p_vel_PH(path_PH)
# #
# # #compare for radius 0.2
# # path_PH = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\20211211\solQ0.002.npy'
# # path_sv = r'C:\Users\Faiza\simvascular_simulation\stenosis_doubleradius'
# #
# # p_sv_02, v_sv_02 = get_p_vel(path_sv, 31)
# # p_ph_02, v_ph_02 = get_p_vel_PH(path_PH)
# #
# # #compare for radius 0.2
# # path_PH = r'C:\Users\Faiza\PycharmProjects\PHM_FSI_Pycharm\Results\20211211\solQ0.003.npy'
# # path_sv = r'C:\Users\Faiza\simvascular_simulation\stenosis_doubleradius'
# #
# # p_sv_03, v_sv_03 = get_p_vel(path_sv, 31)
# # p_ph_03, v_ph_03 = get_p_vel_PH(path_PH)
#
#
