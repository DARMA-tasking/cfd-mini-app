import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def openFoam_results_reader(path_to_file):
    velocity_values = []
    with open(path_to_file) as f:
        f = f.readlines()
        in_value_list = False
        line = 0
        # detect beginning of mesh velocity values
        while not in_value_list:
            line_content = f[line]
            if line_content == '(\n':
                in_value_list = True
            line += 1
        i = 0
        # extract velocities from openFoam results
        while in_value_list:
            line_content = f[line]
            if line_content == ')\n': # exit at end of values list
                break
                in_value_list = False
            line_values = line_content[1:len(line_content) - 2].split(' ')
            velocity_values.append([float(line_values[0]), float(line_values[1]), float(line_values[2])])
            line += 1
            i += 1
    return(velocity_values)

def cppfd_results_reader(path_to_file):
    velocity_values = []
    with open(path_to_file) as f:
        f = f.readlines()
        n_lines = len(f)
        # extract velocities from
        for i in range(n_lines):
            line_content = f[i]
            line_values = line_content[:len(line_content) - 1].split(' ')
            velocity_values.append([float(line_values[0]), float(line_values[1])])
    return(velocity_values)

def main():
    # input paths to result files and size of square mesh
    path_to_openFoam_file = str(sys.argv[1])
    path_to_cppfd_file = str(sys.argv[2])

    # extract velocity values from openFoam results
    openFoam_velocity_values = openFoam_results_reader(path_to_openFoam_file)
    N_oF = int(round(np.sqrt(len(openFoam_velocity_values))))

    # reshape array of horizontal velocities in correct order
    openFoam_velocity_values = openFoam_velocity_values[::-1]
    t = np.array([openFoam_velocity_values[i][0] for i in range(len(openFoam_velocity_values))])
    openFoam_velocity_matrix = t.reshape((N_oF, N_oF))

    # extract velocity values from cppfd results and reshape to array of horizontal velocities
    cppfd_velocity_values = cppfd_results_reader(path_to_cppfd_file)
    N_cppfd = int(round(np.sqrt(len(cppfd_velocity_values))))
    t = np.array([cppfd_velocity_values[i][0] for i in range(len(cppfd_velocity_values))])
    cppfd_velocity_matrix = t.reshape((N_cppfd, N_cppfd))

    # plot velocities along center line
    y_openFoam = [openFoam_velocity_matrix[i][N_oF // 2] for i in range(N_oF - 1, -1, -1)]
    y_cppfd = [cppfd_velocity_matrix[i][N_cppfd // 2] for i in range(N_cppfd - 1, -1, -1)]
    x_openFoam = np.arange(0, 1, 1 / N_oF)
    x_cppfd = np.arange(0, 1, 1 / N_cppfd)
    plt.plot(x_openFoam, y_openFoam)
    plt.plot(x_cppfd, y_cppfd)
    plt.show()

if __name__ == '__main__':
    main()
