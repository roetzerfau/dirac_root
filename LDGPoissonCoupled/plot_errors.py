import matplotlib.pyplot as plt
import numpy as np
import re

# Read file
#with open("convergence_results_test_coupled_09_04_finalResults_cons_sol_3_geoconfig_2_gradedMesh_true_coupled_true_paper_solution_true_solution_linear_1_vessel_false_omegaonface_true_LA_false_rad_0.001000_D_1.000000_penalty_10.000000.txt", "r") as f:
with open("convergence_results_test_coupled_25_05_finalResults_cons_sol_3_geoconfig_2_gradedMesh_true_coupled_true_paper_solution_true_solution_linear_1_vessel_false_omegaonface_true_LA_false_rad_0.001000_D_1.000000_penalty_10.000000.txt", "r") as f:
    content = f.read()

# Split by sections
sections = content.strip().split("\n\n")
data_dict = {}

for section in sections:
    lines = section.strip().splitlines()
    title = lines[0].strip()
    header = lines[1].strip()
    degrees = [int(p) for p in re.findall(r'error p=(\d+)', header)]

    for i,p in enumerate(degrees):
        #print(i,p)
        h_array, error_array = [], []
        for line in lines[3:]:
            p_parts = line.split(",")
        
            part = p_parts[i]

           
            
            part_array = part.split(";")
            if(i == 0):
                part_array = part_array[1:]

            # Extract h and error 
            h = float(part_array[0])
            try:
                err = float(part_array[2].split()[0])
            except ValueError:
                err = np.nan
            h_array.append(h)
            error_array.append(err)
        if(i == 0):
            data_dict[title]={"h_p" + str(int(p)): np.array(h_array), "error_p"+ str(int(p)): np.array(error_array)}
        else:
            data_dict[title].update({"h_p" + str(int(p)): np.array(h_array), "error_p"+ str(int(p)): np.array(error_array)})

# Plotting setup
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = axs.flatten()

colors = ['orangered', 'orange', 'blue', 'purple']
titles = list(data_dict.keys())

for i, key in enumerate(titles):
    ax = axs[i]
    d = data_dict[key]
    print(data_dict.keys())
    print(d)
    ax.loglog(d["h_p1"], d["error_p1"], 's-', color=colors[i], label=r"$p=1$", markersize=6, linewidth=2)
    ax.loglog(d["h_p2"], d["error_p2"], 'o--', color=colors[i], label=r"$p=2$", markersize=6, linewidth=2, alpha=0.7)

    ax.set_title(key, fontsize=12)
    ax.set_xlabel(r"$h$", fontsize=11)
    if(key == "U_Omega" or key == "Q_Omega"):
        ax.set_ylabel(r"$L^2_{\alpha}$ error", fontsize=11)
    else:
        ax.set_ylabel(r"$L^2$ error", fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)
    #ax.tick_params(labelsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)


    #from matplotlib.patches import Polygon
    #triangle = Polygon([[0.011, 0.006], [0.014, 0.006], [0.014, 0.012]], closed=True,
    #                   edgecolor='black', facecolor='white', lw=2)
    #ax.add_patch(triangle)
    #ax.text(0.014, 0.0045, r'$1$', fontsize=10)

    # Draw a triangle showing the slope visually
# Triangle base from x1 to x2
    x1 = 1e-3
    x2 = 1e-2
    y1 = 1e-3
    y2 = 1e-2

    # Draw triangle
    triangle_x = [x1, x2, x2]
    triangle_y = [y1, y1, y2]
    ax.plot(triangle_x + [x1], triangle_y + [y1], 'k', lw=1.5)
    ax.text(x1 * 1.1, y1 * 1.5, r"$\sim h^1$", fontsize=12)



plt.tight_layout()
plt.show()