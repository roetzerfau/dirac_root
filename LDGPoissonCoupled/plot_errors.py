import matplotlib.pyplot as plt
import numpy as np
import re
import math

# Read file
#with open("convergence_results_test_coupled_09_04_finalResults_cons_sol_3_geoconfig_2_gradedMesh_true_coupled_true_paper_solution_true_solution_linear_1_vessel_false_omegaonface_true_LA_false_rad_0.001000_D_1.000000_penalty_10.000000.txt", "r") as f:
#with open("convergence_results_test_coupled_25_05_finalResults_cons_sol_3_geoconfig_2_gradedMesh_true_coupled_true_paper_solution_true_solution_linear_1_vessel_false_omegaonface_true_LA_false_rad_0.001000_D_1.000000_penalty_10.000000.txt", "r") as f:
#with open("convergence_results_test_coupled_21_05_finalResults_cons_sol_3_geoconfig_2_gradedMesh_true_coupled_true_paper_solution_true_solution_linear_1_vessel_false_omegaonface_true_LA_false_rad_0.001000_D_1.000000_penalty_10.000000.txt", "r") as f:
#with open("convergence_results_test_coupled_22_05_finalResults_cons_sol_1_geoconfig_2_gradedMesh_true_coupled_true_paper_solution_true_solution_linear_2_vessel_false_omegaonface_true_LA_false_rad_0.001000_D_1.000000_penalty_10.000000.txt", "r") as f:


#with open("convergence_results_uncoupled_12_05_finalResults_cons_sol_3_geoconfig_0_gradedMesh_false_coupled_false_paper_solution_true_solution_linear_3_vessel_true_omegaonface_true_LA_false_rad_0.250000_D_0.100000_penalty_5.000000_onedim_gap_true.txt", "r") as f:
name = "convergence_results_uncoupled_12_05_finalResults_cons_sol_3_geoconfig_0_gradedMesh_false_coupled_false_paper_solution_true_solution_linear_3_vessel_true_omegaonface_true_LA_false_rad_0.250000_D_0.100000_penalty_5.000000_onedim_gap_true.txt"
name = "convergence_results_test_coupled_16_05_finalResults_cons_sol_3_geoconfig_2_gradedMesh_false_coupled_true_paper_solution_true_solution_linear_3_vessel_false_omegaonface_true_LA_false_rad_0.050000_D_1.000000_penalty_5.000000_onedim_gap_true_modified.txt"
match = re.search(r'geoconfig_(\d+)', name)

geocfg = int(match.group(1))
#print("Number after geoconfig:", geocfg)

with open(name, "r") as f:
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
        for j,line in enumerate(lines[2:]):
            
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
                print("j")
                err = np.nan
            #if(p != 2 or j !=4):

            if(err == 0):
                #print(j)
                break
            h_array.append(h)
            error_array.append(err)

        #print(np.array(error_array))            
        if(i == 0):
            data_dict[title]={"h_p" + str(int(p)): np.array(h_array), "error_p"+ str(int(p)): np.array(error_array)}
        else:
            data_dict[title].update({"h_p" + str(int(p)): np.array(h_array), "error_p"+ str(int(p)): np.array(error_array)})
        #print(error_array)
# Plotting setup
if(geocfg == 2):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
else:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
axs = axs.flatten()

colors = ['orangered', 'orange', 'blue', 'purple']
titles = list(data_dict.keys())
#print(titles)
if(geocfg == 0):
    titles.pop()
    titles.pop()
titles.remove('U_star_Omega')
for i, key in enumerate(titles):
    ax = axs[i]
    d = data_dict[key]
    #print(data_dict.keys())
    #print(d)
    ax.loglog(d["h_p0"], d["error_p0"], 's:', color=colors[i], label=r"$k=0$", markersize=6, linewidth=2)
    ax.loglog(d["h_p1"], d["error_p1"], 's-', color=colors[i], label=r"$k=1$", markersize=6, linewidth=2)
    ax.loglog(d["h_p2"], d["error_p2"], 'o--', color=colors[i], label=r"$k=2$", markersize=6, linewidth=2, alpha=0.7)

    if(key == "U_Omega"):
        ax.set_title(r"$\Vert U_h-U \Vert_{L^2(\Omega)}$", fontsize=12)
    elif (key == "Q_Omega"):
        ax.set_title(r"$\Vert \mathbf{Q}_h-\mathbf{Q} \Vert_{L^2(\Omega)}$", fontsize=12)
    elif(key== "u_omega"):
        ax.set_title(r"$\Vert u_h-u \Vert_{L^2(\omega)}$", fontsize=12)
    elif (key == "q_omega"):
        ax.set_title(r"$\Vert \mathbf{q}_h-\mathbf{q} \Vert_{L^2(\omega)}$", fontsize=12)
    else:
        ax.set_title(key, fontsize=12)


    ax.set_xlabel(r"$h$", fontsize=11)
    if(key == "U_Omega" or key == "Q_Omega"):
        ax.set_ylabel(r"$L^2$ error", fontsize=11)#_{\alpha}
    else:
        ax.set_ylabel(r"$L^2$ error ", fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='lower right')
    #ax.tick_params(labelsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)


    #from matplotlib.patches import Polygon
    #triangle = Polygon([[0.011, 0.006], [0.014, 0.006], [0.014, 0.012]], closed=True,
    #                   edgecolor='black', facecolor='white', lw=2)
    #ax.add_patch(triangle)
    #ax.text(0.014, 0.0045, r'$1$', fontsize=10)

    # Draw a triangle showing the slope visually
# Triangle base from x1 to x2
    if(key == "U_Omega" or key == "Q_Omega" or key == "u_omega" or key == "q_omega"):
        d = data_dict[key]
        diffx = abs(math.log10(d["h_p0"][-2]) - math.log10(d["h_p0"][-3]))
        
        x1 = d["h_p0"][-2]#pow(10,-1.5)
        x2 = d["h_p0"][-3]#pow(10,-1.25)

        y1_exponent = math.log10(d["error_p1"][-2])#math.log10(d["error_p0"][-2]) - abs(math.log10(d["error_p0"][-2])-math.log10(d["error_p1"][-2]))/2
        y1 = pow(10, y1_exponent) #pow(10,-8)
        con_ord = 1
        if ((key == "U_Omega" and geocfg == 2 ) or key == "u_omega"):
            con_ord = 2
        elif ((key == "U_Omega" and geocfg == 0 ) or key == "u_omega"):
            con_ord = 1.5
        elif((key == "Q_Omega" and geocfg == 2 )or key == "q_omega" ):
            con_ord = 1
        elif((key == "Q_Omega" and geocfg == 0 )or key == "q_omega" ):
            con_ord = 0.5
        else:
            con_ord = 1

        y2 = pow(10, y1_exponent+ con_ord * diffx )# pow(10,-7.5)
        # Draw triangle
        triangle_x = [x1, x2, x2]
        triangle_y = [y1, y1, y2]
        ax.plot(triangle_x + [x1], triangle_y + [y1], 'k', lw=1.5)


        # Eckpunkte
        A = np.array([x1, y1])
        B = np.array([x2, y1])
        C = np.array([x2, y2])

        # Seitenlängen gegenüber der Punkte
        a = np.linalg.norm(B - C)  # gegenüber A
        b = np.linalg.norm(A - C)  # gegenüber B
        c = np.linalg.norm(A - B)  # gegenüber C

        # Inkreismittelpunkt
        incenter = (a * A + b * B + c * C) / (a + b + c)

        ix, iy = incenter

        #print("Inkreismittelpunkt:", ix, iy)

        if(key == "U_Omega" and geocfg == 0 ):
            ax.text(x1, y2, rf"$\sim h^{{3/2}}$", fontsize=12)
        elif(key == "Q_Omega" and geocfg == 0 ):
            ax.text(x1, y2, rf"$\sim h^{{1/2}}$", fontsize=12)
        else:
            ax.text(x1, y2, rf"$\sim h^{con_ord}$", fontsize=12)
    else:
        x1 = pow(10,-1.5)
        x2 = pow(10,-1.25)
        y1 = pow(10,-4)
        y2 = pow(10,-3.75)
        # Draw triangle
        triangle_x = [x1, x2, x2]
        triangle_y = [y1, y1, y2]
        ax.plot(triangle_x + [x1], triangle_y + [y1], 'k', lw=1.5)
        ax.text(x1 * 1.1, y1 * 1.5, r"$\sim h^1$", fontsize=12)




n = 3

label = rf"$\sim h^{con_ord}$"
#print(label)
plt.tight_layout()
plt.show()