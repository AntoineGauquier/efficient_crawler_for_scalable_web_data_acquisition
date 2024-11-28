import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
import sys

def plot_results(name_of_plot, locations, site_name, legend, colors, markers):
    max_len_site_nb = 0
    max_len_site_volume = 0

    plt.figure(figsize=(12, 6))
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].set_xlabel('Number of HTTP queries (k)', fontsize=17.5)
    axs[0].set_ylabel('Number of crawled targets (k)', fontsize=17.5)
    axs[1].set_xlabel('Volume of non-target responses (GB)', fontsize=17.5)
    axs[1].set_ylabel('Volume of target responses (GB)', fontsize=17.5)

    markevery = 0
    max_resources = 0
    
    for method_idx, method in enumerate(locations):
        resource_data = []
        data_volume = []
        marker = markers[method_idx]
        
        site_path = os.path.join(os.getcwd(), method, site_name)
       
        print(site_path)

        if not os.path.exists(site_path):
            continue

        for run in os.listdir(site_path):
            run_path = os.path.join(site_path, run)
            
            resource_file = os.path.join(run_path, 'nb_data_resources.npy')
            data_volume_file = os.path.join(run_path, 'data_volumes.npy')
            
            if os.path.exists(resource_file) and os.path.exists(data_volume_file):
                resource_data = np.load(resource_file)
                data_volume = np.load(data_volume_file, allow_pickle=True)

                if resource_data[-1]/1e3 > max_resources:
                    max_resources = resource_data[-1]/1e3
        
        markevery = int(len(resource_data)/10)

        X = [i/1e3 for i in range(len(resource_data))]
        Y = [val/1e3 for val in resource_data]

        x_aligned = np.linspace(0, X[-1], 1005)

        interp = interp1d(X, Y, kind='nearest')
        interpolated = list(interp(x_aligned))
        markevery_interp = int(len(x_aligned)/10)

        axs[0].plot(x_aligned, interpolated, label=legend[method_idx], color=colors[method_idx], marker=marker, markevery=markevery_interp)
                
        X = []
        Y = []
        X_set = set()
        for array in data_volume:
            x, y = array
            if x not in X_set:
                X_set.add(x)
                X.append(x/1e9)
                Y.append(y/1e9)

            x_aligned = np.linspace(0, X[-1], 1005)

        interp = interp1d(X, Y, kind='nearest')
        interpolated = list(interp(x_aligned))
        markevery_interp = int(len(x_aligned)/10)

        axs[1].plot(x_aligned, interpolated, label=legend[method_idx], color=colors[method_idx], marker=marker, markevery=markevery_interp) 
               
    axs[0].plot([0, max_resources], [0, max_resources], color = 'red', linestyle='dashed', label="OMNISCIENT")
    axs[0].legend(fontsize=13, loc=4)

    for ax in axs:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "graphical_results/plots", f'{name_of_plot}.pdf'))
    print("Plot successfully saved at " + os.path.join(os.getcwd(), "graphical_results/plots", f'{name_of_plot}.pdf'))
        

if __name__ == "__main__":
    name_of_plot = sys.argv[1]
    site_name = sys.argv[2]

    locations = sys.argv[3].split(" ")

    if len(locations) == 1 and locations[0] == "":
        print("You must select at least one crawler.")
    
    else:
        legend = []
        colors = []
        markers = []
        for location in locations:
            if 'auer_crawler' in location:
                legend.append('SB-CRAWLER')
                colors.append('C1')
                markers.append('s')
            elif 'focused_online_crawler' in location:
                legend.append('FOCUSED')
                colors.append('C3')
                markers.append('o')
            elif 'dom_off_online_crawler' in location:
                legend.append('DOM-OFF')
                colors.append('C2')
                markers.append('D')
            elif 'bfs_online_crawler' in location:
                legend.append('BFS')
                colors.append('C4')
                markers.append('X')
            elif 'dfs_online_crawler' in location:
                legend.append('DFS')
                colors.append('C5')
                markers.append('*')
            elif 'random_online_crawler' in location:
                legend.append('RANDOM')
                colors.append('C6')
                markers.append('P')



        plot_results(name_of_plot, locations, site_name, legend, colors, markers)

