import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
import sys

def plot_results(name_of_plot, locations, site_name, legend, colors, markers):
    max_len_site_nb = 0
    max_len_site_volume = 0
    for dir_r in os.listdir(os.path.join(os.getcwd(), 'crawlers', locations[-1], site_name)):
        max_len_site_nb = len(list(np.load(os.path.join(os.getcwd(), 'crawlers', locations[-1], site_name, dir_r, "nb_data_resources.npy"))))/1e3
        data_volume_r = np.load(os.path.join(os.getcwd(), 'crawlers', locations[-1], site_name, dir_r, "data_volumes.npy"), allow_pickle = True)

        for array in data_volume_r:
            x, y = array
            if x > max_len_site_volume:
                max_len_site_volume = x/1e9

    plt.figure(figsize=(12, 6))
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].set_xlabel('Number of HTTP queries (k)', fontsize=17.5)
    axs[0].set_ylabel('Number of crawled targets (k)', fontsize=17.5)
    axs[1].set_xlabel('Volume of non-target responses (GB)', fontsize=17.5)
    axs[1].set_ylabel('Volume of target responses (GB)', fontsize=17.5)

    max_resources = 0
    max_volume = 0
    max_length = 0
    max_length_volume = 0
    markevery = 0
    
    for method_idx, method in enumerate(locations):
        resource_data_list = []
        data_volume_list = []
        marker = markers[method_idx]
        
        site_path = os.path.join(os.getcwd(), 'crawlers', method, site_name)
        
        if not os.path.exists(site_path):
            continue

        for run in os.listdir(site_path):
            run_path = os.path.join(site_path, run)
            
            resource_file = os.path.join(run_path, 'nb_data_resources.npy')
            data_volume_file = os.path.join(run_path, 'data_volumes.npy')
            
            if os.path.exists(resource_file) and os.path.exists(data_volume_file):
                resource_data_list.append(np.load(resource_file))
                data_volume_list.append(np.load(data_volume_file, allow_pickle=True))
        
        if resource_data_list:
            max_length = max(len(data) for data in resource_data_list)
            markevery = int(max_length/10)
            
            resource_data_list = [np.pad(data, (0, max_length - len(data)), 'edge') for data in resource_data_list]

            x_aligned = np.linspace(0, max_len_site_nb, 1005)
            interpolated_list = []
            for resource_data in resource_data_list:
                X = [i/1e3 for i in range(max_length)]
                Y = [val/1e3 for val in resource_data]
                X.append(max_len_site_nb+1)
                Y.append(Y[-1])
                interp = interp1d(X, Y, kind='nearest')
                interpolated_list.append(list(interp(x_aligned)))
            
            resources_data = np.array(interpolated_list)
            mean_resources = np.mean(resources_data, axis=0)
            max_resources = mean_resources[-1]
            std_resources = np.std(resources_data, axis=0)
            markevery_interp = int(len(x_aligned)/10)

            axs[0].plot(x_aligned, mean_resources, label=legend[method_idx], color=colors[method_idx], marker=marker, markevery=markevery_interp)
            axs[0].fill_between(x_aligned, mean_resources - std_resources, mean_resources + std_resources, alpha=0.2, color=colors[method_idx])

            if len(data_volume_list) >= 1:
                
                for data_volume in data_volume_list:
                    X = []
                    Y = []
                    for array in data_volume:
                        x, y = array
                        if y > max_volume:
                            max_volume = y/1e9
                        if x > max_length_volume:
                            max_length_volume = x/1e9

                x_aligned = np.linspace(0, max_len_site_volume, 1005)
                interpolated_list = []
                
                for data_volume in data_volume_list:
                    X = []
                    Y = []
                    X_set = set()
                    for array in data_volume:
                        x, y = array
                        if x not in X_set:
                            X_set.add(x)
                            X.append(x/1e9)
                            Y.append(y/1e9)

                    X.append(max_len_site_volume + 1)
                    Y.append(Y[-1])
                    interp = interp1d(X, Y, kind='nearest')
                    interpolated_list.append(list(interp(x_aligned)))

                mean_volume = np.mean(interpolated_list, axis=0)
                std_volume = np.std(interpolated_list, axis=0)
                markevery_interp = int(len(x_aligned)/10)
                axs[1].plot(x_aligned, mean_volume, label=legend[method_idx], color=colors[method_idx], marker=marker, markevery=markevery_interp)
                axs[1].fill_between(x_aligned, mean_volume - std_volume, mean_volume + std_volume, alpha=0.2, color=colors[method_idx])             
                
            else:
                X = []
                Y = []
                for array in data_volume_list[0]:
                    x, y = array
                    X.append(x/1e9)
                    Y.append(y/1e9)
                axs[1].plot(X, Y, color = colors[method_idx], label=legend[method_idx], marker=marker, markevery=markevery)   
               
    axs[0].plot([0, max_resources, max_len_site_nb], [0, max_resources, max_resources], color = 'red', linestyle='dashed', label="OMNISCIENT")
    axs[0].legend(fontsize=13, loc=4)

    axs[1].plot([0, 0, max_len_site_volume], [0, max_volume, max_volume], color = 'red', linestyle='dashed', label="OMNISCIENT")
    axs[1].legend(fontsize=13, loc=4)

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
            if 'auer/logs/' in location:
                legend.append('SB-CRAWLER')
                colors.append('C1')
                markers.append('s')
            elif 'focused/logs/' in location:
                legend.append('FOCUSED')
                colors.append('C3')
                markers.append('o')
            elif 'offline_dom/logs/' in location:
                legend.append('DOM-OFF')
                colors.append('C2')
                markers.append('D')
            elif 'bfs/logs/' in location:
                legend.append('BFS')
                colors.append('C4')
                markers.append('X')
            elif 'dfs/logs/' in location:
                legend.append('DFS')
                colors.append('C5')
                markers.append('*')
            elif 'random/logs/' in location:
                legend.append('RANDOM')
                colors.append('C6')
                markers.append('P')



        plot_results(name_of_plot, locations, site_name, legend, colors, markers)

