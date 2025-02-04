import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_ordered_map_v2(map_data, excel_path="ICNs_v2.xlsx", save_path=None, 
                       range_val=None, cmap='RdBu_r', font_size=14,
                       show_grid=True, show_boundaries=True,
                       title_size=14, label_rotation=30, dpi=300,
                       figsize=(10, 8), boundary_width=2.0,plot_title='Network Connectivity Map'):
    
    map_data = np.maximum(map_data, map_data.T)
    
    df = pd.read_excel(excel_path)
    T = df.values
    icn_idx = df.columns.get_loc('Label')
    
    network_info = [
        ('Visual Network', 'VI'),
        ('Cerebellar network', 'CB'),
        ('Temporal network', 'TM'),
        ('Subcortical network (SC)', 'SC'),
        ('Sensorimotor network (SM)', 'SM'),
        ('Higher Cognition network (HC)', 'HC')
    ]
    
    networks = {}
    all_indices = []
    positions = []
    labels = []
    boundaries = [0]
    current_pos = 0
    
    for name, label in network_info:
        idx = np.where([str(x) == name for x in T[:, icn_idx]])[0]
        if len(idx) > 0:
            networks[name] = T[idx, 0].astype(int) - 1
            size = len(networks[name])
            all_indices.extend(networks[name])
            positions.append(current_pos + size/2)
            labels.append(label)
            current_pos += size
            boundaries.append(current_pos)
    
    total_size = len(all_indices)
    organized_data = np.zeros((total_size, total_size))
    
    for i, idx_i in enumerate(all_indices):
        for j, idx_j in enumerate(all_indices):
            if idx_i < map_data.shape[0] and idx_j < map_data.shape[0]:
                organized_data[i, j] = map_data[idx_i, idx_j]
    
    plt.figure(figsize=figsize)
    im = plt.imshow(organized_data, cmap=cmap, aspect='equal')
    
    if range_val is not None:
        plt.clim(range_val[0], range_val[1])
    else:
        max_val = np.max(np.abs(organized_data))
        plt.clim(-max_val, max_val)
    
    if show_boundaries:
        for boundary in boundaries:
            plt.axvline(x=boundary-0.5, color='black', linewidth=boundary_width)
            plt.axhline(y=boundary-0.5, color='black', linewidth=boundary_width)
    
    if show_grid:
        plt.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.1)
    
    plt.xticks(positions, labels, rotation=label_rotation, 
              fontsize=font_size, fontweight='bold')
    plt.yticks(positions, labels, fontsize=font_size, fontweight='bold')
    
    plt.title(plot_title, pad=20, 
             fontsize=title_size, fontweight='bold')
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size-2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show() 
       
    return None
