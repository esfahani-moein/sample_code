import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

def show_ordered_map_grid(map_data_list, excel_path="ICNs_v2.xlsx", 
                         save_path=None, output_dir="figures",
                         save_format="pdf", dpi=600,
                         titles=None, grid_shape=(2,3), range_val=None, 
                         cmap='RdBu_r', font_size=10, show_boundaries=True, 
                         title_size=12, label_rotation=45, figsize=(15, 10), 
                         boundary_width=2.0):
    
    if titles is None:
        titles = [f'Network Map {i+1}' for i in range(len(map_data_list))]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_shape[0], grid_shape[1] + 1,
                          width_ratios=[1]*grid_shape[1] + [0.05])
    
    if range_val is None:
        all_vals = np.concatenate([map_data.flatten() for map_data in map_data_list])
        max_val = np.max(np.abs(all_vals))
        range_val = (-max_val, max_val)
    
    images = []
    
    for idx, (map_data, title) in enumerate(zip(map_data_list, titles)):
        row = idx // grid_shape[1]
        col = idx % grid_shape[1]
        ax = fig.add_subplot(gs[row, col])
        
        # Process data
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
            idx_net = np.where([str(x) == name for x in T[:, icn_idx]])[0]
            if len(idx_net) > 0:
                networks[name] = T[idx_net, 0].astype(int) - 1
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
        
        im = ax.imshow(organized_data, cmap=cmap, aspect='equal')
        im.set_clim(range_val[0], range_val[1])
        images.append(im)
        
        if show_boundaries:
            for boundary in boundaries:
                ax.axvline(x=boundary-0.5, color='black', linewidth=boundary_width)
                ax.axhline(y=boundary-0.5, color='black', linewidth=boundary_width)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=label_rotation, fontsize=font_size, fontweight='bold')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=font_size, fontweight='bold')
        ax.set_title(title, pad=10, fontsize=title_size, fontweight='bold')
    
    # Add colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(images[-1], cax=cbar_ax)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if save_path:
        import os
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate full save path
        if not save_path.endswith(f'.{save_format}'):
            save_path = f"{save_path}.{save_format}"
        full_save_path = os.path.join(output_dir, save_path)
    
    if save_path:
        plt.savefig(full_save_path, 
                   dpi=dpi,
                   format=save_format,
                   bbox_inches='tight',
                   pad_inches=0.1,
                   transparent=True,
                   metadata={'Creator': 'show_ordered_map_grid'})
        print(f"Figure saved as: {full_save_path}")
    
    plt.show()