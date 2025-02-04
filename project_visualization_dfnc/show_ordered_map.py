import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_ordered_map(map_data, excel_path="ICNs_v2.xlsx", save_path=None, range_val=None):
    try:
        # Ensure matrix symmetry
        map_data = np.maximum(map_data, map_data.T)
        
        # Read Excel file
        df = pd.read_excel(excel_path)
        T = df.values
        icn_idx = df.columns.get_loc('Label')
        
        # Define networks and their short labels
        network_info = [
            ('Visual Network', 'VI'),
            ('Cerebellar network', 'CB'),
            ('Temporal network', 'TM'),
            ('Subcortical network (SC)', 'SC'),
            ('Sensorimotor network (SM)', 'SM'),
            ('Higher Cognition network (HC)', 'HC')
        ]
        
        # Extract indices for each network
        networks = {}
        for name, label in network_info:
            idx = np.where([str(x) == name for x in T[:, icn_idx]])[0]
            if len(idx) > 0:
                networks[name] = {
                    'indices': T[idx, 0].astype(int) - 1,  # Convert to 0-based indexing
                    'label': label
                }
        
        # Create ordered indices and labels
        all_indices = []
        positions = []
        labels = []
        current_pos = 0
        
        for name, info in networks.items():
            indices = info['indices']
            size = len(indices)
            if size > 0:
                all_indices.extend(indices)
                positions.append(current_pos + size/2)
                labels.append(info['label'])
                current_pos += size
        
        # Create and fill organized data matrix
        total_size = len(all_indices)
        organized_data = np.zeros((total_size, total_size))
        
        for i, idx_i in enumerate(all_indices):
            for j, idx_j in enumerate(all_indices):
                if idx_i < map_data.shape[0] and idx_j < map_data.shape[0]:
                    organized_data[i, j] = map_data[idx_i, idx_j]
        
        # Visualization
        plt.figure(figsize=(10, 8))
        im = plt.imshow(organized_data, cmap='RdBu_r', aspect='equal')
        
        # Set color range
        if range_val is not None:
            plt.clim(range_val[0], range_val[1])
        else:
            plt.clim(-np.max(np.abs(organized_data)), np.max(np.abs(organized_data)))
        
        # Add colorbar and labels
        plt.colorbar(im)
        plt.xticks(positions, labels, rotation=45)
        plt.yticks(positions, labels)
        plt.title('Network Connectivity Map')
        
        # Adjust layout and save
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing visualization: {str(e)}")
        print(f"Network sizes: {[len(net['indices']) for net in networks.values()]}")
        print(f"Positions: {positions}")
        print(f"Labels: {labels}")
        raise
