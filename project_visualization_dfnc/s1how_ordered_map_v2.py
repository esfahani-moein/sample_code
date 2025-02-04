import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_ordered_map_v2(corr_mat, comp_names=None, title_str=None, save_path=None):
    """
    Display correlation matrix as a heatmap with ordered components
    
    Parameters:
    -----------
    corr_mat : numpy.ndarray
        Correlation matrix to display
    comp_names : list, optional
        List of component names
    title_str : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # If component names not provided, create numeric labels
    if comp_names is None:
        comp_names = [str(i) for i in range(1, corr_mat.shape[0] + 1)]
    
    # Create heatmap
    sns.heatmap(corr_mat, 
                xticklabels=comp_names,
                yticklabels=comp_names,
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'})
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Set title if provided
    if title_str:
        plt.title(title_str)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample correlation matrix
    np.random.seed(42)
    sample_corr = np.random.uniform(-1, 1, (10, 10))
    sample_corr = (sample_corr + sample_corr.T) / 2  # Make it symmetric
    np.fill_diagonal(sample_corr, 1)  # Set diagonal to 1
    
    # Example component names
    comp_names = [f"Comp{i}" for i in range(1, 11)]
    
    # Show map
    show_ordered_map_v2(sample_corr, 
                       comp_names=comp_names,
                       title_str="Sample Correlation Matrix",
                       save_path="correlation_map.png")