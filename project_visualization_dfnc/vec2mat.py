import numpy as np

def vec2mat(vec, full=True, fill_nan=False):
    """
    Convert a vector to a correlation matrix.
    
    Parameters:
    vec : array_like
        Input vector or 2D array
    full : bool, optional
        If True, returns the symmetric matrix
    fill_nan : bool, optional
        If True, initializes matrix with NaN instead of zeros
    
    Returns:
    mat : ndarray
        Correlation matrix
    """
    
    # Handle 1D vector input
    if np.ndim(vec) == 1:
        N = len(vec)
        # Calculate matrix dimension using the quadratic formula
        n = int(1/2 + np.sqrt(1 + 8*N)/2)
        
        # Initialize matrix
        mat = np.full((n, n), np.nan) if fill_nan else np.zeros((n, n))
        
        # Create mask for lower triangle
        temp = np.ones((n, n))
        ind = np.where(temp - np.triu(temp) > 0)
        
        # Fill lower triangle
        mat[ind] = vec
        
        if full:
            # Create symmetric matrix
            tempmat = np.rot90(np.flipud(mat))
            tempmat[ind] = vec
            mat = np.flipud(np.rot90(tempmat))
            
    # Handle 2D array input
    elif np.ndim(vec) == 2:
        p, N = vec.shape
        n = int(1/2 + np.sqrt(1 + 8*N)/2)
        mat = np.zeros((p, n, n))
        
        # Create mask for lower triangle
        temp = np.ones((n, n))
        ind = np.where(temp - np.triu(temp) > 0)
        
        for i in range(p):
            tempmat = np.zeros((n, n))
            tempmat[ind] = vec[i, :]
            
            if full:
                tempmat2 = np.rot90(np.flipud(tempmat))
                tempmat2[ind] = vec[i, :]
                mat[i] = np.flipud(np.rot90(tempmat2))
            else:
                mat[i] = tempmat
                
    return mat