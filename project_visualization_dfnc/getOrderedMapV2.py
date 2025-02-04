import numpy as np
import pandas as pd

def getOrderedMapV2(map_matrix):
    # Convert input to numpy array if not already
    sim = np.array(map_matrix)
    
    # Read excel file
    tbl = pd.read_excel('ICNs_v2.xlsx')
    
    # Get the order and sort it
    leafOrder = tbl['new_order'].values
    leafOrder = np.argsort(leafOrder)
    
    # Reorder the matrix using the sorted indices
    sim = sim[leafOrder][:, leafOrder]
    
    orderedMap = sim
    return orderedMap