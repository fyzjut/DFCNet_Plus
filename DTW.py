import numpy as np

def find_continuous_odd_sequences(path):
    odd_sequences = []
    start_index = None  
    in_sequence = False 
    previous_value = None  

    for i, value in enumerate(path):
        if value % 2 != 0: 
            if not in_sequence:
                start_index = i 
                in_sequence = True
            elif value != previous_value: 
              
                odd_sequences.append((start_index, i - 1))
                start_index = i
        else:
            if in_sequence:  
                odd_sequences.append((start_index, i - 1))
                in_sequence = False
        
        previous_value = value  
    
 
    if in_sequence:
        odd_sequences.append((start_index, len(path) - 1))

    return odd_sequences
def DTW(X):
    #input : X is a probability matrix X:T * num_glosses
    #output: The index of the starting frame corresponding to each vocab
    i = 0
    j = 0
    X =  X.t()
    X = X * 10

    a, b= X.shape
    DTW_matrix = np.zeros((a, b))
    N = int((b-1)/2)
    G = [[] for _ in range(2 * N + 1)] 
    DTW_matrix[0,0] = X[0,0]
    DTW_matrix[0,1] = X[0,1]

    for j in range(b):
        if j == 0:
            G[j] = [j]
        elif j%2==0 or j==1:
            G[j] = [j-1,j]
        else:
            G[j] = [j-2,j-1,j]
  
        for i in range(1,a):
        
            DTW_matrix[i,j] = X[i,j]*max(DTW_matrix[i-1, p] for p in G[j] if p >= 0)

    path = np.zeros(a, dtype=int)
    path[a - 1] = 2 * N-1 + np.argmax(DTW_matrix[a - 1, [2 * N-1, 2 * N ]]) 
    for t in range(a - 2, -1, -1):
        
        path[t] = G[path[t + 1]][np.argmax([DTW_matrix[t, j] for j in G[path[t + 1]]])]

    path_index = find_continuous_odd_sequences(path)

    return path, path_index
    
