import numpy as np

def TD(x, O, T):
    # TD learning for partially observable MDP.
    # Author: Dr. Samuel J. Gershman
    #
    # USAGE: results = TD(x,O,T)
    #
    # INPUTS:
    #   x - stimulus timeseries (1 = nothing, 2 = stimulus, 3 = reward)
    #   O - [S x S x M] observation distribution: O(i,j,m) = P(x'=m|s=i,s'=j)
    #   T - [S x S] transition distribution: T(i,j) = P(s'=j|s=i)
    #
    # OUTPUTS:
    #   results - dictionary with the following keys:
    #               'w' - weight vector at each time point (prior to updating)
    #               'b' - belief state at each time point (prior to updating)
    #               'rpe' - TD error (after updating)
    
    # initialization
    S = T.shape[0]      # number of states
    b = np.ones(S) / S    # belief state
    w = np.zeros(S)     # weights
    
    # learning rates
    alpha = 0.1          # value learning rate
    gamma = 0.98
    
    # Create a dictionary to store the results
    results = {'w': [], 'b': [], 'rpe': []}
    
    for t in range(len(x)):
        
        b0 = b.copy() # old posterior, used later
        b = np.dot(b, (T * O[:,:,x[t]]).T)
        b = b / np.sum(b)

        # TD update
        w0 = w.copy()
        r = int(x[t]==3)        # reward
        rpe = r + np.dot(w, gamma*b-b0)  # TD error
        w = w + alpha*rpe*b0         # weight update
        
        # store results
        results['w'].append(w0)
        results['b'].append(b0)
        results['rpe'].append(rpe)
        results['value'] = w.dot(b0) #estimated value
    
    return results
