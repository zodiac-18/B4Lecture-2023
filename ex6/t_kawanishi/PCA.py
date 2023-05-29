"""Function about principal component analysis"""
import numpy as np
import matplotlib.pyplot as plt

def normalize(data:np.ndarray) -> np.ndarray:
    """Normalization.

    Args:
        data (np.ndarray): data

    Returns:
        np.ndarray: normalized data
    """
    return (data-np.mean(data,axis=0))/np.std(data,axis=0)

def PCA(data:np.ndarray,norm= False) ->np.ndarray:
    """Principal component analysis.

    Args:
        data (np.ndarray): dataset
        norm (bool, optional): decide whether normalization.
                                Defaults to False.

    Returns:
        np.ndarray: (lambda,1+dim)
        example:[[lambda1,w1,w2,w3],
                 [lambda2,w1,w2,w3],
                 [lambda3,w1,w2,w3]]
    """
    #normalizing dataset
    if norm:
        data = normalize(data)

    # calculate variance-covariance matrix
    cov_matrix = np.cov(data.T)

    # calculate eigenvalues and eigenvectors
    eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
    #print(eigen_vals)
    #print(eigen_vecs)

    # sort by eigenvalues
    eigen_pairs = zip(eigen_vals,eigen_vecs.T)
    eigen_pairs = sorted(list(eigen_pairs))
    eigen_pairs.reverse()
    eigen_vals, eigen_vecs = zip(*eigen_pairs)
    eigen_vals = np.array(eigen_vals)
    eigen_vecs = np.array(eigen_vecs)
    #print(eigen_vals)
    #print(eigen_vecs)

    # grouping
    eigen_pairs = np.hstack(
        [eigen_vals[:,np.newaxis],eigen_vecs]
        )
    #print(eigen_pairs)

    # calculate contribution rate
    c_rate = eigen_pairs[:,0] / sum(eigen_pairs[:,0])
    #print(c_rate)

    return eigen_pairs, c_rate

def dimComp(data:np.ndarray,eigen_pair:np.ndarray,dim:int)->np.ndarray:
    """Dimension compression.

    Args:
        data (np.ndarray): dataset
        eigen_pair (np.ndarray): eigen values and eigen vectors
        dim (int): decide how much dimension compression
    Returns:
        np.ndarray: compressed dataset
    """
    #Get eigen vectors
    t_vectors = eigen_pair[:dim,1:]
    return data @ t_vectors.T

def PlotConRate(c_rate:np.ndarray, targetR=0.9)->int:
    """Find out in what dimension
        cumulative contribution rate are bigger than targetR.

    Args:
        c_rate (np.ndarray): contribution rate
        targetR (float, optional): target cumulative contribution rate.
                                    Defaults to 0.9.

    Returns:
        int: dimension
    """
    #compute cumulative contribution rate
    cumCrate = np.zeros_like(c_rate)
    for i in range(len(cumCrate)):
        cumCrate[i] = np.sum(c_rate[:i+1])
    dim = np.count_nonzero(cumCrate<0.9)+1

    #plot graph
    plt.figure()
    plt.plot(np.arange(1,len(cumCrate)+1),cumCrate,label="Cum contribution R")
    plt.plot([1,len(cumCrate)],[targetR,targetR])
    plt.plot(
        dim,cumCrate[dim-1],"o",color="r",
        label="dimension: " + str(dim) +"\nCum contribution R: "
        + "{:.3f}".format(cumCrate[dim-1]))
    plt.title("cumulative contribution rate")
    plt.legend()
    return None

if __name__ == "__main__":
    a = np.array([3,1,2,4]).reshape(2,2)
    eigen_vals, eigen_vecs = np.linalg.eig(a)
    eigen_pairs = zip(eigen_vals,eigen_vecs.T)
    eigen_pairs = sorted(list(eigen_pairs))
    eigen_pairs.reverse()
    eigen_vals, eigen_vecs = zip(*eigen_pairs)
    eigen_vals = np.array(eigen_vals)
    eigen_vecs = np.array(eigen_vecs)


