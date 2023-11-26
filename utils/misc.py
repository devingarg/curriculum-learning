import numpy as np
from sklearn.cluster import KMeans

def get_features(extractor, dataloader, device):
    """ 
    Takes in a feature extractor and a dataloader. Extracts
    features for all the data samples in the dataloader using the
    extractor and returns the features as an ndarray.
    
    Note: The batch dimension is dissolved.
    """
    
    extractor.eval()
    extractor = extractor.to(device)
    
    result = None
    for i, (x, y) in enumerate(dataloader):
        output = extractor(x.to(device)).squeeze().detach().cpu().numpy()
        if type(result)==np.ndarray:
            # not empty
            result = np.concatenate([result, output], axis=0)
        else:
            result = output.copy()

    return result


def get_pairwise_distance(features):
    """
    Given N input features, returns an NxN matrix that has
    pairwise distances between every pair of vectors in the input. 
    """
    dist = pdist(features, metric="euclidean")
    return dist


def cluster_features(features, num_clusters=5):
    """ 
    Given N features & a cluster count as input, clusters the features
    using KMeans and returns an array containing N cluster assignments - 
    one for every input feature.
    """
    cobj = KMeans(n_clusters=num_clusters)
    
    # do the clustering
    cobj.fit(features)
    
    # get cluster assignments for the feature vectors
    assignments = cobj.labels_
    
    return assignments