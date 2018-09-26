# Automatic-Clustering-via-Outward-Statistical-Testing-on-Density-Metrics
This algorithm proposes a new clustering algorithm that can detect the clustering centers automatically via statistical testing. 

Specifically, the proposed algorithm first defines a new metric to measure the density of an object that is more robust to the preassigned parameter, further generates a metric to evaluate the centrality of each object.   

Afterwards, it identifies the objects with extremely large centrality metrics as the clustering centers via an outward statistical testing method.   

Finally, it groups the remaining objects into clusters containing their nearest neighbors with higher density.

### Run clustering.py  
#### Sample Input:
    path = '/Clusterdata/vector_.txt'   
    auto_cluster(path,50,10,20,1)     
#### Input Index:
    num_of_cluster  
    k_nearest_neighbor: number of nearest neighborhood when calculating the K-density.  
    top_n_of_each_cluster: Choose the top n from each cluster results based on similarity distance from centers.  
    0/other: Whether to show the graph. Yes: input a number to represent the length of x_axle; No: 0  
### Result Explanation:  
    result_df: cluster results for top n values in each cluster  
    cluster_df_top: value with its predicted attributes and probability  
