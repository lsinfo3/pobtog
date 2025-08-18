import numpy as np 
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.distances import great_circle_distance_matrix

def solve_tsp(points):
    """
    Solves the Traveling Salesman Problem for a set of points with lon/lat coordinates.
    
    Args:
        points: A 2D numpy array where each row represents a point [lon, lat, ...]
               Only the first two columns (lon, lat) are used for distance calculation.
        
    Returns:
        A 2D numpy array where each row contains two indices representing
        an edge in the TSP tour.
    """
    if len(points) <= 1:
        return np.array([])
    
    # Extract lon/lat coordinates (first two columns)
    coords = points[:, :2]
    
    # Calculate great circle distance matrix
    # Note: great_circle_distance_matrix expects coordinates as [lat, lon], so swap columns
    lat_lon_coords = coords[:, [1, 0]]  # Swap lon, lat to lat, lon
    
    try:
        dist_matrix = great_circle_distance_matrix(lat_lon_coords)
        
        # Verify the distance matrix has no NaN or infinite values
        if np.any(np.isnan(dist_matrix)) or np.any(np.isinf(dist_matrix)):
            raise ValueError("Distance matrix contains NaN or infinite values")
        
        # Solve TSP using simulated annealing
        permutation, distance = solve_tsp_simulated_annealing(dist_matrix)
        
        # Create path that includes returning to the start
        path = list(permutation) + [permutation[0]]
        
        # Convert the path to edge format (pairs of indices)
        edges = np.array([[path[i], path[i+1]] for i in range(len(path)-1)])
        
        return edges
        
    except Exception as e:
        print(f"TSP solver error: {e}. Falling back to simple path.")
        # Fallback to a simple path when the solver fails
        n = len(points)
        if n <= 1:
            return np.array([])
            
        # Create a simple path: 0->1->2->...->n-1->0
        path = list(range(n)) + [0]
        edges = np.array([[path[i], path[i+1]] for i in range(len(path)-1)])
        return edges

def solve_tsp_large(points, max_points=1000):
    """
    A more efficient TSP solver for large datasets using hierarchical clustering.
    
    Args:
        points: A 2D numpy array where each row represents a point [x, y]
        max_points: Maximum number of points to consider for full optimization
        
    Returns:
        A 2D numpy array where each row contains two indices representing
        an edge in the TSP tour
    """
    if len(points) <= 1:
        return np.array([])
        
    if len(points) <= max_points:
        return solve_tsp(points)  # Use the regular algorithm for smaller datasets
    
    n = len(points)
    
    # Determine number of clusters based on data size
    n_clusters = min(max(int(np.sqrt(n)), 2), 200)
    
    # Cluster the points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(points)
    cluster_centers = kmeans.cluster_centers_
    
    # Sort points by cluster
    cluster_points = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        cluster_points[label].append(i)
    
    # Solve TSP for cluster centers
    center_tour = solve_tsp(cluster_centers)
    
    # Extract the order of clusters to visit
    if len(center_tour) > 0:
        cluster_order = [int(center_tour[0, 0])]
        for edge in center_tour:
            next_cluster = int(edge[1])
            if next_cluster not in cluster_order:
                cluster_order.append(next_cluster)
    else:
        cluster_order = list(range(n_clusters))
    
    # Create final path
    path = []
    
    # Safely initialize the path with a point from the first non-empty cluster
    for cluster_idx in cluster_order:
        if cluster_points[cluster_idx]:
            path.append(cluster_points[cluster_idx][0])
            cluster_points[cluster_idx].remove(path[0])
            break
    
    # If no points found (all clusters empty), return empty result
    if not path:
        return np.array([])
    
    # Now process the clusters in the determined order
    for cluster_idx in cluster_order:
        if not cluster_points[cluster_idx]:  # Skip empty clusters
            continue
        
        # For each cluster, find the best entry point from our current path
        last_point = path[-1]
        best_dist = float('inf')
        best_idx = None
        
        for idx in cluster_points[cluster_idx]:
            dist = np.sum((points[last_point] - points[idx])**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        # Connect to the best entry point
        if best_idx is not None:
            path.append(best_idx)
            cluster_points[cluster_idx].remove(best_idx)
            
            # Add remaining points in this cluster using nearest neighbor
            while cluster_points[cluster_idx]:
                last_point = path[-1]
                nearest = min(cluster_points[cluster_idx], 
                              key=lambda idx: np.sum((points[last_point] - points[idx])**2))
                path.append(nearest)
                cluster_points[cluster_idx].remove(nearest)
    
    # Close the loop
    if len(path) > 1:
        path.append(path[0])
        
        # Convert path to edges
        edges = np.array([[path[i], path[i+1]] for i in range(len(path)-1)])
        return edges
    else:
        return np.array([])