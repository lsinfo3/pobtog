from collections import deque
import numpy as np
import geopandas as gpd

def bfs_traverse_tree(root):
    result = []
    queue = deque([(root, 0, 0, [])])  # (node, depth, parent_id, ancestors)
    id_counter = 0
    max_depth = 0

    # First pass to determine the maximum depth of the tree
    temp_queue = deque([(root, 0)])
    while temp_queue:
        node, depth = temp_queue.popleft()
        max_depth = max(max_depth, depth)
        if not node.is_leaf:
            temp_queue.append((node.child1, depth + 1))
            temp_queue.append((node.child2, depth + 1))

    # Second pass to traverse and collect results
    while queue:
        node, depth, parent_id, ancestors = queue.popleft()

        # Assign ID to current node
        id_counter += 1
        node_id = id_counter

        # Extract centroid coordinates
        centroid = node.centroid
        if centroid is None:
            continue

        # Pad ancestors list with np.nan to match max_depth
        padded_ancestors = np.full(max_depth, np.nan)
        padded_ancestors[:len(ancestors)] = ancestors

        # Append node information to result
        result.append((node_id, parent_id, *centroid, depth, node.total_weight, node.is_leaf, node.geo_boundary(), *padded_ancestors))

        # If not a leaf node, add children to the queue
        if not node.is_leaf:
            new_ancestors = ancestors + [node_id]
            queue.append((node.child1, depth + 1, node_id, new_ancestors))
            queue.append((node.child2, depth + 1, node_id, new_ancestors))

    return result

def dualtree_to_hierarchy(tree):
    h_centers = bfs_traverse_tree(tree)

    df = gpd.GeoDataFrame(np.array(h_centers), 
                      columns=['id', 'parent_id', 'lon', 'lat', 'level', 'pop', 'is_leaf', 'bounds', 
                               *[f'ancestor{i}' for i in range(len(h_centers[0])-8)]],
                               geometry='bounds',
                               crs='EPSG:4326')

    return df 