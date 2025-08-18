import numpy as np
from copy import copy


def mean(*x):
    return sum(x) / len(x)

class Rectangle:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max 
        self.y_min = y_min
        self.y_max = y_max

    @classmethod
    def from_points(cls, x):
        return cls(x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max())
    
    def containes(self, x, y):
        if x > self.x_max:
            return False
        if x < self.x_min:
            return False
        if y > self.y_max:
            return False
        if y < self.y_min:
            return False 
        return True 
    
    def longest_axis(self) -> int:
        if abs(self.x_max - self.x_min) < abs(self.y_max - self.y_min):
            return 1
        return 0
    
    @property
    def midpoint_x(self):
        return (self.x_max + self.x_min) / 2

    @property
    def midpoint_y(self):
        return (self.y_max + self.y_min) / 2
    
    def __copy__(self):
        return __class__(self.x_min, self.x_max, self.y_min, self.y_max)

class DualTree:
    def __init__(self, boundary: Rectangle, points, weights, capacity, max_div_iter=5, indices=None):
        self.boundary = boundary
        self.points = points 
        self.weights = weights
        self.capacity = capacity
        self.total_weight = np.sum(self.weights)
        self.max_div_iter = max_div_iter
        self.child1 = None
        self.child2 = None
        self.is_leaf = False
        self.tol = 0.05
        # store original indices for cluster mapping
        if indices is None:
            self.indices = np.arange(len(points))
        else:
            self.indices = indices

    def create_child(self, boundary, points, weights, indices):
        return __class__(boundary, points, weights, self.capacity, self.max_div_iter, indices)

    def __iter__(self):
        if self.is_leaf:
            yield self
        else:
            yield from self.child1
            yield from self.child2
    
    @classmethod
    def build_from_df(cls, df, return_tree=False, assign_clusters=False, **kwargs):
        pts = df[['lon', 'lat']].values
        bounds = Rectangle.from_points(pts)
        # pass indices as np.arange(len(df))
        tree = DualTree(bounds, pts, df['pop'].values, indices=np.arange(len(df)), **kwargs)
        tree = DualTree.build(tree)
        centers = np.array([[*leaf.centroid, leaf.total_weight] for leaf in tree if len(leaf.points) > 0])
        if not assign_clusters:
            return centers 
        cluster_ids = tree.assign_clusters()
        if return_tree:
            return centers, cluster_ids, tree 
        return centers, cluster_ids

    @staticmethod
    def build(node):
        if node.total_weight < node.capacity or len(node.points) <= 1:
            node.child1 = None
            node.child2 = None
            node.is_leaf = True 
            return node
                
        longest_axis = node.boundary.longest_axis()
        thresh = np.average(node.points[:, longest_axis], weights=node.weights)

        if longest_axis == 0:
            rect_above = copy(node.boundary)
            rect_above.x_min = thresh
            rect_below = copy(node.boundary)
            rect_below.x_max = thresh
        else: 
            rect_above = copy(node.boundary)
            rect_above.y_min = thresh
            rect_below = copy(node.boundary)
            rect_below.y_max = thresh
        
        mask_above = node.points[:, longest_axis] > thresh

        node.child1 = node.create_child(rect_above, node.points[mask_above],
                                        node.weights[mask_above],
                                        node.indices[mask_above])
        node.child2 = node.create_child(rect_below, node.points[~mask_above],
                                        node.weights[~mask_above],
                                        node.indices[~mask_above])
        
        DualTree.build(node.child1)
        DualTree.build(node.child2)
        return node

    @property
    def centroid(self):
        return np.mean(self.points, axis=0)
    
    def assign_clusters(self):
        """Returns an integer array of cluster ids for the original points.
           Each leaf is assigned a unique id.
        """
        clusters = np.empty(np.max(self.indices) + 1, dtype=int)
        for cluster_id, leaf in enumerate(self):
            clusters[leaf.indices] = cluster_id
        return clusters

if __name__ == '__main__': 
    N = 1000
    points = np.random.multivariate_normal(mean=[10, 10], cov=np.array([[5, 3],
                                                                         [3, 5]]),
                                           size=N)
    # weights = np.random.exponential(size=N)
    weights = np.ones(N)
    
    tree = DualTree(Rectangle.from_points(points), points, weights, capacity=100)
    tree = DualTree.build(tree)

    clusters = tree.assign_clusters()
    print("Cluster assignment:", clusters)

    plt.scatter(points[:, 0], points[:, 1], c=clusters, cmap='viridis')
    plt.show()



















