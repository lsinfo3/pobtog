import numpy as np
import copy
from copy import copy 
import dataclasses
import math
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm
import pandas as pd

from . import wquantiles
from src.utils.geoutils import idx_to_lonlat, idx_to_lonlat_tuple

class _CornerProxy:
    def __init__(self, rect, which):
        self._rect = rect
        self._which = which  # either "lower" or "upper"

    def __getitem__(self, index):
        if self._which == "lower":
            if index == 0:
                return self._rect.x_min
            elif index == 1:
                return self._rect.y_min
        elif self._which == "upper":
            if index == 0:
                return self._rect.x_max
            elif index == 1:
                return self._rect.y_max
        raise IndexError("Index out of range, valid indices are 0 and 1.")

    def __setitem__(self, index, value):
        if self._which == "lower":
            if index == 0:
                self._rect.x_min = value
                return
            elif index == 1:
                self._rect.y_min = value
                return
        elif self._which == "upper":
            if index == 0:
                self._rect.x_max = value
                return
            elif index == 1:
                self._rect.y_max = value
                return
        raise IndexError("Index out of range, valid indices are 0 and 1.")

def aspect_ratio(width, height):
        return max(width, height) / min(width, height)

class Rectangle:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max 
        self.y_min = y_min
        self.y_max = y_max

    @classmethod
    def from_shape(cls, shape):
        # shape is (rows, cols)
        rows, cols = shape
        return cls(0, rows - 1, 0, cols - 1)
    
    def longest_axis(self) -> int:
        # Compare the lengths of the two sides
        if (self.x_max - self.x_min) < (self.y_max - self.y_min):
            return 1
        return 0
    
    def __copy__(self):
        return Rectangle(self.x_min, self.x_max, self.y_min, self.y_max)
    
    @property
    def lower_corner(self):
        return _CornerProxy(self, "lower")

    @property
    def upper_corner(self):
        return _CornerProxy(self, "upper")
    

@dataclasses.dataclass
class TreeParams:
    easy_split : bool
    stop_easy_factor : float
    n_skips : int
    skip_depth : int 
    estimated_branching_factor : int 

@dataclasses.dataclass
class TreeParamsConst:

    transform : rasterio.transform.Affine = None 
    use_median : bool = False
    use_quantile : bool = False
    alpha :float = 0.05

    
class RasterDualTree:
    def __init__(self, boundary: Rectangle, raster, capacity, params: TreeParams, params_const: TreeParamsConst = None, parent=None):
        self.boundary = boundary
        self.original_raster = raster
        self.capacity = capacity
        self.parent = parent 
        self.child1 = None
        self.child2 = None
        self.is_leaf = False
        self.axis_sums = None
        
        # self.axis = self.boundary.longest_axis()
        self.axis = self.longest_axis()

        self.params = dataclasses.replace(params)
        self.params_const = params_const  # Store shared constants
        
    def longest_axis(self):

        axis_0_sums = self.raster.sum(axis=0)
        first_nonzero = np.argmax(axis_0_sums != 0)
        last_nonzero = len(axis_0_sums) - np.argmax(np.flip(axis_0_sums != 0)) - 1
        axis_0_len = last_nonzero - first_nonzero

        axis_1_sums = self.raster.sum(axis=1)
        first_nonzero = np.argmax(axis_1_sums != 0)
        last_nonzero = len(axis_1_sums) - np.argmax(np.flip(axis_1_sums != 0)) - 1
        axis_1_len = last_nonzero - first_nonzero

        if axis_0_len > axis_1_len:
            self.axis_sums = axis_0_sums
            self.axis = 1
        else:
            self.axis_sums = axis_1_sums
            self.axis = 0
        
        if self.boundary.longest_axis() != self.axis:
            # print(axis_0_len, axis_1_len, self.boundary.longest_axis())
            pass
        return self.axis
        


    @staticmethod
    def build(node):


        if node.params.skip_depth > 0:
            node.params.skip_depth -= 1
            easy_split = True 
            node.total_weight = np.inf
            skip = True
        else:
            easy_split = False 
            node.total_weight = node.calculate_weight()
            skip = False 

        if node.total_weight < node.capacity or node.raster.size <= 1:
            node.is_leaf = True
            return node

        relative_weight = node.total_weight / node.capacity

        if not skip and node.params.n_skips >= 1 and node.params.skip_depth <= 0 and relative_weight > node.params.estimated_branching_factor**node.params.n_skips:
            node.params.skip_depth = node.params.n_skips

        # node.params.n_skips = math.floor(math.log(relative_weight, node.params.estimated_branching_factor))
        
        if easy_split:
            thresh = node.raster.shape[node.axis] // 2
        else:
            if node.params_const.use_quantile and node.total_weight > 1.5 * node.capacity and node.total_weight < 8 * node.capacity:

                weighted_avg_left = wquantiles.quantile(np.arange(node.raster.shape[node.axis]), node.axis_sums, node.capacity/node.total_weight + node.params_const.alpha)
                weighted_avg_right = wquantiles.quantile(np.arange(node.raster.shape[node.axis]), node.axis_sums, 1 - (node.capacity/node.total_weight + node.params_const.alpha))

                len_oaxis = node.raster.shape[not node.axis]
                if aspect_ratio(weighted_avg_left, len_oaxis) < aspect_ratio(node.raster.shape[node.axis] - weighted_avg_right, len_oaxis):
                    weighted_avg = weighted_avg_left
                else:
                    weighted_avg = weighted_avg_right
            elif node.params_const.use_median:
                weighted_avg = wquantiles.median(np.arange(node.raster.shape[node.axis]), node.axis_sums)
            else:
                weighted_avg = np.sum(np.arange(node.raster.shape[node.axis]) * node.axis_sums) / node.total_weight
            thresh = int(round(weighted_avg))

        if thresh < 0 or thresh + 1 >= node.raster.shape[node.axis]:
            node.is_leaf = True
            return node
        
        global_thresh = node.boundary.lower_corner[node.axis] + thresh
        
        boundary_child1 = copy(node.boundary)
        boundary_child1.lower_corner[node.axis] = global_thresh + 1
        boundary_child2 = copy(node.boundary)
        boundary_child2.upper_corner[node.axis] = global_thresh
        
        
        node.child1 = RasterDualTree(boundary_child1, node.original_raster, node.capacity, 
                                     node.params, params_const=node.params_const, parent=node)
        node.child2 = RasterDualTree(boundary_child2, node.original_raster, node.capacity, 
                                     node.params, params_const=node.params_const, parent=node)
        RasterDualTree.build(node.child1)
        RasterDualTree.build(node.child2)
        return node

    def calculate_weight(self):
        # Calculate the total weight of the node's region.
        if self.axis_sums is None:
            self.axis_sums = self.raster.sum(axis=int(not self.axis))

        return np.sum(self.axis_sums)
    

    @property 
    def raster(self):
        return self.original_raster[self.boundary.x_min:self.boundary.x_max+1, self.boundary.y_min:self.boundary.y_max+1]
    
    @classmethod
    def build_from_raster(cls, raster, capacity, transform=None, use_median=False, use_quantile=False, assign_clusters=False, return_tree=False, return_geodf=False, easy_split=False, stop_easy_factor=4, n_skips=0, estimated_branching_factor=2, alpha=0.05):
        boundary = Rectangle.from_shape(raster.shape)
        params = TreeParams(easy_split=easy_split, stop_easy_factor=stop_easy_factor, n_skips=n_skips, skip_depth=0, estimated_branching_factor=estimated_branching_factor)
        params_const = TreeParamsConst(transform=transform, use_median=use_median, use_quantile=use_quantile, alpha=alpha)
        tree = cls(boundary, raster, capacity, params, params_const=params_const)
        tree.full_shape = raster.shape
        cls.build(tree)
        
        centers = []
        for leaf in tree:
            centroid = leaf.centroid
            if centroid is None:
                continue
            centers.append([*centroid, leaf.total_weight])
        centers = np.array(centers)


        return_args = [centers]
        if assign_clusters:
            clusters = tree.assign_clusters()
            return_args.append(clusters)
        if return_tree:
            return_args.append(tree)

        if len(return_args) == 1:
            return return_args[0]
        return tuple(return_args)


    @property
    def centroid(self):
        rows = np.arange(self.boundary.x_min, self.boundary.x_max + 1)
        row_weights = self.raster.sum(axis=1)
        if np.all(row_weights == 0):
            return 
        centroid_row = np.average(rows, weights=row_weights)
        cols = np.arange(self.boundary.y_min, self.boundary.y_max + 1)
        col_weights = self.raster.sum(axis=0)
        centroid_col = np.average(cols, weights=col_weights)
        if self.params_const.transform is not None:
            centroid_row, centroid_col = idx_to_lonlat_tuple((centroid_row, centroid_col), self.params_const.transform)
        return (centroid_row, centroid_col)
    
    def geo_centroid(self, accurate=False):
        """
        Calculate the geodesic centroid that properly accounts for Earth's curvature.

        Parameters
        ----------
        accurate : bool, default=False
            If True, calculates a true geodesic centroid using 3D Cartesian coordinates,
            which is more accurate for areas spanning many degrees.
            If False, uses a simple weighted average of latitudes and longitudes.
        
        Returns
        -------
        tuple
            (longitude, latitude) of the geodesic centroid
        """
        if self.params_const is None or self.params_const.transform is None:
            raise ValueError("Transform not set, cannot compute geographic centroid.")
                
        # Get the masked raster with non-zero values
        data = self.raster.copy()
        if np.all(data == 0):
            return None
        
        # Create coordinate meshgrid for our region
        y_idx, x_idx = np.mgrid[self.boundary.x_min:self.boundary.x_max+1,
                               self.boundary.y_min:self.boundary.y_max+1]
        
        # Stack coordinates into (n, 2) array of [row, col] pairs
        points = np.column_stack([y_idx.flatten(), x_idx.flatten()])
        
        # Get weights for each point
        weights = data.flatten()
        mask = weights > 0
        
        # Skip zero-weight points
        if not np.any(mask):
            return None
        
        points = points[mask]
        weights = weights[mask]
        
        # Convert all points to geographic coordinates
        geo_points = idx_to_lonlat(points, transform=self.params_const.transform)
        
        if not accurate:
            # Simple weighted average of longitude and latitude
            lon = np.average(geo_points[:, 0], weights=weights)
            lat = np.average(geo_points[:, 1], weights=weights)
            return (lon, lat)
        else:
            # More accurate geodesic centroid calculation
            # Convert to 3D Cartesian coordinates
            # Earth radius in meters
            R = 6371000.0
            
            # Convert degrees to radians
            lon_rad = np.radians(geo_points[:, 0])
            lat_rad = np.radians(geo_points[:, 1])
            
            # Convert to 3D Cartesian coordinates
            x = R * np.cos(lat_rad) * np.cos(lon_rad)
            y = R * np.cos(lat_rad) * np.sin(lon_rad)
            z = R * np.sin(lat_rad)
            
            # Take weighted average
            avg_x = np.average(x, weights=weights)
            avg_y = np.average(y, weights=weights)
            avg_z = np.average(z, weights=weights)
            
            # Convert back to spherical coordinates
            lon_avg = np.arctan2(avg_y, avg_x)
            hyp = np.sqrt(avg_x**2 + avg_y**2)
            lat_avg = np.arctan2(avg_z, hyp)
            
            # Convert radians to degrees
            lon_avg_deg = np.degrees(lon_avg)
            lat_avg_deg = np.degrees(lat_avg)
            
            return (lon_avg_deg, lat_avg_deg)
    
    def assign_clusters(self):
        # Create an empty array to store cluster ids for every pixel in the original raster.
        clusters = np.empty(self.full_shape, dtype=int)
        def fill(node, cluster_id):
            if node.is_leaf:
                clusters[node.boundary.x_min:node.boundary.x_max+1,
                         node.boundary.y_min:node.boundary.y_max+1] = cluster_id
                return cluster_id + 1
            else:
                cluster_id = fill(node.child1, cluster_id)
                cluster_id = fill(node.child2, cluster_id)
                return cluster_id
        fill(self, 0)
        return clusters

    def __iter__(self):
        if self.is_leaf:
            yield self
        else:
            yield from self.child1
            yield from self.child2
    
    @staticmethod
    def mean_aspect_ratio(tree):
        ratios = []
        for leaf in tree:
            width = leaf.boundary.x_max - leaf.boundary.x_min + 1
            height = leaf.boundary.y_max - leaf.boundary.y_min + 1
            ratio = aspect_ratio(width, height)
            ratios.append(ratio)
        
        return np.mean(ratios)
    
    def __hash__(self):
        return hash((self.boundary.x_min, self.boundary.x_max, self.boundary.y_min, self.boundary.y_max))
    
    @staticmethod
    def assign_child_counts(tree):

        nodes_counts = {}
        def traverse(node):
            if node.is_leaf:
                node.child_count = 0
                return

            traverse(node.child1)
            traverse(node.child2)
            node.child_count = node.child1.child_count + node.child2.child_count

        traverse(tree)
        
        for leaf in tree:
            leaf.child_count = 0
    
    def __len__(self):
        __class__.assign_child_counts(self)
        return self.child_count 

    # def transform_boundaries(self):
    #     if self.params_const is None or self.params_const.transform is None:
    #         raise ValueError("Transform not set, cannot transform boundaries.")
            
    #     # Create a function to recursively transform nodes
    #     def transform_node(node):
    #         # Get corner coordinates
    #         corners = np.array([
    #             [node.boundary.x_min, node.boundary.y_min],  # lower left
    #             [node.boundary.x_min, node.boundary.y_max],  # upper left
    #             [node.boundary.x_max, node.boundary.y_min],  # lower right
    #             [node.boundary.x_max, node.boundary.y_max],  # upper right
    #         ])
            
    #         # Transform corners to geographic coordinates
    #         idx_to_lonlat(corners, self.params_const.transform, inplace=True)
            
    #         # Update the boundary with transformed coordinates
    #         # Note: After transformation, the coordinates may not form a perfect rectangle in lon/lat space
    #         # We're saving the min/max values to maintain the rectangle structure
    #         node.geo_boundary = Rectangle(
    #             x_min=np.min(corners[:, 0]),
    #             x_max=np.max(corners[:, 0]),
    #             y_min=np.min(corners[:, 1]),
    #             y_max=np.max(corners[:, 1])
    #         )
            
    #         # Process children if this is not a leaf
    #         if not node.is_leaf:
    #             transform_node(node.child1)
    #             transform_node(node.child2)
        
    #     # Start the transformation from the root
    #     transform_node(self)

    def geo_boundary(self):
        """
        Return a shapely Polygon representing the geographic boundary of this node.
        
        Returns
        -------
        shapely.geometry.Polygon
            A polygon with the geographic coordinates of the node's boundary corners.
            
        Raises
        ------
        ValueError
            If transform is not set in params_const.
        """
        if self.params_const is None or self.params_const.transform is None:
            raise ValueError("Transform not set, cannot compute geographic boundary.")
        
        # Get the four corners of the rectangle in pixel coordinates (row, col)
        corners = np.array([
            [self.boundary.x_min, self.boundary.y_min],  # lower left
            [self.boundary.x_max, self.boundary.y_min],  # lower right
            [self.boundary.x_max, self.boundary.y_max],  # upper right
            [self.boundary.x_min, self.boundary.y_max],  # upper left
            [self.boundary.x_min, self.boundary.y_min],  # close the polygon
        ])
        
        # Transform corners to geographic coordinates (lon, lat)
        geo_corners = idx_to_lonlat(corners, self.params_const.transform)
        
        # Create a shapely Polygon from the coordinates
        return Polygon(geo_corners)


def dualtree_on_rasters(rasters, transforms, capacity=100_000, easy_split=False, stop_easy_factor=10, n_skips=0, estimated_branching_factor=4, **kwarg):

    coords = []
    # for raster, transform in tqdm(zip(rasters, transforms), total=len(rasters)):
    for raster, transform in tqdm(zip(rasters, transforms), total=len(rasters), desc="Building dual trees"):
        centers = RasterDualTree.build_from_raster(raster, capacity=capacity, easy_split=easy_split, stop_easy_factor=stop_easy_factor, n_skips=n_skips, estimated_branching_factor=estimated_branching_factor, **kwarg)
        idx_to_lonlat(centers, transform, inplace=True)
        coords.append(centers)

    coords = np.concatenate(coords, axis=0)
    coords = pd.DataFrame(coords, columns=['lon', 'lat', 'pop'])

    return coords

if __name__ == '__main__':
    from pyinstrument import Profiler

    # Start the profiler
    profiler = Profiler()
    profiler.start()

    # Your existing code
    import rasterio
    raster = rasterio.open('data/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0_R7_C30.tif').read(1)
    centers = RasterDualTree.build_from_raster(raster, capacity=10_000, easy_split=False, 
                                              stop_easy_factor=4, n_skips=4, 
                                              estimated_branching_factor=5)
    print(len(centers))
    print(centers[:, 2].std() / centers[:, 2].mean())

    # Stop the profiler and output HTML
    profiler.stop()
    
    # Generate and open HTML report in browser
    # profiler.open_in_browser()
    
    # Alternatively, save the HTML to a file
    with open('dualtree_prof.html', 'w') as f:
        f.write(profiler.output_html())