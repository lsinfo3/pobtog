import numpy as np
from .weighted_dualtree import Rectangle   # <-- imported Rectangle from weighted_dualtree

class WeightedQuadtree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary  # now an instance of dualtree.Rectangle
        self.capacity = capacity  # maximum total weight allowed in this node
        self.points = np.empty((0, 3))  # each row is (x, y, weight)
        self.total_weight = 0
        self.divided = False
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None

    def subdivide(self):
        # Convert dualtree.Rectangle attributes into quadrant boundaries.
        x_min, x_max = self.boundary.x_min, self.boundary.x_max
        y_min, y_max = self.boundary.y_min, self.boundary.y_max
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Define quadrants (using conventional Cartesian coordinates,
        # where y increases upward)
        ne = Rectangle(center_x, x_max, center_y, y_max)   # Northeast
        nw = Rectangle(x_min, center_x, center_y, y_max)      # Northwest
        se = Rectangle(center_x, x_max, y_min, center_y)      # Southeast
        sw = Rectangle(x_min, center_x, y_min, center_y)      # Southwest

        self.northeast = WeightedQuadtree(ne, self.capacity)
        self.northwest = WeightedQuadtree(nw, self.capacity)
        self.southeast = WeightedQuadtree(se, self.capacity)
        self.southwest = WeightedQuadtree(sw, self.capacity)
        self.divided = True

    def insert(self, point):
        # point is a tuple (x, y, weight)
        x, y, weight = point

        # Use the dualtree Rectangle's method "containes" (it accepts x and y separately)
        if not self.boundary.containes(x, y):
            return False

        if not self.divided:
            if self.total_weight + weight <= self.capacity:
                self.points = np.vstack([self.points, [x, y, weight]])
                self.total_weight += weight
                return True
            else:
                self.subdivide()
                for p in self.points:
                    self._insert_into_children(p)
                self.points = np.empty((0, 3))

        inserted = self._insert_into_children(point)
        if inserted:
            self.total_weight += weight
        return inserted

    def _insert_into_children(self, point):
        if self.northeast.boundary.containes(point[0], point[1]):
            return self.northeast.insert(point)
        elif self.northwest.boundary.containes(point[0], point[1]):
            return self.northwest.insert(point)
        elif self.southeast.boundary.containes(point[0], point[1]):
            return self.southeast.insert(point)
        elif self.southwest.boundary.containes(point[0], point[1]):
            return self.southwest.insert(point)
        return False

    def query(self, range_rect, found=None):
        # Return all points within the given range (an instance of dualtree.Rectangle).
        if found is None:
            found = []
        if not self.boundary.intersects(range_rect):
            return found

        if not self.divided:
            for p in self.points:
                if range_rect.containes(p[0], p[1]):
                    found.append(p)
        else:
            self.northwest.query(range_rect, found)
            self.northeast.query(range_rect, found)
            self.southwest.query(range_rect, found)
            self.southeast.query(range_rect, found)
        return found

    def __repr__(self, level=0):
        max_depth = 3
        indent = "\t" * level
        ret = f"{indent}WeightedQuadtree(total_weight={self.total_weight}, capacity={self.capacity})\n"
        if level == max_depth or not self.divided:
            ret += f"{indent}\tPoints: {len(self.points)}\n"
        if self.divided:
            ret += self.northwest.__repr__(level + 1)
            ret += self.northeast.__repr__(level + 1)
            ret += self.southwest.__repr__(level + 1)
            ret += self.southeast.__repr__(level + 1)
        return ret

    def __iter__(self):
        if not self.divided:
            yield self
        else:
            yield from self.northwest
            yield from self.northeast
            yield from self.southwest
            yield from self.southeast

    @staticmethod
    def build_from_arrays(boundary, capacity, xs, ys, weights):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        weights = np.asarray(weights)
        node = WeightedQuadtree(boundary, capacity)

        total = weights.sum() if len(weights) > 0 else 0
        if len(xs) == 0:
            node.points = np.empty((0, 3))
            return node
        if len(xs) == 1 or total <= capacity:
            node.points = np.column_stack((xs, ys, weights))
            node.total_weight = total
            return node

        node.subdivide()
        x_mid = (boundary.x_min + boundary.x_max) / 2
        y_mid = (boundary.y_min + boundary.y_max) / 2

        mask_ne = (xs >= x_mid) & (ys >= y_mid)
        mask_nw = (xs <  x_mid) & (ys >= y_mid)
        mask_se = (xs >= x_mid) & (ys <  y_mid)
        mask_sw = (xs <  x_mid) & (ys <  y_mid)

        node.northeast = WeightedQuadtree.build_from_arrays(node.northeast.boundary,
                                                            capacity,
                                                            xs[mask_ne],
                                                            ys[mask_ne],
                                                            weights[mask_ne])
        node.northwest = WeightedQuadtree.build_from_arrays(node.northwest.boundary,
                                                            capacity,
                                                            xs[mask_nw],
                                                            ys[mask_nw],
                                                            weights[mask_nw])
        node.southeast = WeightedQuadtree.build_from_arrays(node.southeast.boundary,
                                                            capacity,
                                                            xs[mask_se],
                                                            ys[mask_se],
                                                            weights[mask_se])
        node.southwest = WeightedQuadtree.build_from_arrays(node.southwest.boundary,
                                                            capacity,
                                                            xs[mask_sw],
                                                            ys[mask_sw],
                                                            weights[mask_sw])
        node.total_weight = (node.northeast.total_weight +
                             node.northwest.total_weight +
                             node.southeast.total_weight +
                             node.southwest.total_weight)
        return node

    @property
    def centroid(self):
        return np.mean(self.points[:, :2], axis=0)


if __name__ == "__main__":
    # Create a boundary representing a square centered at (0, 0) with limits -100 to 100.
    boundary = Rectangle(-100, 100, -100, 100)
    capacity = 5  # maximum cumulative weight per node
    tree_iter = WeightedQuadtree(boundary, capacity)

    sample_points = [
        (10, 10, 2),
        (-10, -10, 3),
        (20, -20, 4),
        (-20, 20, 1),
        (5, 5, 5),
    ]

    for pt in sample_points:
        success = tree_iter.insert(pt)
        print(f"Inserting point {pt} - Success: {success}")

    print("\nQuadtree structure (iterative insertion):")
    print(tree_iter)

    xs = np.array([10, -10, 20, -20, 5])
    ys = np.array([10, -10, -20, 20, 5])
    weights = np.array([2, 3, 4, 1, 5])

    tree_bulk = WeightedQuadtree.build_from_arrays(boundary, capacity, xs, ys, weights)
    print("\nQuadtree structure (bulk build):")
    print(tree_bulk)