import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def format_big_number(n):
    try: 
        return [f"{int(round(c)):_}" for c in n]
    except TypeError:
        return f"{int(round(n)):_}"

def weighted_median(values, weights, total_sum=None):
    """
    Compute the weighted median of an array of values.
    
    This implementation sorts values and computes the cumulative
    sum of the weights. The weighted median is the smallest value for
    which the cumulative sum is greater than or equal to half of the
    total sum of weights.
    Parameters
    ----------
    values : array-like
        List or array of values on which to calculate the weighted median.
    weights : array-like
        List or array of weights corresponding to the values.
    Returns
    -------
    float
        The weighted median of the input values.
    """
    
    # Get the indices that would sort the array
    sort_indices = np.argsort(values)
    
    # Sort values and weights according to the sorted indices
    values_sorted = values[sort_indices]
    weights_sorted = weights[sort_indices]  

    # Compute the cumulative sum of the sorted weights
    cumsum = weights_sorted.cumsum()
    
    # Calculate the cutoff as half of the total weight sum
    if total_sum is None:
        total_sum = weights_sorted.sum()
    cutoff = total_sum / 2.
    
    # Return the smallest value for which the cumulative sum is greater
    # than or equal to the cutoff
    return values_sorted[cumsum >= cutoff][0]

class PerformanceTest:
    """Context manager for measuring execution time and memory consumption."""
    
    def __init__(self, description="Performance test"):
        self.description = description
        
    def __enter__(self):
        from time import perf_counter
        self.start_time = perf_counter()
        
        # Try to use psutil for accurate memory measurement
        try:
            import psutil
            import os
            self.process = psutil.Process(os.getpid())
            self.start_memory = self.process.memory_info().rss / (1024 * 1024 * 1024)  # GB
            self.use_psutil = True
        except ImportError:
            # Fallback to resource module on Linux
            try:
                import resource
                self.start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # GB
                self.use_psutil = False
            except (ImportError, AttributeError):
                print("Warning: Could not measure memory usage (install psutil package)")
                self.start_memory = 0
                self.use_psutil = False
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from time import perf_counter
        end_time = perf_counter()
        
        # Get final memory usage
        if hasattr(self, 'use_psutil') and self.use_psutil:
            end_memory = self.process.memory_info().rss / (1024 * 1024 * 1024)  # GB
        else:
            try:
                import resource
                end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # GB
            except (ImportError, AttributeError):
                end_memory = 0
        
        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        print(f"\n--- {self.description} ---")
        print(f"Execution time: {execution_time:.4f} seconds")
        if end_memory > 0:
            print(f"Memory usage: {end_memory:.4f} GB")
            print(f"Memory increase: {memory_delta:.4f} GB")
        print("-" * (len(self.description) + 8))

def block_concat(block):
    return np.concatenate([np.concatenate(row, axis=1) for row in block], axis=0)


def print_combined_raster_stats(rasters):
    """
    Print combined statistics for a list of raster arrays using a single print statement.
    
    Parameters:
    -----------
    rasters : list
        List of numpy arrays representing rasters
    """
    # Calculate combined statistics
    pops = sorted([raster.sum() for raster in rasters], reverse=True)
    total_population = sum(pops)
    total_elements = sum(raster.size for raster in rasters)
    shapes = sorted([raster.shape for raster in rasters], key=lambda x: x[0] * x[1], reverse=True)
    total_nonzero = sum(np.count_nonzero(raster) for raster in rasters)
    total_ram_gb = sum(raster.nbytes for raster in rasters) / 1e9
    
    # Create a single string with all statistics
    stats = (
        f"Population: {format_big_number(total_population)}\n"
        f"populations: {format_big_number(pops)}"
        f"array size: {format_big_number(total_elements)}\n"
        f"array shapes: {shapes}\n"
        f"nonzero points: {format_big_number(total_nonzero)}\n"
        f"Ram usage: {total_ram_gb:.2f} GB"
    )
    
    # Print all statistics with a single print call
    print(stats)

