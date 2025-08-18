import json
import sys
import time
from tqdm import tqdm

class WebTqdm(tqdm):
    """
    Custom tqdm subclass that formats progress updates for web display via SSE.
    Each update is sent as a complete status update, not expecting terminal control characters.
    """
    def __init__(self, *args, **kwargs):
        # Set defaults before calling parent constructor
        self.last_print_time = time.time()
        self.min_update_interval = kwargs.get('mininterval', 0.5)
        
        # Disable terminal output features that don't work in web streaming
        kwargs['bar_format'] = '{desc}: {n}/{total} ({percentage:.1f}%) | {rate_fmt} | ETA: {remaining_s:.1f}s'
        kwargs['file'] = sys.stdout
        kwargs['dynamic_ncols'] = False
        kwargs['leave'] = True  # Always leave the progress messages
        
        # Initialize the parent tqdm
        super().__init__(*args, **kwargs)
    
    def display(self, msg=None, pos=None):
        """
        Custom display to ensure each progress update is a complete line.
        """
        if not msg:
            msg = self.format_meter(**self.format_dict)
        
        # Safely check for last_print_time attribute
        current_time = time.time()
        if not hasattr(self, 'last_print_time'):
            self.last_print_time = current_time - self.min_update_interval - 1  # Ensure first update prints
            
        if current_time - self.last_print_time >= self.min_update_interval:
            print(msg)  # Simple print, will be captured by the output queue
            self.last_print_time = current_time

    def close(self):
        """Ensure final progress is shown when completed"""
        try:
            self.display()
        except Exception as e:
            print(f"Error during progress display: {e}")
        finally:
            # Call parent class close method
            super().close()