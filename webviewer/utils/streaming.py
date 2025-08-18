from flask import stream_with_context, Response
import sys
import io
import os
from pathlib import Path

class StreamToLogger(object):
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        self.logger(message)

    def flush(self):
        pass

def stream_terminal_output(logger):
    stream = StreamToLogger(logger)
    sys.stdout = stream
    sys.stderr = stream

    yield "Streaming terminal output...\n"
    while True:
        # This is a placeholder for actual output streaming logic
        # You can implement a way to capture and yield output from your functions
        yield "Output from terminal...\n"  # Replace with actual output
        # Add a sleep or break condition as needed

def generate_output_stream(logger):
    return Response(stream_with_context(stream_terminal_output(logger)), mimetype='text/event-stream')

def generate_plot(data):
    # ... your existing plot creation code ...
    
    # Instead of returning plot as JSON
    # fig.write_html to a file in static folder
    plots_dir = Path("static/plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Create unique filename or use session ID
    plot_filename = f"plot_{int(time.time())}.html"
    plot_path = plots_dir / plot_filename
    
    # Write full HTML with all the interactive features
    fig.write_html(plot_path, include_plotlyjs='cdn', full_html=False)
    
    # Return the path to the HTML file
    return {'plot_path': f'/static/plots/{plot_filename}'}