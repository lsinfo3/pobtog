from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import json
import os
import io
import sys
import pandas as pd
import plotly
import plotly.graph_objects as go
from queue import Queue
import threading
import time
import numpy as np
from contextlib import redirect_stdout
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Fix import paths - import directly from the prep_world module
from prep_world import load_country_raster_splitted, dualtree_on_rasters, print_combined_raster_stats, load_country_raster_splitted_mp
from prep_world import plot_centroids_from_df

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Queue for storing terminal output
output_queue = Queue()
# Flag for tracking if processing is complete
processing_complete = threading.Event()

# Add global variable to store plot data
current_plot_data = None

def capture_output(func, *args, **kwargs):
    """Capture stdout from a function and put it in the queue"""
    global current_plot_data
    output_buffer = io.StringIO()
    result = None
    
    # Redirect stdout to our buffer
    with redirect_stdout(output_buffer):
        try:
            result = func(*args, **kwargs)
            # Store the result in the global variable
            current_plot_data = result
        except Exception as e:
            print(f"ERROR: {str(e)}")
            logger.exception("Error in processing")
    
    # Put captured output in the queue
    output_text = output_buffer.getvalue()
    output_queue.put(output_text)
    
    return result

def process_data(country, capacity, population_threshold):
    """Process the data and capture all output"""
    output_queue.queue.clear()
    processing_complete.clear()
    
    print(f"Processing data for {country}")
    print(f"Capacity: {capacity}")
    print(f"Population threshold: {population_threshold}")
    
    # Step 1: Load country data
    rasters, transforms = load_country_raster_splitted(country, population_tresh=population_threshold)
    # rasters, transforms = load_country_raster_splitted_mp(country, population_tresh=population_threshold)
    print(f"Loaded {len(rasters)} raster(s) for {country}")
    
    # Step 2: Print raster statistics
    print_combined_raster_stats(rasters)
    
    # Step 3: Process with dualtree
    print(f"Running dualtree with capacity={capacity}...")
    centers = dualtree_on_rasters(rasters, transforms, capacity=capacity)
    print(f"Generated {len(centers)} data points")
    
    # Step 4: Create plot
    print("Creating plot...")
    fig = plot_centroids_from_df(centers, scale=5)
    
    # Create a directory for plots if it doesn't exist
    plots_dir = os.path.join(app.static_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate a unique filename
    plot_filename = f"plot_{country}_{int(time.time())}.html"
    plot_path = os.path.join(plots_dir, plot_filename)
    
    # Write the figure to an HTML file
    fig.write_html(plot_path, include_plotlyjs='cdn', full_html=False)
    
    # Return the URL path to the plot file instead of JSON data
    relative_path = f"/static/plots/{plot_filename}"
    print(f"Plot saved to: {relative_path}")
    
    processing_complete.set()
    return relative_path

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/countries')
def get_countries():
    """Return the list of countries"""
    try:
        # Use absolute path based on the app.py location
        countries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'countries.json')
        with open(countries_path, 'r') as f:
            countries_dict = json.load(f)
            # Extract the keys (country names) from the dictionary
            country_names = list(countries_dict.keys())
            # Sort the country names alphabetically for better user experience
            country_names.sort()
        return jsonify(country_names)
    except Exception as e:
        msg = f"Error loading countries.json: {str(e)}"
        logger.exception(msg)
        return jsonify([msg]), 500

@app.route('/process', methods=['POST'])
def start_processing():
    """Start the data processing in a separate thread"""
    try:
        data = request.json
        country = data.get('country')
        capacity = int(data.get('capacity'))
        population_threshold = int(data.get('population_threshold'))
        
        # Clear previous data
        output_queue.queue.clear()
        processing_complete.clear()
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=capture_output,
            args=(process_data, country, capacity, population_threshold)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "processing"})
    except Exception as e:
        logger.exception("Error starting processing")
        return jsonify({"error": str(e)}), 500

@app.route('/stream')
def stream():
    """Stream the output as server-sent events"""
    def generate():
        last_output = ""
        
        # Continue until processing is complete and queue is empty
        while not (processing_complete.is_set() and output_queue.empty()):
            # Check if new output is available
            try:
                output = ""
                while not output_queue.empty():
                    new_output = output_queue.get_nowait()
                    output += new_output
                
                if output and output != last_output:
                    last_output = output
                    yield f"data: {json.dumps({'output': output})}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
            time.sleep(0.1)
        
        # Send completion message
        if processing_complete.is_set():
            yield f"data: {json.dumps({'complete': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/result')
def get_result():
    """Get the final result with the plot data"""
    global current_plot_data
    
    if not processing_complete.is_set():
        return jsonify({"error": "Processing not complete"}), 400
    
    # Get all remaining output
    output = ""
    while not output_queue.empty():
        output += output_queue.get()
    
    # Check if plot data is available
    if current_plot_data:
        # Return both the terminal output and the path to the plot HTML file
        return jsonify({
            "output": output,
            "plot_path": current_plot_data  # This should now be a path instead of JSON data
        })
    else:
        return jsonify({
            "output": output,
            "error": "No plot data generated"
        })

if __name__ == '__main__':
    # Get port from environment or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Run the app, accessible from any machine
    app.run(host='0.0.0.0', port=port, debug=True)

