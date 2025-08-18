# Geo Population Viewer

## Overview
Geo Population Viewer is a web application that allows users to visualize population data for different countries. Users can select a country, input a population threshold and a capacity number, and view the results as a plot. The application utilizes Flask for the backend and Plotly for dynamic plotting.

## Features
- Dropdown selection of countries sourced from `data/countries.json`.
- Input fields for capacity and population threshold.
- Real-time plotting of population data.
- Live terminal output streaming to the web interface.

## Project Structure
```
geo-population-viewer
├── app.py                # Main entry point of the application
├── static
│   ├── css
│   │   └── style.css     # CSS styles for the web application
│   └── js
│       └── main.js       # JavaScript for client-side interactions
├── templates
│   └── index.html        # HTML template for the user interface
├── utils
│   ├── __init__.py       # Marks the utils directory as a Python package
│   ├── geo_processing.py  # Functions for processing geographical data
│   └── streaming.py       # Functions for streaming terminal output
├── data
│   └── countries.json     # JSON file containing country names
├── requirements.txt       # Lists dependencies for the project
├── Dockerfile             # Instructions for building a Docker image
└── README.md              # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd geo-population-viewer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application in your web browser at `http://localhost:5000`.

## Usage
- Select a country from the dropdown list.
- Enter the desired capacity and population threshold.
- Click the submit button to generate the plot.
- View live terminal outputs for processing status.

## License
This project is licensed under the MIT License. See the LICENSE file for details.