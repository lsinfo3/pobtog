// This file contains the JavaScript code that handles user interactions on the client side, such as fetching the list of countries, submitting the form, and updating the plot dynamically.

document.addEventListener('DOMContentLoaded', function() {
    const countrySelect = document.getElementById('country');
    const capacityInput = document.getElementById('capacity');
    const thresholdInput = document.getElementById('population-threshold');
    const form = document.getElementById('data-form');
    const terminalOutput = document.getElementById('terminal-output');
    const plotContainer = document.getElementById('plot-container');
    
    // Set default values for capacity and threshold
    capacityInput.value = 20000; // Set default capacity to 1000
    thresholdInput.value = 500; // Set default threshold to 500
    
    // Fetch the list of countries
    fetch('/countries')
        .then(response => response.json())
        .then(countries => {
            countries.forEach(country => {
                const option = document.createElement('option');
                option.value = country;
                option.textContent = country;
                countrySelect.appendChild(option);
                
                // Set a default country (e.g., "USA")
                if (country === "Spain") {
                    option.selected = true;
                }
            });
            
            // If you prefer to select the first country as default instead
            // if (countries.length > 0 && !countrySelect.value) {
            //     countrySelect.value = countries[0];
            // }
        })
        .catch(error => {
            console.error('Error loading countries:', error);
            terminalOutput.textContent = 'Error loading countries. Please refresh the page.';
        });
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const country = countrySelect.value;
        const capacity = capacityInput.value;
        const threshold = thresholdInput.value;
        
        if (!country || !capacity || !threshold) {
            terminalOutput.textContent = 'Please fill in all fields.';
            return;
        }
        
        // Clear previous output and plot
        terminalOutput.textContent = 'Processing...';
        plotContainer.innerHTML = '';
        
        // Start processing
        fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                country: country,
                capacity: parseInt(capacity),
                population_threshold: parseInt(threshold)
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                terminalOutput.textContent = 'Error: ' + data.error;
                return;
            }
            
            // Set up event source for streaming output
            const eventSource = new EventSource('/stream');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.output) {
                    // Update to replace the content rather than append
                    terminalOutput.textContent = data.output;
                    terminalOutput.scrollTop = terminalOutput.scrollHeight;
                }
                
                // Check if processing is complete
                if (data.complete) {
                    eventSource.close();
                    
                    // Fetch final result with plot
                    fetch('/result')
                        .then(response => response.json())
                        .then(result => {
                            if (result.plot_path) {
                                // Instead of using Plotly.newPlot, load the HTML directly
                                const plotFrame = document.createElement('iframe');
                                plotFrame.src = result.plot_path;
                                plotFrame.style.width = '100%';
                                plotFrame.style.height = '500px';
                                plotFrame.style.border = 'none';
                                
                                // Clear and append
                                plotContainer.innerHTML = '';
                                plotContainer.appendChild(plotFrame);
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching result:', error);
                            terminalOutput.textContent += '\n\nError fetching final result.';
                        });
                }
            };
            
            eventSource.onerror = function() {
                terminalOutput.textContent += '\n\nError: Stream connection closed.';
                eventSource.close();
            };
        })
        .catch(error => {
            console.error('Error:', error);
            terminalOutput.textContent = 'Error: ' + error.message;
        });
    });
});