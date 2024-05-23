# Synthetic Data Generator for Protein Stability

This project is a Flask-based web application for generating synthetic stability data for monoclonal antibodies using kinetic modeling. The application allows users to enter specific parameters and generate synthetic data along with prediction intervals, which are then plotted and displayed in a new browser window. The generated data can also be downloaded as a CSV file.

For context of mAb stability, see an example in Kuzman, D., Bunc, M., Ravnik, M. et al. Long-term stability predictions of therapeutic monoclonal antibodies in solution using Arrhenius-based kinetics. Sci Rep 11, 20534 (2021). https://doi.org/10.1038/s41598-021-99875-9

### Note: this code is not related or owned by any institution. No real laboratory data was used.  

## Features

- User-friendly web interface for entering model parameters
- Generates synthetic stability data based on user input
- Fits kinetic model to the generated data
- Calculates prediction intervals using Monte Carlo simulations
- Plots experimental data, fitted model, and predicted stability
- Allows users to download the generated synthetic data as a CSV file

## Project Structure

/your_project_directory
/templates
index.html
result.html
/static
script.js
style.css
app.py
requirements.txt

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- A web browser with pop-ups enabled

File Descriptions
app.py: Main application file containing the Flask routes and logic for generating synthetic data, fitting the model, and plotting results.
index.html: HTML template for the home page with the form for entering parameters.
result.html: HTML template for displaying the generated plot.
script.js: JavaScript file for handling pop-up windows.
style.css: CSS file for styling the HTML templates.
requirements.txt: List of Python dependencies required for the project.