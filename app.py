from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging
import io

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("simulation_log.log"),
    logging.StreamHandler()
])

# Constants
R = 8.314  # Gas constant in J/(mol*K)
Tref = 278.15  # Reference temperature (5°C) in Kelvin

def simulate_synthetic_data(time_points, temperatures_C, Ea_mean, Ea_std, k_ref_mean, k_ref_std, initial_value, noise_level):
    temperatures_K = np.array(temperatures_C) + 273.15
    Ea_values = np.random.normal(Ea_mean, Ea_std, len(temperatures_C))
    k_ref_values = np.random.normal(k_ref_mean, k_ref_std, len(temperatures_C))
    k_values = k_ref_values * np.exp(-Ea_values / R * (1 / temperatures_K - 1 / Tref))
    synthetic_data = initial_value * np.exp(-np.outer(k_values, time_points)) + norm.rvs(scale=noise_level, size=(len(temperatures_C), len(time_points)))
    return synthetic_data

def predict_stability(fit_params, time_points, final_month, temperatures_C, initial_value):
    prediction_points = np.linspace(time_points.max(), final_month, 100)
    predictions = {}
    
    for temp in np.unique(temperatures_C):
        temp_K = temp + 273.15
        params = fit_params[temp]
        predictions[temp] = arrhenius_model(prediction_points, params[0], params[1], temp_K, initial_value)
    
    return prediction_points, predictions

def arrhenius_model(t, Ea, k_ref, temp_K, initial_value):
    k = k_ref * np.exp(-Ea / R * (1 / temp_K - 1 / Tref))
    return initial_value * np.exp(-k * t)

def fit_model(time_points, degradation, temperatures_C, n_simulations, initial_value, noise_level):
    unique_temperatures = np.unique(temperatures_C)
    fit_params = {}
    predictions = {}

    for temp in unique_temperatures:
        temp_idx = np.where(temperatures_C == temp)[0]
        temp_degradation = degradation[temp_idx].flatten()
        temp_K = temp + 273.15

        try:
            popt, pcov = curve_fit(lambda t, Ea, k_ref: arrhenius_model(t, Ea, k_ref, temp_K, initial_value), time_points, temp_degradation[:len(time_points)], method='dogbox')
            fit_params[temp] = popt
            logging.info(f'Fitted parameters for {temp}°C: Ea = {popt[0]}, k_ref = {popt[1]}')
        except RuntimeError as e:
            logging.error(f'Error fitting model for {temp}°C: {e}')
            continue

        temp_predictions = np.zeros((n_simulations, len(time_points)))

        for i in range(n_simulations):
            perturbed_degradation = temp_degradation[:len(time_points)] + norm.rvs(scale=noise_level, size=len(time_points))
            try:
                popt_perturbed, _ = curve_fit(lambda t, Ea, k_ref: arrhenius_model(t, Ea, k_ref, temp_K, initial_value), time_points, perturbed_degradation, method='dogbox')
                temp_predictions[i, :] = arrhenius_model(time_points, *popt_perturbed, temp_K, initial_value)
            except RuntimeError as e:
                logging.error(f'Error in Monte-Carlo simulation {i} for {temp}°C: {e}')
                temp_predictions[i, :] = np.nan

        predictions[temp] = temp_predictions

    return fit_params, predictions

def calculate_prediction_intervals(predictions):
    return {temp: (np.nanpercentile(pred, 2.5, axis=0), np.nanpercentile(pred, 97.5, axis=0)) for temp, pred in predictions.items()}

def plot_results(time_points, degradation, fit_params, prediction_intervals, temperatures_C, initial_value, prediction_points, predicted_stability):
    unique_temperatures = np.unique(temperatures_C)

    plt.figure(figsize=(10, 6))
    for temp in unique_temperatures:
        temp_idx = np.where(temperatures_C == temp)[0]
        temp_degradation = degradation[temp_idx].flatten()
        plt.scatter(np.tile(time_points, len(temp_idx)), temp_degradation, label=f'Experimental data {temp}°C', s=10)

        unique_time_points = np.unique(time_points)
        fitted_values = arrhenius_model(unique_time_points, *fit_params[temp], temp + 273.15, initial_value)
        lower_bounds, upper_bounds = prediction_intervals[temp]
        plt.plot(unique_time_points, fitted_values, label=f'Fitted model {temp}°C')
        plt.fill_between(unique_time_points, lower_bounds, upper_bounds, alpha=0.2, label=f'95% prediction interval {temp}°C')
        
        # Plot predicted stability
        plt.plot(prediction_points, predicted_stability[temp], '--', label=f'Predicted stability {temp}°C')

    plt.xlabel('Time (months)')
    plt.ylabel('CEX sum of acid variants (%)')
    plt.title('Kinetic Model Fitting with Prediction Intervals and Stability Prediction')
    plt.legend()
    plt.grid(True)

    plot_path = 'static/plot.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get form data
        time_points = np.array([float(x) for x in request.form['time_points'].split(',')])
        temperatures_C = [
            float(request.form['temperature1_C']),
            float(request.form['temperature2_C']),
            float(request.form['temperature3_C'])
        ]
        final_month = float(request.form['final_month'])
        Ea_mean = float(request.form['Ea_mean'])
        Ea_std = float(request.form['Ea_std'])
        k_ref_mean = float(request.form['k_ref_mean'])
        k_ref_std = float(request.form['k_ref_std'])
        initial_value = float(request.form['initial_value'])
        noise_level = float(request.form['noise_level'])
        n_simulations = int(request.form['n_simulations'])

        # Generate synthetic data
        synthetic_data = simulate_synthetic_data(time_points, temperatures_C, Ea_mean, Ea_std, k_ref_mean, k_ref_std, initial_value, noise_level)

        # Fit model to synthetic data
        fit_params, predictions = fit_model(time_points, synthetic_data, temperatures_C, n_simulations, initial_value, noise_level)

        # Calculate prediction intervals
        prediction_intervals = calculate_prediction_intervals(predictions)

        # Predict stability up to the final month
        prediction_points, predicted_stability = predict_stability(fit_params, time_points, final_month, temperatures_C, initial_value)

        # Plot results
        plot_path = plot_results(time_points, synthetic_data, fit_params, prediction_intervals, temperatures_C, initial_value, prediction_points, predicted_stability)

        # Prepare data for export
        time_grid, temp_grid = np.meshgrid(time_points, temperatures_C)
        time_list = time_grid.flatten()
        temp_list = temp_grid.flatten()
        degradation_list = synthetic_data.flatten()

        # Create DataFrame
        data = {
            'Time (months)': time_list,
            'Temperature (C)': temp_list,
            'CEX sum of acid variants (%)': degradation_list,
        }
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = 'static/synthetic_data.csv'
        df.to_csv(csv_path, index=False)

        return render_template('result.html', plot_url=plot_path)

    except Exception as e:
        logging.error(f'Error generating synthetic data: {e}')
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)