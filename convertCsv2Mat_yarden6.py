import csv
import numpy as np
import matplotlib.pyplot as plt
import wfdb.processing
import scipy.signal as signal
from scipy.fft import fft, fftfreq

CSV_PATH = "Sheba_Real2023/Samples 2024/060624 YV 356.csv"
# CSV_PATH = "Sheba_Real2023/Samples 2024/060824 OZ 355.csv"
# CSV_PATH = "Sheba_Real2023/Samples 2024/180724 GZ 363.csv"


# OUTPUT_MAT_PATH = "Sheba_Real2023/Samples 2024/060624_YV_356.mat"
PLOT_PATH = "output_plot44.png"
OUTPUT_CSV_ABDOMEN = "filtered_abdomen2.csv"
OUTPUT_CSV_THORAX = "filtered_thorax2.csv"
OUTPUT_CSV_BOTH = "filtered_thorax_abdomen_060624_356.csv"
# OUTPUT_CSV_BOTH = "filtered_thorax_abdomen_060824_355.csv"
# OUTPUT_CSV_BOTH = "filtered_thorax_abdomen_180724_363.csv"

def estimate_heart_rate(ecg_signal, fs):
    # Perform FFT on the ECG signal
    n = len(ecg_signal)
    yf = fft(ecg_signal)
    xf = fftfreq(n, 1/fs)
    
    # Consider only the positive frequencies
    idx = np.where(xf > 0)
    xf = xf[idx]
    yf = np.abs(yf[idx])

    # Limit to physiological heart rate frequencies (0.5 - 4 Hz)
    lower_bound = 0.5  # 30 bpm
    upper_bound = 4.0  # 240 bpm
    valid_idx = np.where((xf >= lower_bound) & (xf <= upper_bound))
    xf = xf[valid_idx]
    yf = yf[valid_idx]
    
    # Find the peak frequency in this range
    dominant_freq = xf[np.argmax(yf)]
    
    # Convert frequency to beats per minute (bpm)
    heart_rate = dominant_freq * 60
    
    return heart_rate

def fit_polynomial_segments(ecg_samples, peaks, degree=3):
    polynomials = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        segment = ecg_samples[start_idx:end_idx]
        x = np.arange(start_idx, end_idx)
        
        # Fit a polynomial to the segment
        coeffs = np.polyfit(x, segment, degree)
        polynomial = np.poly1d(coeffs)
        polynomials.append((x, polynomial))
        
    return polynomials

def clean_float(s):
    try:
        return float(s)
    except ValueError:
        return float(s.strip())

if __name__ == "__main__":
    data = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f, delimiter='\t')  # Use tab as the delimiter
        for row in reader:
            rowData = [clean_float(elem) for elem in row if elem.strip()]
            data.append(rowData)
    
    data_array = np.array(data)
    subset_data_array = data_array[:100000, :]
    subset_data_array_thorax = subset_data_array[:, 0]
    subset_data_array_abdomen = subset_data_array[:, 1]
    fs = 1000

    peaks_thorax, thorax_properties = signal.find_peaks(subset_data_array_thorax, distance=fs/2)
    peaks_abdomen, abdomen_properties = signal.find_peaks(subset_data_array_abdomen, distance=fs/2)

   
    # Fit polynomials and compute result arrays
    polynomial_segments_thorax = fit_polynomial_segments(subset_data_array_thorax, peaks_thorax, degree=2)
    polynomial_segments_abdomen = fit_polynomial_segments(subset_data_array_abdomen, peaks_abdomen, degree=2)

    result_array_thorax = np.zeros_like(subset_data_array_thorax)
    result_array_abdomen = np.zeros_like(subset_data_array_abdomen)

    for x, poly in polynomial_segments_thorax:
        result_array_thorax[x] = subset_data_array_thorax[x] - poly(x)

    for x, poly in polynomial_segments_abdomen:
        result_array_abdomen[x] = subset_data_array_abdomen[x] - poly(x)

    # Save the results to a CSV file
    with open(OUTPUT_CSV_BOTH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Thorax', 'Abdomen'])
        for thorax_value, abdomen_value in zip(result_array_thorax, result_array_abdomen):
            writer.writerow([thorax_value, abdomen_value])

    # Plotting the data for thorax
    fig, ax = plt.subplots()
    ax.plot(result_array_thorax[:100000], marker='o')  # Plot with markers for better visibility
    ax.set(title='Result Array Thorax', xlabel='Index', ylabel='Difference')

    # Save the plot as an image
    fig.savefig(PLOT_PATH)
    plt.close(fig)

    print("Done")
