import os
import glob
import netCDF4 as nc
import pandas as pd
import numpy as np

def get_lat_lon_bounds(file_path):
    """Extracts the min and max latitude and longitude from a NetCDF file."""
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            # Adjust variable names if necessary (e.g., 'latitude' or 'lat')
            latitudes = dataset.variables['lat'][:]
            longitudes = dataset.variables['lon'][:]
            return {
                "min_lat": np.min(latitudes),
                "max_lat": np.max(latitudes),
                "min_lon": np.min(longitudes),
                "max_lon": np.max(longitudes),
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_files_in_directory(directory, file_pattern="*.nc"):
    """Processes all .nc files in a directory."""
    results = []
    files = glob.glob(os.path.join(directory, file_pattern))
    for file_path in files:
        print(f"Processing file: {file_path}")
        bounds = get_lat_lon_bounds(file_path)
        if bounds:
            bounds["file"] = os.path.basename(file_path)
            bounds["directory"] = directory
            results.append(bounds)
    return results

def main():
    # Directories containing the .nc files
    directories = [
        r"C:\Users\psteeves\coastal\ocean_tide",
        r"C:\Users\psteeves\coastal\load_tide"
    ]
    
    all_results = []
    for directory in directories:
        print(f"Processing directory: {directory}")
        results = process_files_in_directory(directory)
        all_results.extend(results)

    # Save results to a CSV file
    df = pd.DataFrame(all_results)
    output_csv = r"C:\Users\psteeves\coastal\bounding_boxes_orig.csv"
    df.to_csv(output_csv, index=False)
    print(f"Bounding boxes saved to {output_csv}")

if __name__ == "__main__":
    main()
