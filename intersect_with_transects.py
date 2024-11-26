import geopandas as gpd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point

# Load transects
transects_path = 'C:/Users/psteeves/coastal/planetscope_coastsat/user_inputs/transects/TUK_TRANSECTS_SMALL_LONG.geojson'
transects = gpd.read_file(transects_path)
print("Transects columns:", transects.columns)
# Directory path containing the shoreline shapefiles
shoreline_dir = 'C:/Users/psteeves/coastal/of_7685/shp/Tuktoyaktuk/coastline'

# Function to calculate intersections and distances along transects
def calculate_intersection_distances(transects, shoreline_file):
    shoreline = gpd.read_file(shoreline_file)
    shoreline = shoreline.to_crs(transects.crs)  # Ensure CRS match
    
    # Perform intersection, allowing different geometry types
    intersections = gpd.overlay(transects, shoreline, how='intersection', keep_geom_type=False)
    intersections = intersections[intersections.geometry.type == 'Point']  # Filter for Point geometries

    # Calculate distance along each transect for each intersection
    distance_data = []
    for transect_idx, transect in transects.iterrows():
        transect_line = transect.geometry
        transect_id = transect['name']  # Using 'name' as the identifier
        
        # Get intersections for the current transect
        transect_intersections = intersections[intersections.intersects(transect_line)]
        
        for _, intersection in transect_intersections.iterrows():
            intersection_point = intersection.geometry
            distance_along_transect = transect_line.project(intersection_point)  # Distance along transect line
            year = shoreline_file.split('_')[-1].split('.')[0]  # Extract year from filename
            
            distance_data.append({
                'transect_id': transect_id,
                'year': int(year),
                'distance': distance_along_transect
            })
    
    return distance_data

# Aggregate distance data for all shorelines
shoreline_files = [f for f in os.listdir(shoreline_dir) if f.endswith('.shp')]
all_distance_data = []

for file in shoreline_files:
    file_path = os.path.join(shoreline_dir, file)
    distance_data = calculate_intersection_distances(transects, file_path)
    all_distance_data.extend(distance_data)

# Convert the distance data to a DataFrame for easy plotting
import pandas as pd
distance_df = pd.DataFrame(all_distance_data)
print("Distance DataFrame columns:", distance_df.columns)

# Plot time series for each transect
transect_ids = distance_df['transect_id'].unique()
fig, ax = plt.subplots(figsize=(12, 8))

for transect_id in transect_ids:
    transect_data = distance_df[distance_df['transect_id'] == transect_id]
    transect_data = transect_data.sort_values(by='year')  # Sort by year for proper time series plotting
    ax.plot(transect_data['year'], transect_data['distance'], label=f'Transect {transect_id}')

ax.set_title("Shoreline Intersections Distance Along Transects Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Distance Along Transect")
plt.legend()
plt.show()
