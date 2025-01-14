import os
import xarray as xr
import json
import numpy as np
from glob import glob

def get_geometry_from_geojson(geojson_file):
    """
    Load the geometry (polygon) from a GeoJSON file.
    """
    with open(geojson_file) as f:
        geojson = json.load(f)
    return geojson["features"][0]["geometry"]

def clip_to_region(nc_files, geometry, output_dir):
    """
    Clips NetCDF files to the specified region and saves the clipped files.
    Longitudes are kept in the 0 to 360 range, and filenames are not altered.
    """
    coords = np.array(geometry["coordinates"][0])
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360

    for file_path in nc_files:
        print(f"Processing: {file_path}")
        ds = xr.open_dataset(file_path, engine="netcdf4")

        # Ensure longitude is in 0 to 360 format
        if ds.lon.min() < 0:
            ds = ds.assign_coords({"lon": (ds.lon % 360)}).sortby("lon")

        # Select the region
        clipped_ds = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max)
        )

        # Preserve metadata
        clipped_ds.attrs = ds.attrs
        for var in clipped_ds.data_vars:
            clipped_ds[var].attrs = ds[var].attrs

        # Save the clipped file to the same name in the output directory
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        clipped_ds.to_netcdf(output_path)
        print(f"Saved clipped file to: {output_path}")

def main():
    # Path to the GeoJSON file
    geojson_file = r"C:\Users\psteeves\coastal\canada_region.geojson"

    # Directories containing NetCDF files
    load_tide_dir = r"C:\Users\psteeves\coastal\load_tide"
    ocean_tide_dir = r"C:\Users\psteeves\coastal\ocean_tide"

    # Output directory for clipped files
    output_dir = r"C:\Users\psteeves\coastal\clipped_files"

    # Load the geometry
    geometry = get_geometry_from_geojson(geojson_file)

    # Find NetCDF files
    load_tide_files = glob(os.path.join(load_tide_dir, "*.nc"))
    ocean_tide_files = glob(os.path.join(ocean_tide_dir, "*.nc"))

    # Clip and save files
    print("Clipping load_tide files...")
    clip_to_region(load_tide_files, geometry, os.path.join(output_dir, "load_tide"))

    print("Clipping ocean_tide files...")
    clip_to_region(ocean_tide_files, geometry, os.path.join(output_dir, "ocean_tide"))

if __name__ == "__main__":
    main()
