# visualize.py

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import numpy as np

# Folder Setup
data_folder = 'data'
output_folder = 'Outputs'
os.makedirs(output_folder, exist_ok=True)

# Helper Function: Create Cluster Map
def create_cluster_map(df, lat_col, lon_col, color_col, save_path, popup_cols=None, zoom_start=8):
    m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()],
                   zoom_start=zoom_start, tiles='cartodbpositron')
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in df.dropna(subset=[lat_col, lon_col]).iterrows():
        value = row[color_col]
        try:
            # Clean and safely convert value to float
            value = float(str(value).replace(',', '').replace('$', '').replace(' ', '').strip())
        except:
            value = np.nan
        
        if np.isnan(value):
            continue

        # Set color based on price
        if value < 500000:
            color = 'green'
        elif value < 1000000:
            color = 'blue'
        elif value < 2000000:
            color = 'orange'
        else:
            color = 'red'

        popup_text = ""
        if popup_cols:
            for col in popup_cols:
                col_value = row.get(col, 'N/A')
                popup_text += f"<b>{col}:</b> {col_value}<br>"

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)

    m.save(save_path)
    print(f"✅ Saved: {save_path}")

# Main Process
def main():
    print("🔄 Loading CSVs...")

    # Define file mappings
    datasets = [
        {
            'file': 'home_sales.csv',
            'lat_col': 'LATITUDE',
            'lon_col': 'LONGITUDE',
            'price_col': 'PRICE',
            'popup_cols': ['ADDRESS', 'CITY', 'PRICE'],
            'output_file': 'sales_cluster_map.html'
        },
        {
            'file': 'home_rentals.csv',
            'lat_col': 'LATITUDE',
            'lon_col': 'LONGITUDE',
            'price_col': 'RENT_PER_MONTH',
            'popup_cols': ['ADDRESS', 'CITY', 'RENT_PER_MONTH'],
            'output_file': 'rentals_cluster_map.html'
        },
        {
            'file': 'sales.csv',
            'lat_col': 'LATITUDE',
            'lon_col': 'LONGITUDE',
            'price_col': 'PRICE',
            'popup_cols': ['ADDRESS', 'CITY', 'PRICE'],
            'output_file': 'sales_scraped_cluster_map.html'
        },
        {
            'file': 'rentals.csv',
            'lat_col': 'LATITUDE',
            'lon_col': 'LONGITUDE',
            'price_col': 'PRICE-VALUE',
            'popup_cols': ['Homecard__Address', 'City_card', 'PRICE-VALUE'],
            'output_file': 'rentals_scraped_cluster_map.html'
        }
    ]

    # Loop through datasets and generate maps
    for d in datasets:
        file_path = os.path.join(data_folder, d['file'])
        if os.path.exists(file_path):
            print(f"🗺️ Generating map for {d['file']}...")
            df = pd.read_csv(file_path)
            df.columns = [col.strip() for col in df.columns]  # Strip all spaces in header names

            create_cluster_map(
                df=df,
                lat_col=d['lat_col'],
                lon_col=d['lon_col'],
                color_col=d['price_col'],
                save_path=os.path.join(output_folder, d['output_file']),
                popup_cols=d['popup_cols']
            )
        else:
            print(f"⚠️ File not found: {d['file']}")

    print("\n✅ All maps generated and saved in Outputs/ folder!")

if __name__ == "__main__":
    main()
