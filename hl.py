import pandas as pd

# Create new data for the year 2023 for all four junctions
new_data = pd.DataFrame({
    'DateTime': pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='H'),
    'Junction': [1, 2, 3, 4] * 2190,  # Data for each junction repeated for all hours of the year
    'Vehicles': [0] * 8760  # Placeholder for the number of vehicles, replace with actual values
})

# Load existing dataset
existing_data = pd.read_csv("traffic1.csv")  # Replace "traffic1.csv" with your actual dataset filename

# Concatenate existing dataset with new data
combined_data = pd.concat([existing_data, new_data], ignore_index=True)

# Save the combined dataset back to a file
combined_data.to_csv("combined_traffic_data.csv", index=False)
