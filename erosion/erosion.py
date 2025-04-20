def calculate_r_factor(precipitation_mm, rainfall_energy, rain_days):
    """
    Calculate the R factor for a given region.
    
    :param precipitation_mm: Annual precipitation in millimeters (mm)
    :param rainfall_energy: Rainfall energy in MJ·mm/ha·h (example: 75 for temperate regions)
    :param rain_days: Number of rainy days per year
    :return: R factor (MJ·mm/(ha·h·yr))
    """
    # Calculate the R factor using the formula
    r_factor = (rainfall_energy * precipitation_mm) / rain_days
    return r_factor

def predict_erosion(r_factor):
    """
    Predict the erosion risk based on the R factor.
    
    :param r_factor: The calculated R factor
    :return: Erosion risk category (Low, Moderate, High, Severe)
    """
    if r_factor < 200:
        return "Low Erosion Risk"
    elif 200 <= r_factor < 500:
        return "Moderate Erosion Risk"
    elif 500 <= r_factor < 1000:
        return "High Erosion Risk"
    else:
        return "Severe Erosion Risk"

# Example inputs
precipitation_mm = 1000  # Example annual precipitation in mm
rainfall_energy = 75     # Example rainfall energy in MJ·mm/ha·h (common for temperate regions)
rain_days = 150          # Example number of rain days per year

# Calculate the R factor
r_factor = calculate_r_factor(precipitation_mm, rainfall_energy, rain_days)

# Predict the erosion risk
erosion_risk = predict_erosion(r_factor)

# Print results
print(f"Calculated R factor: {r_factor:.2f}")
print(f"Erosion Risk: {erosion_risk}")
