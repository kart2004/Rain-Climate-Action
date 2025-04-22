import csv

# Function to read the CSV and load the data into a dictionary
def load_state_data(csv_filename):
    state_data = {}
    with open(csv_filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            state = row['State']
            AAR = int(row['AAR'])
            AER = float(row['AER'])
            state_data[state] = {"AAR": AAR, "AER": AER}
    return state_data

# Function to calculate the R Factor based on state and rainfall
def calculate_r_factor(state, rainfall_mm, csv_filename="data/erosion_data.csv"):
    # Load the state data from the CSV file
    state_data = load_state_data(csv_filename)
    
    # Check if the state is valid
    if state not in state_data:
        return f"State '{state}' not found. Please check the state name."
    
    # Get the AAR and AER for the state
    state_info = state_data[state]
    AAR = state_info["AAR"]
    AER = state_info["AER"]
    
    # Calculate the energy of the rainfall (E)
    E = rainfall_mm * 0.05  # Simple estimate for energy calculation
    
    # Calculate the R factor for the given rainfall
    R_factor = (rainfall_mm * E) / AAR
    
    return {
        "State": state,
        "AAR": AAR,
        "AER": AER,
        "R Factor": R_factor
    }
