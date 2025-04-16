import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# State mapping dictionary
STATE_MAPPING = {
    "WB": {"full_name": "GANGETIC WEST BENGAL", "terrain": "Coastal-plateau"},
    "AN": {"full_name": "ANDAMAN & NICOBAR ISLANDS", "terrain": "Island"},
    "AR": {"full_name": "ARUNACHAL PRADESH", "terrain": "Hilly"},
    "AS": {"full_name": "ASSAM & MEGHALAYA", "terrain": "Hilly"},
    "BR": {"full_name": "BIHAR", "terrain": "Plain-land"},
    "CG": {"full_name": "CHATTISGARH", "terrain": "Hilly"},
    "AD": {"full_name": "COASTAL ANDHRA PRADESH", "terrain": "Coastal"},
    "KA": {"full_name": "NORTH INTERIOR KARNATAKA", "terrain": "Coastal"},
    "MP": {"full_name": "WEST MADHYA PRADESH", "terrain": "Plain-land"},
    "RJ": {"full_name": "WEST RAJASTHAN", "terrain": "Desert"},
    "UP": {"full_name": "WEST UTTAR PRADESH", "terrain": "Hilly"},
    "GJ": {"full_name": "GUJARAT REGION", "terrain": "Desert/marsh"},
    "DL": {"full_name": "HARYANA DELHI & CHANDIGARH", "terrain": "Plain-land"},
    "HP": {"full_name": "HIMACHAL PRADESH", "terrain": "Hilly"},
    "JK": {"full_name": "JAMMU & KASHMIR", "terrain": "Hilly"},
    "JH": {"full_name": "JHARKHAND", "terrain": "Forest"},
    "KL": {"full_name": "KERALA", "terrain": "Coastal"},
    "GA": {"full_name": "KONKAN & GOA", "terrain": "Hilly/coastal"},
    "LD": {"full_name": "LAKSHWADEEP", "terrain": "Island"},
    "MH": {"full_name": "MADHYA MAHARASHTRA", "terrain": "Plain-land"},
    "MT": {"full_name": "MATATHWADA", "terrain": "Barren"},
    "MN": {"full_name": "NAGA MANI MIZO TRIPURA", "terrain": "Hilly"},
    "OD": {"full_name": "ORISSA", "terrain": "Coastal"},
    "PB": {"full_name": "PUNJAB", "terrain": "Plain-land"},
    "RS": {"full_name": "RAYALSEEMA", "terrain": "Plain-land"},
    "SK": {"full_name": "SUB HIMALAYAN WEST BENGAL & SIKKIM", "terrain": "Hilly"},
    "TN": {"full_name": "TAMIL NADU", "terrain": "Hilly/coastal"},
    "TS": {"full_name": "TELANGANA", "terrain": "Hilly/plain"},
    "UK": {"full_name": "UTTARAKHAND", "terrain": "Hilly"},
    "VD": {"full_name": "VIDARBHA", "terrain": "Plain-land"}
}

# Month mapping
MONTH_MAPPING = {
    '01': {'quarter': 'Jan-Feb', 'duration': 2},
    '02': {'quarter': 'Jan-Feb', 'duration': 2},
    '03': {'quarter': 'Mar-May', 'duration': 3},
    '04': {'quarter': 'Mar-May', 'duration': 3},
    '05': {'quarter': 'Mar-May', 'duration': 3},
    '06': {'quarter': 'Jun-Sep', 'duration': 4},
    '07': {'quarter': 'Jun-Sep', 'duration': 4},
    '08': {'quarter': 'Jun-Sep', 'duration': 4},
    '09': {'quarter': 'Jun-Sep', 'duration': 4},
    '10': {'quarter': 'Oct-Dec', 'duration': 3},
    '11': {'quarter': 'Oct-Dec', 'duration': 3},
    '12': {'quarter': 'Oct-Dec', 'duration': 3}
}

# API Keys
BING_API_KEY = 'AozIVsiQ675xXwo2NwGtEuv8vtcQ098NSmpCuV1QAl7nFQ9wfjtcwSI_gdbH4sZV'
OPENWEATHER_API_KEY = 'e31020243ddd05cc3d37ad5f4816190f'

# Global preprocessing objects
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
labelencoder_X_3 = LabelEncoder()
onehotencoder = OneHotEncoder(sparse_output=False)
sc_X = StandardScaler()