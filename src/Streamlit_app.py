import streamlit as st
import pandas as pd
import numpy as np
import joblib


#######################################################
#### LOAD THE TRAINED MODEL
#######################################################

@st.cache_resource  # -decorated function returns the cached instance of the return value (if the value is already cached). 

# function to load the trained model
def load_model():
    bundle = joblib.load("src/finalized_model.joblib")
    return bundle

bundle = load_model()  # used to load the trained load_model function
model = bundle["model"]  # used in the make predictions section
features = bundle["features"]  # used in the build app section

#######################################################
#### APP TITLE AND DESCRIPTION - Self explanitory
#######################################################

st.title("🏙️ Airbnb Smart Price & Location Explorer")

st.markdown(
    """
    This app helps estimate Airbnb nightly prices in **New York City**
    based on listing characteristics and location preferences.

    Use the controls in the sidebar to explore different stay options.
    """
)

#######################################################
#### CATEGORICAL INPUTS - Side bar options
#######################################################

st.sidebar.header("🏠 Listing Preferences")

# drop box to select room type
# sidebar creates a menu bar on left / selectbox makes the option a drop box
room_type = st.sidebar.selectbox(
    "Room type",
    ["Entire home/apt", "Private room", "Shared room"]
)

# drop box to select area of rental
# sidebar creates a menu bar on left / selectbox makes the option a drop box
neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)


#######################################################
#### NUMERIC INPUTS - Side bar inputs
#######################################################

# slide bar to select number of nights
# sidebar creates a menu bar on left / slider makes the option a slider
minimum_nights = st.sidebar.slider(
    "Minimum nights",
    min_value=1,
    max_value=30,
    value=3 # where the slider starts upon opening
)

# slide bar to select how many reviews you want renter to have
# sidebar creates a menu bar on left / slider makes the option a slider
number_of_reviews = st.sidebar.slider(
    "Number of reviews",
    min_value=0,
    max_value=50,
    value=10 # where the slider starts upon opening
)

# slide bar to select how many listings you want the renter to have
# sidebar creates a menu bar on left / slider makes the option a slider
host_listings = st.sidebar.slider(
    "Host listing count",
    min_value=1,
    max_value=50,
    value=1 # where the slider starts upon opening
)

#######################################################
#### BUILD APP
#######################################################

# create a dictionary where every feature name is a key, and every value starts as 0.
input_data = dict.fromkeys(features, 0)

# numeric features - replaces the 0 default with actual user inputs or start value 
input_data["minimum_nights"] = minimum_nights
input_data["number_of_reviews"] = number_of_reviews
input_data["calculated_host_listings_count"] = host_listings

# one-hot encoded categoricals - controls what we select in the drop box selections
room_col = f"room_type_{room_type}"
neigh_col = f"neighbourhood_group_{neighbourhood_group}"

# if statements for when selecting from the droop boxes
if room_col in input_data:
    input_data[room_col] = 1

if neigh_col in input_data:
    input_data[neigh_col] = 1

# Convert to DataFrame - converts the user’s selections into the exact data structure the model needs in order to calculate a price.
X_input = pd.DataFrame([input_data])

#######################################################
#### MAKE PREDICTIONS - 
#######################################################

# button to click once final on choice to load price
if st.button("💰 Predict nightly price"):
    prediction = model.predict(X_input)[0]

# output of price once clicked above button
    st.success(f"Estimated nightly price: **${prediction:,.2f}**")

# friendly message below price prediction
st.markdown("Enjoy your time in the **Big Apple 🍎**")




### OLD CODE FROM CLASS

#######################################################
#### FEATURE INPUTS WE NEED FROM THE USER
#######################################################

#minimum_nights = st.slider("Choose a minimum number of nights", 1,10)
#num_reviews = st.slider("Number of reviews I want the host to have:", 1,100)
#host_listings = st.slider("How many houses do I want the host to have?", 1, 20)


#######################################################
#### LOAD THE TRAINED MODEL
#######################################################

#filename = 'src/finalized_model.pkl'

#with open(filename, 'rb') as file:
#    loaded_model = pickle.load(file)

#st.write(pd.DataFrame([loaded_model.feature_names_in_,loaded_model.coef_]).T)

#features_list = ['calculated_host_listings_count', 'minimum_nights', 'number_of_reviews']