import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved model components
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
cosine_sim = joblib.load('cosine_similarity.pkl')
medicine_data = joblib.load('medicine_data.pkl')

# Function to get recommendations
def get_recommendations(medicine_name, cosine_sim=cosine_sim):
    # Get the index of the medicine that matches the name
    idx = medicine_data[medicine_data['Medicine Name'] == medicine_name].index[0]

    # Get the pairwise similarity scores of all medicines with that medicine
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the medicines based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 5 most similar medicines (excluding itself)
    sim_scores = sim_scores[1:6]

    # Get the medicine names and return them
    medicine_indices = [i[0] for i in sim_scores]
    return medicine_data['Medicine Name'].iloc[medicine_indices]

# Streamlit App
st.title('Medicine Recommendation System')

# Dropdown for selecting a medicine
medicine_name = st.selectbox('Select a Medicine:', medicine_data['Medicine Name'].unique())

# Button to get recommendations
if st.button('Get Recommendations'):
    recommendations = get_recommendations(medicine_name)
    st.write(f"Recommended medicines similar to '{medicine_name}':")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

# Show the selected medicine details
if st.checkbox('Show selected medicine details'):
    selected_medicine_details = medicine_data[medicine_data['Medicine Name'] == medicine_name]
    st.write(selected_medicine_details[['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Manufacturer']])
