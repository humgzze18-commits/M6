import streamlit as st

# Set the app title 
st.title('My first Streamlit App') 
# Add a header
st.header('This is a header')
# Add a markdown
st.markdown('Esto es texto **markdown**')
# Add a welcome message 
st.write('Welcome to my Streamlit app!') 
# Create a text input 
widgetuser_input = st.text_input('Enter a custom message:', 'Hello, Streamlit!') 
# Display the customized message 
st.write('Mensaje personalizado:', widgetuser_input)
# st.subheader('Mensaje personalizado:', widgetuser_input)
# Add a slider
slider_value = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
