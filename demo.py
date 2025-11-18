import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Simple Data Dashboard")

st.write('Grafica en Contenedor')

with st.container():
    st.write("This is inside the container")
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})  

    st.write(df)
    # You can call any Streamlit command, including custom components.
    st.bar_chart(df)



    x = 10
    y = np.random.normal(0, 1, size=1000)
    fig, ax = plt.subplots()
    ax.hist(y, bins=20)

    st.write(fig)

# Formateando máximos de una tabla

    df = pd.DataFrame(np.random.randn(10, 20), columns=("col %d" % i for i in range(20)))

    st.dataframe(df.style.highlight_max(axis=0))

# edición, selección y orden

    df = pd.DataFrame(
    [
       {"postre": "pastel", "rating": 4, "is_widget": True},
       {"postre": "helado", "rating": 5, "is_widget": False},
       {"postre": "galletas", "rating": 3, "is_widget": True},
   ]
)
edited_df = st.data_editor(df)

favorite_dessert = edited_df.loc[edited_df["rating"].idxmax()]["postre"]
st.markdown(f"Your favorite dessert is: **{favorite_dessert}** ")