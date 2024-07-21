import streamlit as st
from io import BytesIO
from IPython.display import Image, display
from PIL import Image as PILImage

def query_and_print_results(image_vdb,query):
    results=3
    # Query the database
    query_results = image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances']
    )

    # Print the results
    for idx, uri in enumerate(query_results['uris'][0]):
        img = Image(filename=uri, width=300)
        st.image(img) # type: ignore

# Testing it out
