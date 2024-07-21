import streamlit as st
import os
from src.utils.ingest_text import create_vector_database
from src.utils.ingest_image import extract_and_store_images
from src.utils.text_qa import qa_bot
from src.utils.image_qa import query_and_print_results
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

def get_answer(query,chain):
    response = chain.invoke(query)
    return response['result']

st.title("MULTIMODAL DOC QA")
uploaded_file = st.file_uploader("File upload",type="pdf")
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get the absolute path of the saved file
    path = os.path.abspath(uploaded_file.name)
    st.write(f"File saved to: {path}")
    print(path)

st.write("Document uploaded successfuly!")


if st.button("Start Processing"):
    with st.spinner("Processing"):
        client = create_vector_database(path)
        image_vdb = extract_and_store_images(path)
    chain = qa_bot(client) 


    if user_input := st.chat_input("User Input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Generating Response..."):
            response = get_answer(chain,user_input)
            answer = response['result']
            st.markdown(answer)
            query_and_print_results(image_vdb,user_input)
            