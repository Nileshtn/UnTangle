import streamlit as st
from utils.core import *
from pypdf import PdfReader


def file_processor(file_manager: FileManager, db : VectorStoreManager,  uploaded_files : list):
    file_manager.get_files(uploaded_files)
    with st.status("Processing documents...", expanded=True) as status:
        db.add_documents(uploaded_files)
        file_manager.add_db_identifier(uploaded_files)
        status.update(label="Files processed successfully!", state="complete", expanded=False)

if __name__ == "__main__":
    llm = DocLLM("gemma4:31b-cloud")
    file_manager = FileManager()
    db_manager = VectorStoreManager('db')


    st.title('DocLLM', text_alignment="center")
    with st.sidebar:
        st.title("Document LLM")
        uploaded_files = st.file_uploader("Choose a file", 
                                        accept_multiple_files=True, 
                                        type=["pdf", "txt"])

        st.button("process files", on_click=file_processor, args=(file_manager, db_manager, uploaded_files))
   


    mesg = st.chat_input("Ask a question about your document")
    