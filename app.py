import streamlit as st
from utils import *
from pypdf import PdfReader

class state:
    def __init__(self):
        self.state = True

def toggle_state(state):
    state.state = not state.state


def remove_element(collection, remove_idx):
    return [element for ix, element in enumerate(collection) if ix not in remove_idx]

def file_processor(file_manager: FileManager, db : VectorStoreManager,  uploaded_files : list):
    file_manager.get_files(uploaded_files)
    remove_idx = []
    for ix, file in enumerate(uploaded_files):
        if file_manager.is_identifier_exists(file.name):
            remove_idx.append(ix)

    uploaded_files = remove_element(uploaded_files, remove_idx)

    with st.status("Processing documents...", expanded=True) as status:
        if len(uploaded_files) == 0:
            status.update(label="Files already exsist!", state="complete", expanded=False)
        db.add_documents(uploaded_files)
        file_manager.add_db_identifier(uploaded_files)
        status.update(label="Files processed successfully!", state="complete", expanded=False)

    db.init_retriever()



if __name__ == "__main__":

    if "llm" not in st.session_state:
        st.session_state.llm = DocLLM("gemma4:31b-cloud")
    if "file_manager" not in st.session_state:
        st.session_state.file_manager = FileManager()
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = VectorStoreManager('db')
    if "mesg_record" not in st.session_state:
        st.session_state.mesg_record = MesgRecord()

    st.title('DocLLM', text_alignment="center")
    with st.sidebar:
        st.title("Document LLM")
        uploaded_files = st.file_uploader("Choose a file", 
                                        accept_multiple_files=True, 
                                        type=["pdf", "txt"])

        st.button("process files", on_click=file_processor, args=(st.session_state.file_manager, st.session_state.db_manager, uploaded_files))
        st.button("clear chat", on_click=st.session_state.mesg_record.clear)

    if st.session_state.db_manager.retriever and st.session_state.llm.chain is None:
        st.session_state.llm.init_chain(st.session_state.db_manager.retriever)
    
    mesg = st.chat_input("Ask a question about your document")
    if mesg :
        st.session_state.mesg_record.user_msg(mesg)
        responce = st.session_state.llm.chain.invoke(mesg)
        st.session_state.mesg_record.assitant_msg(responce.content)

    st.session_state.mesg_record.render_chat()


    