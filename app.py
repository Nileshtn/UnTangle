import streamlit as st

st.set_page_config(page_title="DocChat App", layout="wide")
st.title("Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_context" not in st.session_state:
    st.session_state.document_context = None 
with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "csv"])
    
    if uploaded_file is not None:
        if st.button("Process File"):
            try:
                file_bytes = uploaded_file.getvalue()
                result = "worked"
                
                st.session_state.document_context = result
                st.success("File processed and ready for chat!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

st.header("2. Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        response = "This is a placeholder response."
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})