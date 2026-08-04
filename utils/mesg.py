import streamlit as st

class chat:
    def __init__(self, who, mesg):
        self.who = who
        self.mesg = mesg

    def __call__(self):
        st.chat_message(self.who).write(self.mesg)
        

class MesgRecord:
    def __init__(self):
        self.record = []

    def user_msg(self, mesg):
        self.record.append(chat('user', mesg))

    def assitant_msg(self, mesg):
        self.record.append(chat('assistant', mesg))

    def clear(self):
        self.record = []

    def render_chat(self):
        if len(self.record) == 0:
            return
        for chat in self.record:
            chat()