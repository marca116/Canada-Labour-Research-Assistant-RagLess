import argparse
import logging

import streamlit as st

from config import BaseChatbotInterfaceConfig, ChatbotInterfaceConfig, vLLMChatbotInterfaceConfig, vLLMRAGConfig, OllamaRAGConfig
from tools import retrieve_database_stream
from translations import Translator

logging.basicConfig(filename='.debugging/debugging.log', level=logging.DEBUG)

class App:
    def __init__(self, 
                 engine="ollama", 
                 eval_mode=False, 
                 is_remote=False,
                 hyperparams=OllamaRAGConfig.HyperparametersAccuracyConfig):
        self.eval_mode = eval_mode
        self.is_remote = is_remote
        self.engine = engine
        self.hyperparams = hyperparams

        self.config: BaseChatbotInterfaceConfig = ChatbotInterfaceConfig if self.engine=="ollama" else vLLMChatbotInterfaceConfig

        self.model = self.config.default_model_local if not self.is_remote else self.config.default_model_remote
        self.nb_previous_questions = self.config.nb_previous_questions

        user_language = "fr" if st.context.locale and st.context.locale.startswith('fr') else "en"

        # Initialize language in session state if not present
        if 'language' not in st.session_state:
            st.session_state.language = user_language

        self.system_prompt = ""
        self.translator = Translator(st.session_state.language)

        if "expander_state" not in st.session_state:
            st.session_state["expander_state"] = True

        if "messages" in st.session_state and len(st.session_state.messages) > 0:
            last_message_is_user = st.session_state.messages[-1]["role"] == "user"
            if last_message_is_user:
                st.session_state.messages.pop()
            
    def close_expander(self):
        st.session_state["expander_state"] = False

    def open_expander(self):
        st.session_state["expander_state"] = True
    
    def sidebar_config(self):
        with st.sidebar:

            st.markdown(f"# {self.translator.get('sidebar.language')}")
            
            # Create two columns for language buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("EN", use_container_width=True, type="primary" if st.session_state.language == "en" else "secondary"):
                    st.session_state.language = "en"
                    st.rerun()
            with col2:
                if st.button("FR", use_container_width=True, type="primary" if st.session_state.language == "fr" else "secondary"):
                    st.session_state.language = "fr"
                    st.rerun()

            st.markdown(f"# {self.translator.get('sidebar.title')}")

            self.is_remote = st.toggle(label=self.translator.get('sidebar.remote_mode'), 
                                       value=self.is_remote,
                                       help=self.translator.get('sidebar.remote_mode_tooltip'))
            
            model_shortlist = self.config.models_shortlist_local if not self.is_remote else self.config.models_shortlist_remote
            default_model = self.config.default_model_local if not self.is_remote else self.config.default_model_remote
            
            self.model = st.selectbox(label=self.translator.get('sidebar.model_prompt'), 
                                options=model_shortlist, 
                                index=model_shortlist.index(default_model))
            
            self.nb_previous_questions = st.number_input(label=self.translator.get('sidebar.previous_questions'),
                                    value=self.nb_previous_questions,
                                    min_value=0,
                                    help=self.translator.get('sidebar.previous_questions_tooltip'))
            
            with st.popover(self.translator.get('sidebar.advanced_settings'), use_container_width=True, icon=":material/component_exchange:"):
                self.system_prompt = st.text_area(label=str((self.translator.get('sidebar.system_prompt'))), 
                                              height=300,
                                              key="custom_prompt",
                                              value="",
                                              placeholder=self.translator.get('sidebar.system_prompt_placeholder'))
            
            if st.button(self.translator.get('sidebar.reset_button'), 
                         icon=":material/refresh:",
                         key='new_chat', 
                         help=self.translator.get('sidebar.reset_help'),
                         use_container_width=True,
                         on_click=self.open_expander):  
                st.session_state.messages = []
                st.rerun()
                

        return self.model, self.system_prompt
    
    def main(self):

        def display_download_button(content, key):
            st.download_button(
                label=self.translator.get('download'), 
                data=content, 
                file_name="chat_conversation.txt",
                key=key,
                on_click='ignore'
            )

        def display_retrieval_messages(eval_mode=self.eval_mode):
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":

                with st.spinner(self.translator.get('processing'), show_time=False):
                    result_generator = retrieve_database_stream(
                        st.session_state.messages[-1]["content"],
                        language=st.session_state.language,
                        is_remote=self.is_remote,
                        hyperparams=self.hyperparams,
                        chat_model=self.model,
                        custom_system_prompt=self.system_prompt,
                        previous_messages=st.session_state.messages[:-1],
                        nb_previous_questions=self.nb_previous_questions,
                        engine=self.engine
                    )
                    
                    # Create a wrapper generator that captures the full answer
                    class StreamCapture:
                        def __init__(self, generator):
                            self.generator = generator
                            self.full_answer = ""
                        
                        def __iter__(self):
                            for chunk in self.generator:
                                self.full_answer += chunk
                                yield chunk
                    
                    stream_capture = StreamCapture(result_generator)
                    answer = ""
                    
                    with st.container():
                        st.write_stream(stream_capture)
                        
                        # Get the captured answer
                        answer = stream_capture.full_answer

                    st.session_state.messages.append({"role": "assistant", "content": answer, "nb_previous_questions": self.nb_previous_questions})

                display_download_button(answer, key=f"download_{st.session_state.profiling_counter}")

        def process_incoming_input():
            if user_input := st.chat_input(self.translator.get('chat_input'),
                                           on_submit=self.close_expander):
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role":"user", "content":user_input})
                st.empty()

        st.set_page_config(
            page_title=self.translator.get('title'),
            initial_sidebar_state="auto",
            layout="centered",
            menu_items={"Report a bug": "mailto:pierreolivier.bonin@hrsdc-rhdcc.gc.ca",
                        "About":"Developed by Pierre-Olivier Bonin, Ph.D."}
        )

        # Load CSS file
        with open('styles/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        st.title(self.translator.get('title'))

        _, self.system_prompt = self.sidebar_config()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "previous_question_chunks" not in st.session_state:
            st.session_state.previous_question_chunks = []

        previous_messages = []

        for idx, message in enumerate(st.session_state.messages):
            is_assistant = message["role"] == "assistant"
            if is_assistant:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

                display_download_button(message.get("content", ""), key=f"download_{idx}") # only 1 in 2 messages (the assistant's) should have a download button
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

            previous_messages.append(message)

        process_incoming_input()

        display_retrieval_messages()

if __name__ == '__main__':
    default_mode = ChatbotInterfaceConfig.default_mode
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("mode", # as per [this issue](https://discuss.streamlit.io/t/command-line-arguments/386/3) keyword args don't work
                        help="Choose the mode in which you want to run this app",
                        choices=["local", "remote", "evaluation_local", "evaluation_remote", "profiling_local", "profiling_remote", "vllm"],
                        type=str,
                        nargs='?',
                        default=default_mode)

    args = parser.parse_args()
    print(args)
    
    if not hasattr(st.session_state, 'profiling_counter'):
        st.session_state.profiling_counter = 0
    
    is_vllm = args.mode in ["vllm"]
    is_remote = args.mode in ["remote", "evaluation_remote", "profiling_remote"]
    location_mode_name = "Remote" if is_remote else "Local"
    hyperparams = vLLMRAGConfig.HyperparametersAccuracyConfig if is_vllm else OllamaRAGConfig.HyperparametersAccuracyConfig
    engine = "vllm" if is_vllm else "ollama"

    app = App(is_remote=is_remote, engine=engine, hyperparams=hyperparams)
    app.main()