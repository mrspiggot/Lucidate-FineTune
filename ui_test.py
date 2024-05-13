import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM



class LuciEnvConfig:
    def __init__(self):
        load_dotenv()
        self.huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

class LuciModelSetup:
    def __init__(self, api_token):
        self.api_token = api_token

    def setup_cloud_llm(self, repo_id, temperature=0.1, max_length=64):
        template = """Question: {question}
            Answer: Let's think step by step."""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        return LLMChain(prompt=prompt, llm=HuggingFaceEndpoint(repo_id=repo_id, temperature=temperature,
                                                               model_kwargs={"max_length": max_length},
                                                               huggingfacehub_api_token=self.api_token))

    def setup_local_llm(self, model_id='google/flan-t5-large', max_length=100):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=max_length)
        return HuggingFacePipeline(pipeline=pipe)



class LuciQueryExecutor:
    def __init__(self, model):
        self.model = model

    def ask_question(self, question):
        result = self.model.invoke(question)
        # Ensure the output is structured as expected
        if isinstance(result, dict):
            output = {'Question': question, 'Answer': result.get('text', "No answer found.")}
        elif isinstance(result, str):
            output = {'Question': question, 'Answer': result}
        else:
            output = {'Question': question, 'Answer': f"Unexpected result type: {type(result)}"}

        return output



class LuciStreamlitApp:
    def __init__(self):
        load_dotenv()
        self.env_config = LuciEnvConfig()
        self.model_selector = LuciModelSelector(self.env_config)
        self.chat_manager = None

    def setup_page_layout(self):
        st.markdown("""
            <style>
            .message-left {
                text-align: left;
                color: #FFA500;
            }
            .message-right {
                text-align: right;
                color: #00A5FF;
            }
            .chat-messages {
                height: 450px;
                overflow-y: auto;
            }
            </style>
            """, unsafe_allow_html=True)

    def display_messages(self):
        message_container = st.container()
        with message_container:
            st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
            for role, message in st.session_state.messages:
                if role == "You":
                    st.markdown(f"<div class='message-right'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='message-left'>{message}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        self.setup_page_layout()
        st.sidebar.title("Choose your Language Model")
        model_options = self.model_selector.list_models()
        chosen_model = st.sidebar.selectbox("Select a model", model_options)
        self.chat_manager = self.model_selector.get_model(chosen_model)

        st.title("Lucidate Chat Interface")
        # Create a message display area
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Messages display
        self.display_messages()

        # User input at the bottom
        with st.form("chat_form", clear_on_submit=False):
            user_input = st.text_input("Type your message here:", key="chat_input")
            submit_button = st.form_submit_button("Send")
            if submit_button and user_input:
                response = self.chat_manager.ask_question(user_input)
                st.session_state.messages.append(("You", user_input))
                st.session_state.messages.append(("Luci", response['Answer']))
                st.experimental_rerun()


class LuciModelSelector:
    def __init__(self, env_config):
        self.cloud_model_setup = LuciModelSetup(env_config.huggingfacehub_api_token)
        self.local_model_setup = LuciModelSetup(None)  # API token not needed for local models
        self.models = {
            "Local": LuciQueryExecutor(self.local_model_setup.setup_local_llm()),
            "Cloud": LuciQueryExecutor(self.cloud_model_setup.setup_cloud_llm("mistralai/Mistral-7B-Instruct-v0.2"))
        }

    def list_models(self):
        return list(self.models.keys())

    def get_model(self, model_name):
        return self.models[model_name]


if __name__ == "__main__":
    app = LuciStreamlitApp()
    app.run()

