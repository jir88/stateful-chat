import streamlit as st
#from langchain_core.messages import AIMessage, HumanMessage
#from langchain_community.llms import KoboldApiLLM
#from langchain_ollama import OllamaLLM
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import ChatPromptTemplate
from stateful_chat.session import ChatSession
import ollama
import uuid
import json


# load_dotenv()

# app config
st.set_page_config(page_title="Stateful Chatbot", page_icon="🤖")
st.title("Stateful Chatbot")

# initialize session state
if "chat_session" not in st.session_state:
    cs = ChatSession(str(uuid.uuid4()))
    cs.llm = "llama3.1"
    cs.stop_words = ["<|begin_of_text|>","<|start_header_id|>","<|eot_id|>"]
    cs.system_prompt = ""
    st.session_state.chat_session = cs

# make sure regenerate flag is initialized to 'False'
if "needs_regen" not in st.session_state:
    st.session_state.needs_regen = False

# make sure continue flag is initialized to 'False'
if "needs_continue" not in st.session_state:
    st.session_state.needs_continue = False

# callback to set regenerate flag
def request_regen():
    st.session_state.needs_regen = True
    
# callback to set regenerate flag
def request_continue():
    st.session_state.needs_continue = True

def load_state():
    """Load saved session state."""
    session_file = st.session_state.session_loader
  
    if session_file is None:
        st.warning("No settings file selected!")
        return
    st.session_state.chat_session = ChatSession.from_json(session_file)

def get_response():
    o_gen = st.session_state.chat_session.get_response(stream=True)
     # make wrapper
    def ogen_wrapper(o_gen):
        for chunk in o_gen:
            yield chunk['response']
        st.write("Response generation rate: " + str(round(chunk['eval_count']/chunk['eval_duration']*1e9, 2)) + " Tk/sec")
        st.session_state.context_length = chunk['prompt_eval_count'] + chunk['eval_count']
        st.write("Context length: " + str(st.session_state.context_length))

    return ogen_wrapper(o_gen)

def get_continuation():
    o_gen = st.session_state.chat_session.continue_response(stream=True)
     # make wrapper
    def ogen_wrapper(o_gen):
        for chunk in o_gen:
            yield chunk['response']
        st.write("Response generation rate: " + str(round(chunk['eval_count']/chunk['eval_duration']*1e9, 2)) + " Tk/sec")
        st.session_state.context_length = chunk['prompt_eval_count'] + chunk['eval_count']
        st.write("Context length: " + str(st.session_state.context_length))

    return ogen_wrapper(o_gen)

# system prompt
#st.session_state.system_prompt = st.text_area("System prompt:", value = "You are a helpful assistant.")

# stop token list
#st.session_state.chat_session.stop_words = st.text_input("Comma-separated stop words:", ",".join(st.session_state.chat_session.stop_words)).split(",")

# Construct tabs
tab_main, tab_mem, tab_db = st.tabs(["Main", "Memory", "Database"])
# =============== Main Tab ========================

with tab_main:
    # LLM model to use
    st.session_state.chat_session.llm = st.text_input("LLM:", st.session_state.chat_session.llm)
    
    # Base system prompt
    st.session_state.chat_session.system_prompt = st.text_area("System prompt:", st.session_state.chat_session.system_prompt, height = 25)
    
    # conversation roles
    st.session_state.chat_session.user_role = st.text_input("User role:", st.session_state.chat_session.user_role)
    st.session_state.chat_session.ai_role = st.text_input("AI role:", st.session_state.chat_session.ai_role)
    
    # mode-switching check-box
    if st.checkbox(label = "Manual editing mode"):
        # we're using manual mode
        # convert conversation to text and stick it in a text editor
        formatted_text = st.session_state.chat_session.format_readable()
        edited_text = st.text_area("Edit raw conversation text:", formatted_text, height = 500)
        # when the text area changes, put the new version into the session
        st.session_state.chat_session.import_readable(edited_text)
    else:
        # we're in automatic mode
        # do we need to regenerate a message?
        # if so, we need to drop the last message before loading history
        if st.session_state.needs_regen:
            # delete old response
            del st.session_state.chat_session.messages[-1]
        # now render conversation history in container
        with st.container(height=500):
            # display conversation
            if len(st.session_state.chat_session.messages) > 0:
                for msg in st.session_state.chat_session.messages:
                    with st.chat_message(msg['role']):
                        st.write(msg['content'])
            # if we're regenerating, do that now
            if st.session_state.needs_regen:
                with st.chat_message(st.session_state.chat_session.ai_role):
                    response = st.write_stream(get_response())
                    st.session_state.chat_session.append_message(role=st.session_state.chat_session.ai_role,
                                                                 content=response)
                    #st.session_state.chat_session.messages.append({"role": st.session_state.chat_session.ai_role, "content": response})
                # reset flag
                st.session_state.needs_regen = False
            # if we're continuing, do that now
            if st.session_state.needs_continue:
                with st.chat_message(st.session_state.chat_session.ai_role):
                    response = st.write_stream(get_continuation())
                    st.session_state.chat_session.append_message(role=st.session_state.chat_session.ai_role,
                                                                 content=response)
                    #st.session_state.chat_session.messages.append({"role": st.session_state.chat_session.ai_role, "content": response})
                # reset flag
                st.session_state.needs_continue = False
            # finally, provide chat input and regen buttons
            user_query = st.chat_input("Type your message here...")
            if user_query is not None and user_query != "":
                st.session_state.chat_session.append_message(role=st.session_state.chat_session.user_role,
                                                             content=user_query)
                #st.session_state.chat_session.messages.append({"role": st.session_state.chat_session.user_role, "content": user_query})
            
                with st.chat_message(st.session_state.chat_session.user_role):
                    st.markdown(user_query)
            
                with st.chat_message(st.session_state.chat_session.ai_role):
                    response = st.write_stream(get_response())
                    st.session_state.chat_session.append_message(role=st.session_state.chat_session.ai_role,
                                                                 content=response)
                    #st.session_state.chat_session.messages.append({"role": st.session_state.chat_session.ai_role, "content": response})
            # regenerate last message button
            st.button(label="Regenerate", on_click=request_regen)
            # regenerate last message button
            st.button(label="Continue", on_click=request_continue)
        # end chat container
    # end auto/manual selection

    # load button
    uploaded_file = st.file_uploader(label="Load a saved session", key = "session_loader", on_change = load_state)
    
    # save button
    # define state to save
    settings_to_download = {"session_id": st.session_state.chat_session.session_id,
                            "messages": st.session_state.chat_session.messages,
                            "stop_words": st.session_state.chat_session.stop_words,
                           "llm": st.session_state.chat_session.llm,
                           "user_role": st.session_state.chat_session.user_role,
                           "ai_role": st.session_state.chat_session.ai_role
                           }
        
    button_download = st.download_button(label="Save Settings",
                                         data = st.session_state.chat_session.to_json(),
                                           #data=json.dumps(settings_to_download),
                                           file_name=f"settings.json",
                                         key = "session_saver",
                                           help="Click to Download Current Settings")
    
    
    # manually generate summary of 6 oldest messages
    num_summ = st.number_input("Number of messages to summarize and archive:",
                               min_value=2, max_value=max([2, len(st.session_state.chat_session.messages)]),
                               value=2, step=1)
    if st.button(label="Summarize"):
        # update top-level summary and entity list
        st.session_state.chat_session.update_full_summary(0, num_summ)
        # make summary of the message chunk
        st.session_state.chat_session.messages_to_summary(0, num_summ)
        # archive the summarized messages
        st.session_state.chat_session.archive_messages(0, num_summ)

# ====================== Memory Tab ========================

with tab_mem:
    st.session_state.chat_session.prompt_full_summary = st.text_area("Full summary prompt:", 
                                                                     st.session_state.chat_session.prompt_full_summary,
                                                                     height = 100)
    if st.session_state.chat_session.full_summary is None:
        st.write("Full summary: None")
    else:
        st.session_state.chat_session.full_summary = st.text_area("Full summary:", st.session_state.chat_session.full_summary, height = 150)
    
    st.session_state.chat_session.prompt_msg_summary = st.text_area("Message summary prompt:", 
                                                                     st.session_state.chat_session.prompt_msg_summary,
                                                                     height = 100)
    if len(st.session_state.chat_session.message_summaries) == 0:
        st.write("Message chunk summary: None")
    else:
        st.session_state.chat_session.message_summaries[-1]['content'] = st.text_area("Message chunk summary:", 
                                                                           st.session_state.chat_session.message_summaries[-1]['content'],
                                                                           height = 150)
    
    st.session_state.chat_session.prompt_entity_list = st.text_area("Entity list prompt:", 
                                                                     st.session_state.chat_session.prompt_entity_list,
                                                                     height = 100)
    if st.session_state.chat_session.entity_list is None:
        st.write("Entity list: None")
    else:
        st.session_state.chat_session.entity_list = st.text_area("Entity list:", st.session_state.chat_session.entity_list, height = 150)
    # display current system prompt
    st.write("System prompt:")
    st.write(st.session_state.chat_session.compile_system_prompt())

# ====================== Database Tab ========================

with tab_db:
    query = st.text_input("Vector database query:")
    if query is not None and query != "":
        db_res = st.session_state.chat_session.query_vector_db(query_text=query, n_results=3)
        st.write(db_res)
    # button to update full vector database
    if st.button(label="Rebuild vector DB"):
        st.session_state.chat_session.update_full_vector_db()
