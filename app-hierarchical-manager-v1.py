import streamlit as st
import stateful_chat.manager as scm
import stateful_chat.llm as scl
import json
import os

# app config
st.set_page_config(page_title="Stateful Chatbot", page_icon="🤖")
st.title("Stateful Chatbot")

#========= Initialize session state =========

if "chat_session" not in st.session_state:
    # set up LLM backend
    llm = scl.OpenAILLM(model="gemma-3-4B-it-UD-Q4_K_XL-cpu")
    # separate LLM for summarizing
    summary_llm = scl.OpenAILLM(model="gemma-3-4B-it-UD-Q4_K_XL-cpu")
    # initialize session manager
    cs = scm.HierarchicalSummaryManager(llm=llm, summary_llm=summary_llm)
    cs.chat_thread.system_prompt = ""
    st.session_state.chat_session = cs

# make sure regenerate flag is initialized to 'False'
if "needs_regen" not in st.session_state:
    st.session_state.needs_regen = False

# make sure continue flag is initialized to 'False'
if "needs_continue" not in st.session_state:
    st.session_state.needs_continue = False

#========= Callback functions =========

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
    st.session_state.chat_session = scm.HierarchicalSummaryManager.from_json(session_file)
    # set the instruct formats properly in their select boxes
    st.session_state.inst_fmt_box = st.session_state.chat_session.llm.instruct_format.name
    st.session_state.mem_inst_fmt_box = st.session_state.chat_session.chat_memory.summary_llm.instruct_format.name

# method to load all available instruct formats so we can offer them to the user
def load_instruct_formats():
    # get paths to files
    fmt_dir = "./instruct_formats/"
    fmt_files = [(os.path.join(fmt_dir, file)) for file in os.listdir(fmt_dir)]
    # load each file into a list
    fmt_obj = []
    for ff in fmt_files:
        inst_fmt = scl.InstructFormat.from_json(open(ff, mode='r'))
        fmt_obj.append(inst_fmt)
    return fmt_obj
# if instruct formats aren't loaded, do that now
if "instruct_formats" not in st.session_state:
    # load available formats
    st.session_state.instruct_formats = load_instruct_formats()
    # pull out their display names
    st.session_state.instruct_names = [fmt.name for fmt in st.session_state.instruct_formats]

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

def update_llm_configuration():
    """
    Callback to parse the JSON LLM configurations and update the backend chat manager object.
    """
    # update main LLM
    # update instruct format
    fmt_idx = st.session_state.instruct_names.index(st.session_state.sb_main_inst_fmt)
    st.session_state.chat_session.llm.instruct_format = st.session_state.instruct_formats[fmt_idx]
    # update model, if changed
    if st.session_state.ti_main_llm_name != st.session_state.chat_session.llm.model:
        # keep original parameters
        samp_opts = st.session_state.chat_session.llm.sampling_options
        inst_fmt = st.session_state.chat_session.llm.instruct_format
        # initialize new LLM
        st.session_state.chat_session.llm = scl.OpenAILLM(model=st.session_state.ti_main_llm_name,
                                                          sampling_options=samp_opts,
                                                          instruct_fmt=inst_fmt)
    # update sampling parameters
    st.session_state.chat_session.llm.sampling_options = json.loads(st.session_state.ta_main_samp_opts)

    # update LLM used for memory generation
    fmt_idx = st.session_state.instruct_names.index(st.session_state.sb_mem_inst_fmt)
    st.session_state.chat_session.chat_memory.summary_llm.instruct_format = st.session_state.instruct_formats[fmt_idx]
    # update model, if changed
    if st.session_state.ti_mem_llm_name != st.session_state.chat_session.chat_memory.summary_llm.model:
        # keep original parameters
        samp_opts = st.session_state.chat_session.chat_memory.summary_llm.sampling_options
        inst_fmt = st.session_state.chat_session.chat_memory.summary_llm.instruct_format
        st.session_state.chat_session.chat_memory.summary_llm = scl.OpenAILLM(model=st.session_state.ti_mem_llm_name,
                                                          sampling_options=samp_opts,
                                                          instruct_fmt=inst_fmt)
    # update sampling parameters
    st.session_state.chat_session.chat_memory.summary_llm.sampling_options = json.loads(st.session_state.ta_mem_samp_opts)


# Construct tabs
tab_main, tab_mem, tab_arch, tab_db, tab_settings = st.tabs(["Main", "Memory", "Archive", "Database", "Settings"])
# =============== Main Tab ========================

with tab_main:
    # Base system prompt
    st.session_state.chat_session.chat_thread.system_prompt = st.text_area("System prompt:", st.session_state.chat_session.chat_thread.system_prompt, height = 70)
    
    # conversation roles
    st.session_state.chat_session.chat_thread.user_role = st.text_input("User role:", st.session_state.chat_session.chat_thread.user_role)
    st.session_state.chat_session.chat_thread.ai_role = st.text_input("AI role:", st.session_state.chat_session.chat_thread.ai_role)
    
    # mode-switching check-box
    if st.checkbox(label = "Manual editing mode"):
        # we're using manual mode
        # convert conversation to text and stick it in a text editor
        formatted_text = st.session_state.chat_session.chat_thread.format_readable()
        edited_text = st.text_area("Edit raw conversation text:", formatted_text, height = 500)
        # when the text area changes, put the new version into the session
        st.session_state.chat_session.chat_thread.import_readable(edited_text)
    else:
        # we're in automatic mode
        # do we need to regenerate a message?
        # if so, we need to drop the last message before loading history
        if st.session_state.needs_regen:
            # delete old response
            del st.session_state.chat_session.chat_thread.messages[-1]
        # now render conversation history in container
        with st.container(height=500):
            # display conversation
            if len(st.session_state.chat_session.chat_thread.messages) > 0:
                for msg in st.session_state.chat_session.chat_thread.messages:
                    with st.chat_message(msg['role']):
                        st.write(msg['content'])
            # if we're regenerating, do that now
            if st.session_state.needs_regen:
                with st.chat_message(st.session_state.chat_session.chat_thread.ai_role):
                    response = st.write_stream(get_response())
                    st.session_state.chat_session.append_message({
                        'role': st.session_state.chat_session.chat_thread.ai_role,
                        'content': response
                    })
                # reset flag
                st.session_state.needs_regen = False
            # if we're continuing, do that now
            if st.session_state.needs_continue:
                with st.chat_message(st.session_state.chat_session.chat_thread.ai_role):
                    response = st.write_stream(get_continuation())
                    st.session_state.chat_session.append_message({
                        'role': st.session_state.chat_session.chat_thread.ai_role,
                        'content': response
                    })
                # reset flag
                st.session_state.needs_continue = False
            # finally, provide chat input and regen buttons
            user_query = st.chat_input("Type your message here...")
            if user_query is not None and user_query != "":
                # add to chat thread
                st.session_state.chat_session.append_message({
                        'role': st.session_state.chat_session.chat_thread.user_role,
                        'content': user_query
                    })
                # display in chat widget
                with st.chat_message(st.session_state.chat_session.chat_thread.user_role):
                    st.markdown(user_query)
                # stream AI response
                with st.chat_message(st.session_state.chat_session.chat_thread.ai_role):
                    response = st.write_stream(get_response())
                    st.session_state.chat_session.append_message({
                        'role': st.session_state.chat_session.chat_thread.ai_role,
                        'content': response
                    })
            # regenerate last message button
            st.button(label="Regenerate", on_click=request_regen)
            # regenerate last message button
            st.button(label="Continue", on_click=request_continue)
        # end chat container
    # end auto/manual selection

    # load button
    uploaded_file = st.file_uploader(label="Load a saved session",
                                     key = "session_loader", 
                                     on_change = load_state)
    
    # save button
    button_download = st.download_button(label="Save Settings",
                                         data = st.session_state.chat_session.to_json(),
                                         file_name=f"settings.json",
                                         key = "session_saver",
                                         help="Click to Download Current Settings")
    
    if st.button(label="Update memory"):
        # tell state manager to update memory, ensuring all levels are within limits
        st.session_state.chat_session.chat_memory.update_all_memory()

# ====================== Memory Tab ========================

# NOTE: this version assumes that manager is using HierarchicalSummaryMemory!!!
with tab_mem:
    st.session_state.chat_session.chat_memory.summarization_prompt = st.text_area(
        "Summarization system prompt:",
        st.session_state.chat_session.chat_memory.summarization_prompt,
        height = 200
        )
    st.session_state.chat_session.chat_memory.prop_ctx = st.number_input(
        "Maximum context proportion threshold:",
        help="Proportion of the total context window that summaries plus un-summarized messages may use up before triggering a higher-level summary.",
        min_value=0.0, 
        max_value=1.0,
        value=st.session_state.chat_session.chat_memory.prop_ctx, 
        step=0.05
        )
    st.session_state.chat_session.chat_memory.prop_summary = st.number_input(
        "Maximum summary proportion:",
        help="The proportion of a message/summary level that can be occupied by messages/summaries of higher level. Each summary level is allocated prop_summary of the context alloted to the next higher level (total context window for original thread messages).",
        min_value=0.0, 
        max_value=1.0,
        value=st.session_state.chat_session.chat_memory.prop_summary, 
        step=0.05
        )
    st.session_state.chat_session.chat_memory.n_levels = st.number_input(
        "Maximum number of summary levels:",
        min_value=1,
        value=st.session_state.chat_session.chat_memory.n_levels, 
        step=1
        )
    st.session_state.chat_session.chat_memory.n_tok_summarize = st.number_input(
        "Number of tokens to summarize:",
        help="The target number of tokens to summarize in one pass. If this corresponds to less than one message, that whole message will be summarized.",
        min_value=1,
        value=st.session_state.chat_session.chat_memory.n_tok_summarize, 
        step=256
        )
    
    # mode-switching check-box
    if st.checkbox(label = "Manual memory editing"):
        # we're using manual mode
        # convert summaries to text and stick them in a text editor
        formatted_text = st.session_state.chat_session.chat_memory.format_readable()
        edited_text = st.text_area("Edit raw summary text:", formatted_text, height = 500)
        # when the text area changes, put the new version into the session
        st.session_state.chat_session.chat_memory.import_readable(edited_text)
    else:
        # we're in automatic mode
        # now render conversation history in container
        with st.container(height=500):
            # display conversation
            if len(st.session_state.chat_session.chat_memory.all_memory) > 0:
                for msg in st.session_state.chat_session.chat_memory.all_memory:
                    with st.chat_message("Summary"):
                        st.write("[Level " + str(msg['level']) + "] " + msg['content'])
        # end chat container
    # end auto/manual selection
    
    st.session_state.chat_session.chat_memory.prompt_entity_list = st.text_area("Entity list prompt:", st.session_state.chat_session.chat_memory.prompt_entity_list, height = 150)
    if st.session_state.chat_session.chat_memory.entity_list is None:
        st.write("Entity list: None")
    else:
        st.session_state.chat_session.chat_memory.entity_list = st.text_area("Entity list:", st.session_state.chat_session.chat_memory.entity_list, height = 150)
    
    # calculate sizes of all summary levels and print them
    if st.button(label="Context size"):
        total_size = 0
        # calculate size of raw messages
        level_size = 0
        for msg in st.session_state.chat_session.chat_thread.messages:
            level_size += st.session_state.chat_session.llm.count_tokens(msg['content'])
        total_size += level_size
        # calculate percent of alloted space
        level_allowance = st.session_state.chat_session.llm.sampling_options['num_ctx']*st.session_state.chat_session.chat_memory.prop_ctx
        level_pct = int(level_size/level_allowance*100)
        st.write(f"Message size: {level_size} ({level_pct}%)")
        for level in range(1, st.session_state.chat_session.chat_memory.n_levels + 1):
            level_size = st.session_state.chat_session.chat_memory.summary_level_size(level=level)
            level_allowance = st.session_state.chat_session.llm.sampling_options['num_ctx']*st.session_state.chat_session.chat_memory.prop_ctx*st.session_state.chat_session.chat_memory.prop_summary**level
            level_pct = int(level_size/level_allowance*100)
            st.write(f"Level {level} size: {level_size} ({level_pct}%)")
            total_size += level_size
        st.write(f"Total context size: {total_size}")

# ====================== Archive Tab ========================

# NOTE: this version assumes that manager is using HierarchicalSummaryMemory!!!
with tab_arch:
    # render non-editable archived conversation history in container
    with st.container(height=800):
        # display conversation
        if len(st.session_state.chat_session.chat_thread.messages) > 0:
            for msg in st.session_state.chat_session.chat_thread.archived_messages:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])
    # end chat container

# ====================== Database Tab ========================

with tab_db:
    query = st.text_input("Vector database query:")
    if query is not None and query != "":
        db_res = st.session_state.chat_session.chat_memory.query_memory(query_text=query, n_results=3)
        st.write(db_res)
    # button to update full vector database
    if st.button(label="Rebuild vector DB"):
        st.session_state.chat_session.chat_memory.update_all_memory()

# ========================= Sampling Tab ==========================

with tab_settings:
    # main LLM model to use
    st.text_input(
        "Main LLM:",
        st.session_state.chat_session.llm.model,
        key="ti_main_llm_name",
        on_change=update_llm_configuration)

    # main chat format to use
    st.selectbox(
        "Main instruct format to use:", 
        options=st.session_state.instruct_names, 
        key="sb_main_inst_fmt",
        on_change=update_llm_configuration)
    # main LLM sampling parameters
    formatted_text = json.dumps(st.session_state.chat_session.llm.sampling_options, indent=2)
    st.text_area(
        "Edit main LLM sampling parameters:", 
        formatted_text, 
        height = 250, 
        key="ta_main_samp_opts",
        on_change=update_llm_configuration
    )
    
    # memory LLM model to use
    st.text_input(
        "Memory LLM:",
        st.session_state.chat_session.chat_memory.summary_llm.model,
        key="ti_mem_llm_name",
        on_change=update_llm_configuration)

    # memory model chat format to use
    st.selectbox(
        "Memory LLM instruct format to use:", 
        options=st.session_state.instruct_names, 
        key="sb_mem_inst_fmt",
        on_change=update_llm_configuration)
    # memory LLM sampling parameters
    formatted_text = json.dumps(st.session_state.chat_session.chat_memory.summary_llm.sampling_options, indent=2)
    st.text_area(
        "Edit memory LLM sampling parameters:", 
        formatted_text, 
        height = 250, 
        key="ta_mem_samp_opts",
        on_change=update_llm_configuration
    )
