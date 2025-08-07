import copy
import uuid
import json
import re
from stateful_chat.llm import OpenAILLM,InstructFormat,LLM

class StatefulChatManager:
    """
    Top-level class managing all the moving parts of a stateful chat.
    """

    def __init__(self, llm):
        """
        Create a new chat manager with a given large language model back-end.

        Args:
        llm (LLM): an LLM object to use
        """
        self.llm = llm
        # create empty chat thread
        self.chat_thread = ChatThread(str(uuid.uuid4()))
        # create empty memory store
        self.chat_memory = LLMSummaryMemory(llm=llm)

    def append_message(self, message):
        """
        Append message.

        Args:
        message (dict): Dict containing at least 'role' and 'content' keys
        """
        # if missing, set ID to be the message index
        if not "id" in message:
            message['id'] = len(self.chat_thread.messages) + len(self.chat_thread.archived_messages)
        # add to the active chat thread
        self.chat_thread.messages.append(message)
        # Needs updating: embed regenerated AI response
        #st.session_state.chat_session.embed_text(st.session_state.chat_session.messages[-1], "message")

    def messages_to_memory(self, n_msgs):
        """
        Remove a number of the oldest messages from context and commit them
        to memory.

        Args:
        n_msgs (int): Number of oldest messages
        """
        # pull messages
        old_msgs = self.chat_thread.messages[0:n_msgs]
        # add to memory. Send system prompt or the model goes insane
        self.chat_memory.add_messages(old_msgs, context=self.compile_system_prompt())
        # archive messages in thread
        self.chat_thread.archive_messages(0, n_msgs)

    def export_thread(self):
        pass

    def import_thread(self, messages):
        """
        Import a thread formatted as text. Existing messages are altered to
        reflect differences with the imported text.
        """
        pass

    def to_json(self):
        """
        Write this object out as a JSON object.

        Returns: a string containing the JSON object
        """
        # define state to save
        settings_to_download = {"llm": self.llm.to_json(),
                                "chat_thread": self.chat_thread.to_json(),
                                "chat_memory": self.chat_memory.to_json()
                                }
        # dump it to a JSON file
        return json.dumps(settings_to_download)

    @classmethod
    def from_json(cls, json_data):
        """
        Load saved session state from a JSON object.
        Args:
        json_data (str): JSON object or file containing session data

        Returns: a new ChatSession object initialized from the JSON data
        """
        # load saved state
        uploaded_settings = json.load(json_data)
        # if this is an old format, try to recover it
        if uploaded_settings.get('chat_thread') is None:
            return StatefulChatManager._recover_old_json_format(uploaded_settings)
        # initialize LLM
        # TODO: use some dynamic loading to handle other classes
        llm = OpenAILLM.from_json(uploaded_settings.get('llm'))
        # create new memory object
        new_obj = cls(llm=llm)
        # load chat thread
        new_obj.chat_thread = ChatThread.from_json(uploaded_settings.get('chat_thread'))
        # load chat memory
        new_obj.chat_memory = LLMSummaryMemory.from_json(uploaded_settings.get('chat_memory'))
        # return object
        return new_obj

    @classmethod
    def _recover_old_json_format(cls, uploaded_settings):
        """
        Upgrade an old JSON file to the current format.
        """
        # load instruct format
        inst_fmt = uploaded_settings.get('instruct_format')
        if inst_fmt is None:
            # need to add default, which is Llama 3
            inst_fmt = InstructFormat(name="Llama 3 Chat",
                                      message_template="<|start_header_id|>{role}<|end_header_id|>\n\n{content}",
                                      begin_of_text="",
                                      end_of_turn="<|eot_id|>",
                                      continue_template="<|start_header_id|>{role}<|end_header_id|>\n\n")
        else:
            # parse existing format
            inst_fmt = json.loads(inst_fmt)
            inst_fmt = InstructFormat(name=inst_fmt['name'],
                                      message_template=inst_fmt['message_template'],
                                      # old versions didn't have BoT
                                      begin_of_text="",
                                      end_of_turn=inst_fmt['end_of_turn'],
                                      continue_template=inst_fmt['continue_template'])
        
        # initialize LLM
        llm = OpenAILLM(model=uploaded_settings["llm"],
                        # default sampling options
                        sampling_options=None,
                        instruct_fmt=inst_fmt
                       )
        # create new chat manager
        new_manager = cls(llm=llm)
        
        # set up chat thread
        new_manager.chat_thread = ChatThread(session_id=uploaded_settings["session_id"])
        new_manager.chat_thread.system_prompt = uploaded_settings["system_prompt"]
        new_manager.chat_thread.messages = uploaded_settings["messages"]
        new_manager.chat_thread.user_role = uploaded_settings["user_role"]
        new_manager.chat_thread.ai_role = uploaded_settings["ai_role"]
        new_manager.chat_thread.archived_messages = uploaded_settings["archived_messages"]
        # add archived message IDs if they are missing
        if len(new_manager.chat_thread.archived_messages) > 0 and new_manager.chat_thread.archived_messages[0].get('id') is None:
            print("Fixing archived message IDs.")
            for i in range(0, len(new_manager.chat_thread.archived_messages)):
                new_manager.chat_thread.archived_messages[i]['id'] = i
        # also add current message IDs if those are missing
        if len(new_manager.chat_thread.messages) > 0 and new_manager.chat_thread.messages[0].get('id') is None:
            print("Fixing current message IDs.")
            for i in range(0, len(new_manager.chat_thread.messages)):
                new_manager.chat_thread.messages[i]['id'] = i + len(new_manager.chat_thread.archived_messages)
        
        # set up session memory using the main LLM
        # use a copy, though, so we can use different settings for them in the future
        new_manager.chat_memory = LLMSummaryMemory(llm=copy.deepcopy(llm))
        # import message summaries
        new_manager.chat_memory.message_summaries = uploaded_settings["message_summaries"]
        # if summaries are stored as strings, update to dicts with indices
        if len(new_manager.chat_memory.message_summaries) > 0 and str(new_manager.chat_memory.message_summaries[0].__class__) != "<class 'dict'>":
            print(str(new_manager.chat_memory.message_summaries[0].__class__))
            for i in range(0, len(new_manager.chat_memory.message_summaries)):
                new_manager.chat_memory.message_summaries[i] = { 
                    "id": i, 
                    "content": new_manager.chat_memory.message_summaries[i]
                }
        # add full summary
        new_manager.chat_memory.full_summary = uploaded_settings["full_summary"]
        # load entity list
        new_manager.chat_memory.entity_list = uploaded_settings["entity_list"]
        # memory prompts
        new_manager.chat_memory.init_sys_prompt = "You are an expert summarizer. You will summarize the following messages. You will also use the messages to update a running summary of the whole previous exchange. The following messages are a conversation between {ai} and {user}.\n\nContext:\n"
        new_manager.chat_memory.prompt_msg_summary = uploaded_settings.get("prompt_msg_summary")
        if new_manager.chat_memory.prompt_msg_summary is None:
            new_manager.chat_memory.prompt_msg_summary = "Concisely summarize these messages. Include all relevant details. Reference context from prior summaries where relevant, but focus on the most recent messages. Match the tense and perspective of the story."
        new_manager.chat_memory.prompt_full_summary = uploaded_settings.get("prompt_full_summary")
        if new_manager.chat_memory.prompt_full_summary is None:
            new_manager.chat_memory.prompt_full_summary = "Concisely summarize all messages so far. Base this summary on the previous full summary. Include all relevant details. Mention any unresolved discussion topics."
        new_manager.chat_memory.prompt_entity_list = uploaded_settings.get("prompt_entity_list")
        if new_manager.chat_memory.prompt_entity_list is None:
            new_manager.chat_memory.prompt_entity_list = "Provide a list of all entities mentioned thus far and a brief description of each. For people, include a brief description of their personalities. Write more detailed descriptions for more important entities."
        # return object
        return new_manager

    def compile_system_prompt(self):
        """
        Combine raw prompt, the most recent message summary, and the entity list
        into a full system prompt.

        TODO: the memory stuff should probably be delegated to the memory class.
        """
        # start with the system prompt for the current chat thread, if any
        full_sys_prompt = ""
        if self.chat_thread.system_prompt is not None:
            full_sys_prompt += self.chat_thread.system_prompt.strip()
        # add top-level summary from memory
        if self.chat_memory.full_summary is not None:
            full_sys_prompt += "\n\nComplete summary of all previous messages:\n" + self.chat_memory.full_summary
        # add entity list, if any
        if self.chat_memory.entity_list is not None:
            full_sys_prompt += "\n\nEntitites mentioned previously:\n" + self.chat_memory.entity_list
        # add latest message summary, if any
        if len(self.chat_memory.message_summaries) > 0:
            full_sys_prompt += "\n\nSummary of recent previous messages:\n" + self.chat_memory.message_summaries[-1]['content']
        return full_sys_prompt
    
    def get_response(self, stream=True):
        # make the system prompt
        #TODO: should probably delegate a bunch of this formatting to the memory object
        sys_prompt = self.compile_system_prompt().strip()
        all_msgs = [{ 'role': "system", 'content': sys_prompt }]
        # add in-context messages after sys prompt
        all_msgs.extend(self.chat_thread.messages)
        # generate response using current thread's AI role
        return self.llm.generate_instruct(messages=all_msgs,
                                          respond=True,
                                          response_role=self.chat_thread.ai_role,
                                          stream=stream
                                          )

    def continue_response(self, stream=True):
        """
        Continue generating from the end of the most recent message.
        """
        # make the system prompt
        #TODO: should probably delegate a bunch of this formatting to the memory object
        sys_prompt = self.compile_system_prompt().strip()
        all_msgs = [{ 'role': "system", 'content': sys_prompt }]
        # add in-context messages after sys prompt
        all_msgs.extend(self.chat_thread.messages)
        # continue generating from end of last message
        return self.llm.generate_instruct(messages=all_msgs,
                                          respond=False,
                                          stream=stream
                                          )

class ChatThread:
    """
    A single chat thread between a user and an LLM.
    
    Attributes:
    session_id (str): The unique identifier of the chat thread.
    messages (list): A list of messages sent in the chat thread.
    """

    # pulls role names out of a string representation of a thread
    role_regex = re.compile(r"{{(.+?)}}")

    def __init__(self, session_id):
        """
        Initializes the chat session with a given session ID.

        Args:
        session_id (str): The unique identifier of the chat session.
        """
        self.session_id = session_id
        self.system_prompt = None
        self.messages = []
        self.user_role = "user"
        self.ai_role = "assistant"
        # past messages, no longer in context/working memory
        self.archived_messages = []

    def archive_messages(self, start_idx, stop_idx):
        """
        Move some messages from the main message thread to the
        archived messages list.

        Args:
        start_idx (int): index of first message to archive (inclusive)
        end_idx (int): index of last message to archive (exclusive)
        """
        self.archived_messages.extend(self.messages[start_idx:stop_idx])
        del self.messages[start_idx:stop_idx]

    def format_readable(self):
        """
        Convert all messages in this thread into a human-readable and editable
        format. Message roles are displayed in curly brackets: {{role}} with
        message text following. Leading and trailing whitespace are ignored.
        """
        result = ""
        for i in range(0, len(self.messages)):
            result += "{{" + self.messages[i]['role'] + "}}\n" + self.messages[i]['content'] + "\n"
        return result

    def import_readable(self, formatted_messages:str):
        """
        Parse messages exported by format_readable and use them to replace any
        existing messages in this chat session.

        Args:
        formatted_messages (str): The formatted messages to be parsed.
        """
        # splitting with capturing groups returns the roles too
        msg_parts = self.role_regex.split(formatted_messages)
        # drop first item, which is blank for some reason
        msg_parts = msg_parts[1:]
        
        # strip out extra whitespace and format
        # should probably pre-allocate this...
        parsed_messages = []
        for i in range(0, len(msg_parts), 2):
            msg_dict = {
                "id": len(self.archived_messages) + i/2,
                "role": msg_parts[i],
                "content": str.strip(msg_parts[i + 1])
            }
            parsed_messages.append(msg_dict)
        self.messages = parsed_messages

    def to_json(self):
        """
        Write this object out as a JSON object.

        Returns: a string containing the JSON object
        """
        # define state to save
        settings_to_download = {"session_id": self.session_id,
                                "system_prompt": self.system_prompt,
                                "messages": self.messages,
                                "user_role": self.user_role,
                                "ai_role": self.ai_role,
                                "archived_messages": self.archived_messages
                                }
        # dump it to a JSON file
        return json.dumps(settings_to_download)

    @classmethod
    def from_json(cls, json_data):
        """
        Load saved session state from a JSON object.
        Args:
        json_data (str): JSON object or file containing session data

        Returns: a new ChatSession object initialized from the JSON data
        """
        # load saved state
        if type(json_data) == str:
            uploaded_settings = json.loads(json_data)
        else:
            uploaded_settings = json.load(json_data)
        # create new thread object
        new_obj = cls(session_id=uploaded_settings.get('session_id'))
        # load system prompt
        new_obj.system_prompt = uploaded_settings.get('system_prompt')
        # load messages
        new_obj.messages = uploaded_settings.get('messages')
        # load user role
        new_obj.user_role = uploaded_settings["user_role"]
        # load AI role
        new_obj.ai_role = uploaded_settings["ai_role"]
        # load archived messages
        new_obj.archived_messages = uploaded_settings.get('archived_messages')
        # return object
        return new_obj

class ChatMemory:
    """
    Abstract class for various methods of helping LLMs 'remember' information beyond
    their context lengths.
    """

    def add_documents(self, docs):
        """
        Add a set of documents to memory. How these are added will vary by implementation.

        Args:
        docs (list[dict]): a list of dicts, where each dict has the document ('content' key)
            plus any other useful metadata.
        """
        raise NotImplementedError("Method not implemented!")

    def update_memory(self, mem_id, mem_content):
        raise NotImplementedError("Method not implemented!")

    def query_memory(self, query_text, n_results=3):
        raise NotImplementedError("Method not implemented!")

    def update_all_memory(self):
        raise NotImplementedError("Method not implemented!")
    
class LLMSummaryMemory(ChatMemory):
    """
    A class that manages medium- and long-term memory for a single chat thread
    between a user and an LLM. This implementation maintains a single over-all
    thread summary plus a series of periodical summaries. It also uses the LLM
    to keep track of entities and provide descriptions.
    """

    def __init__(self, llm):
        """
        Create a new memory instance.

        Args:
        llm (LLM): the large language model to use for summarizing and entity extraction
        """
        self.llm = llm
        self.user_role = "user"
        self.ai_role = "assistant"
        self.message_summaries = []
        self.full_summary = None
        self.entity_list = None
        # memory prompts
        self.init_sys_prompt = "You are an expert summarizer. You will summarize the following messages. You will also use the messages to update a running summary of the whole previous exchange. The following messages are a conversation between {ai} and {user}.\n\nContext:\n"
        #self.sys_prompt_msg_summary = "You are an expert summarizer. Summarize the following messages."
        self.prompt_msg_summary = "Concisely summarize these messages. Include all relevant details. Reference context from prior summaries where relevant, but focus on the most recent messages. Match the tense and perspective of the story."
        self.prompt_full_summary = "Concisely summarize all messages so far. Base this summary on the previous full summary. Include all relevant details. Mention any unresolved discussion topics."
        self.prompt_entity_list = "Provide a list of all entities mentioned thus far and a brief description of each. For people, include a brief description of their personalities. Write more detailed descriptions for more important entities."
    
    def add_messages(self, msgs, context=None):
        """
        Add a set of messages to memory. The messages will be summarized together and will
        also be used to update the full summary. Finally, the entity list will be updated.

        Args:
        msgs (list[dict]): a list of dicts, where each dict has the message ('content' key),
            the 'role' key, plus any other useful metadata.
        context (str): a short description of the chat context. Usually, this is the system
            message of the chat thread.
        """
        msg_list = []
        # ======  Initial system message ======

        init_msg = ""
        # add the context
        # introduce the LLM to the task at hand
        #if len(self.init_sys_prompt) > 0:
        #    init_msg += self.init_sys_prompt
        # add top-level summary, if any
        #if self.full_summary is not None:
        #    init_msg += "\n\nComplete summary of all previous messages:\n" + self.full_summary
        # add entity list, if any
        #if self.entity_list is not None:
        #    init_msg += "\n\nEntitites mentioned previously:\n" + self.entity_list
        # add previous summary if available
        #if len(self.message_summaries) > 0:
        #    init_msg += "\n\nSummary of most recent previous messages:\n" + self.message_summaries[-1]['content']
        # add it to the task message
        msg_list.extend([{
            'role': "system",
            'content': context
        }])
        print(context)
        
        # ======  Generate summary of current messages ======
        
        # add messages themselves
        msg_list.extend(msgs)
        # add the summarization prompt
        msg_list.extend([{
                'role': "system",
                'content': self.prompt_msg_summary
            }])
        # now generate a response
        o_gen = self.llm.generate_instruct(msgs, respond=True, response_role=self.ai_role, stream=False)
        # add summary to list of summaries
        msg_summ = {
            "id": len(self.message_summaries),
            "content": o_gen['response']
        }
        self.message_summaries.extend([msg_summ])
        # add the summary to our list of messages
        msg_list.extend([{
            'role': self.ai_role,
            'content': o_gen['response']
        }])
        
        # ====== Update full summary ======

        # add the summarization prompt
        msg_list.extend([{
                'role': "system",
                'content': self.prompt_full_summary
            }])
        # now generate a response
        o_gen = self.llm.generate_instruct(msgs, respond=True, response_role=self.ai_role, stream=False)
        # add summary to list of full summaries
        #msg_summ = {
        #    "id": len(self.message_summaries),
        #    "content": o_gen['response']
        #}
        #self.message_summaries.extend([msg_summ])
        self.full_summary = o_gen['response']
        # add the summary to our list of messages
        msg_list.extend([{
            'role': self.ai_role,
            'content': o_gen['response']
        }])
        
        # ====== Update entity list ======

        # add the entity extraction prompt
        msg_list.extend([{
                'role': "system",
                'content': self.prompt_entity_list
            }])
        # now generate a response
        o_gen = self.llm.generate_instruct(msgs, respond=True, response_role=self.ai_role, stream=False)
        # grab the updated list
        self.entity_list = o_gen['response']
        print(msg_list)

    def add_documents(self, docs):
        """
        Adds arbitrary documents to this memory instance by treating them as messages from the user
        requesting that the document be remembered.
        Args:
        docs (list[dict]): a list of dicts, where each dict has the document ('content' key)
            plus any other useful metadata.
        """
        msg_prefix = "Please remember this information:\n\n"
        ai_response = "Thank you, I will remember this."
        fake_messages = []
        for d in docs:
            user_msg = {
                'role': self.user_role,
                'content': msg_prefix + d
            }
            ai_msg = {
                'role': self.ai_role,
                'content': ai_response
            }
            fake_messages.extend([user_msg, ai_msg])
        self.add_messages(fake_messages)

    def update_memory(self, mem_id, mem_content):
        raise NotImplementedError("Method not implemented!")

    def query_memory(self, query_text, n_results=3):
        raise NotImplementedError("Method not implemented!")

    def update_all_memory(self):
        raise NotImplementedError("Method not implemented!")

    def to_json(self):
        """
        Write this object out as a JSON object.

        Returns: a string containing the JSON object
        """
        # define state to save
        settings_to_download = {"llm": self.llm.to_json(),
                                "user_role": self.user_role,
                                "ai_role": self.ai_role,
                                "message_summaries": self.message_summaries,
                                "full_summary": self.full_summary,
                                "entity_list": self.entity_list,
                                "init_sys_prompt": self.init_sys_prompt,
                                "prompt_msg_summary": self.prompt_msg_summary,
                                "prompt_full_summary": self.prompt_full_summary,
                                "prompt_entity_list": self.prompt_entity_list
                                }
        # dump it to a JSON file
        return json.dumps(settings_to_download)

    @classmethod
    def from_json(cls, json_data):
        """
        Load saved session state from a JSON object.
        Args:
        json_data (str): JSON object or file containing session data

        Returns: a new ChatSession object initialized from the JSON data
        """
        # load saved state
        if type(json_data) == str:
            uploaded_settings = json.loads(json_data)
        else:
            uploaded_settings = json.load(json_data)
        # initialize LLM
        llm = OpenAILLM.from_json(uploaded_settings.get('llm'))
        # create new memory object
        new_obj = cls(llm=llm)
        # load user role
        new_obj.user_role = uploaded_settings["user_role"]
        # load AI role
        new_obj.ai_role = uploaded_settings["ai_role"]
        # load message summaries
        new_obj.message_summaries = uploaded_settings["message_summaries"]
        # load full summary
        new_obj.full_summary = uploaded_settings["full_summary"]
        # load entity list
        new_obj.entity_list = uploaded_settings["entity_list"]
        # load baseline system prompt
        new_obj.init_sys_prompt = uploaded_settings["init_sys_prompt"]
        # load prompts
        new_obj.prompt_msg_summary = uploaded_settings.get("prompt_msg_summary")
        new_obj.prompt_full_summary = uploaded_settings.get("prompt_full_summary")
        new_obj.prompt_entity_list = uploaded_settings.get("prompt_entity_list")
        
        # return object
        return new_obj

class HierarchicalSummaryManager(StatefulChatManager):
    """
    Custom chat manager that compresses long chats into the context window using 
    heirarchical summaries of older messages, similar to the method used by
    perchance.ai.
    """

    def __init__(
        self, llm:LLM, summary_llm:LLM=None,
        prop_ctx:float=0.8, prop_summary:float=0.5,
        n_levels:int=3, n_tok_summarize:int=1024):
        """
        Create a new chat manager with a given large language model back-end.

        Args:
        llm (LLM): an LLM object to use
        summary_llm (LLM): the LLM model to use when generating summaries. NOTE: make
            sure this model has the same allocated context window size as the main LLM!
        chat_thread (ChatThread): the chat thread associated with this memory object
        prop_ctx (float): the proportion of the total context window that summaries
            plus un-summarized messages may use up before triggering a higher-level
            summary.
        prop_summary (float): The proportion of a message/summary level that can
            be occupied by messages/summaries of higher level. Each summary
            level is allocated prop_summary of the context alloted to the next higher
            level (total context window for original thread messages).
        n_levels (int): the maximum number of summary levels to use
        n_tok_summarize (int): the target number of tokens to summarize in one pass.
            If this corresponds to less than one message, that whole message will be
            summarized.
        """
        self.llm = llm
        # if no summary LLM specified, use main one
        if summary_llm is None:
            summary_llm = llm
        # create empty chat thread
        self.chat_thread = ChatThread(str(uuid.uuid4()))
        # create empty memory store
        self.chat_memory = HierarchicalSummaryMemory(
            summary_llm=summary_llm,
            chat_thread=self.chat_thread,
            prop_ctx=prop_ctx,
            prop_summary=prop_summary,
            n_levels=n_levels,
            n_tok_summarize=n_tok_summarize
            )
    
    def get_response(self, stream=True):
        """
        Generate an AI response starting with the end of the current thread.
        Uses summary messages to ensure we don't overflow the context window.

        Args:
        stream (bool): whether to stream the response or not

        Returns: a generator if streaming or the response text if not streaming
        """
        sys_prompt = self.compile_system_prompt().strip()
        all_msgs = [{ 'role': "system", 'content': sys_prompt }]
        # add in-context messages after sys prompt
        all_msgs.extend(self.chat_thread.messages)
        # generate response using current thread's AI role
        return self.llm.generate_instruct(messages=all_msgs,
                                          respond=True,
                                          response_role=self.chat_thread.ai_role,
                                          stream=stream
                                          )

    def continue_response(self, stream=True):
        """
        Continue generating from the end of the most recent message.
        """
        # make the system prompt
        sys_prompt = self.compile_system_prompt().strip()
        all_msgs = [{ 'role': "system", 'content': sys_prompt }]
        # add in-context messages after sys prompt
        all_msgs.extend(self.chat_thread.messages)
        # continue generating from end of last message
        return self.llm.generate_instruct(messages=all_msgs,
                                          respond=False,
                                          stream=stream
                                          )

    def compile_system_prompt(self):
        """
        Combine raw prompt and summaries into a full system prompt.
        """
        # start with the system prompt for the current chat thread, if any
        full_sys_prompt = ""
        if self.chat_thread.system_prompt is not None:
            full_sys_prompt += self.chat_thread.system_prompt.strip()
        # add top-level summary from memory
        if len(self.chat_memory.all_memory) > 0:
            mems = [m['content'] for m in self.chat_memory.all_memory]
            full_sys_prompt += "\n\nSummary of all previous messages:\n" + "\n".join(mems)
        return full_sys_prompt

    @classmethod
    def from_json(cls, json_data):
        """
        Load saved session state from a JSON object.
        Args:
        json_data (str): JSON object or file containing session data

        Returns: a new HierarchicalSummaryManager object initialized from the JSON data
        """
        # load saved state
        uploaded_settings = json.load(json_data)
        # if this is an old format, try to recover it
        # TODO: convert regular managers into hierarchical ones with no summaries?
        if uploaded_settings.get('chat_thread') is None:
            return StatefulChatManager._recover_old_json_format(uploaded_settings)
        # initialize LLM
        # TODO: use some dynamic loading to handle other classes
        llm = OpenAILLM.from_json(uploaded_settings.get('llm'))
        # load chat memory, which has required parameters for manager construction
        new_chat_memory = HierarchicalSummaryMemory.from_json(uploaded_settings.get('chat_memory'))
        # create new manager object
        new_obj = cls(
            llm=llm,
            summary_llm=new_chat_memory.summary_llm,
            prop_ctx=new_chat_memory.prop_ctx,
            prop_summary=new_chat_memory.prop_summary,
            n_levels=new_chat_memory.n_levels,
            n_tok_summarize=new_chat_memory.n_tok_summarize
            )
        # assign chat thread from memory
        new_obj.chat_thread = new_chat_memory.chat_thread
        # load chat memory
        new_obj.chat_memory = new_chat_memory
        # return object
        return new_obj

class HierarchicalSummaryMemory(ChatMemory):
    """
    Manages chat memory using a similar mechanism to the one used by perchance.ai.
    The memory structure uses multiple 'levels' of summaries, with the raw messages
    treated as level 0. Each time the length of a summary level plus all higher levels
    exceeds some set proportion of the context window, part of that summary level is
    summarized and added to the next higher summary level. This process is applied to
    all summary levels as needed. The top level is a single summary message. Lower 
    level summaries are merged into this one instead of adding more.

    This class keeps track of all message summaries, the messages that they summarized,
    and the chat message index where the last summarized message is located.
    """

    def __init__(
        self, summary_llm:LLM, chat_thread:ChatThread,
        summary_prompt:str = None,
        prop_ctx:float=0.8,
        prop_summary:float=0.5,
        n_levels:int=3,
        n_tok_summarize:int=1024):
        """
        Construct a new memory object.

        Args:
        summary_llm (LLM): the LLM model to use when generating summaries. NOTE: make
            sure this model has the same allocated context window size as the main LLM!
        chat_thread (ChatThread): the chat thread associated with this memory object
        summary_prompt (str): Optional custom summarization prompt. To include prior
            context in the prompt, use placeholder {context}. If no custom prompt
            provided, uses a default prompt instead.
        prop_ctx (float): the proportion of the total context window that summaries
            plus un-summarized messages may use up before triggering a higher-level
            summary.
        prop_summary (float): The proportion of a message/summary level that can
            be occupied by messages/summaries of higher level. Each summary
            level is allocated prop_summary of the context alloted to the next higher
            level (total context window for original thread messages).
        n_levels (int): the maximum number of summary levels to use
        n_tok_summarize (int): the target number of tokens to summarize in one pass.
            If this corresponds to less than one message, that whole message will be
            summarized.
        """
        self.summary_llm = summary_llm
        self.chat_thread = chat_thread
        self.prop_ctx = prop_ctx
        self.prop_summary = prop_summary
        self.n_levels = n_levels
        self.n_tok_summarize = n_tok_summarize
        
        self.summarization_prompt = """You are summarizing a long series of messages into a concise but accurate summary. You will be given any relevant prior context and the user will provide the messages to be summarized. You must only summarize the content of the messages themselves, not the prior context. Make sure to include all important details.

        Prior context:
        {context}

        Now the user will provide you with the messages to be summarized. Respond only with a single-paragraph summary, no additional commentary."""
        if summary_prompt is not None:
            self.summarization_prompt = summary_prompt
        # summaries are stored as a list of dicts with summary level, the actual
        # messages (or lower-level summaries) that were summarized, and the index
        # of the final summarized message in the full chat thread
        self.all_memory = []
        # summaries which have been collapsed into the top-level summary go here
        self.archived_memory = []

    def update_all_memory(self):
        """
        Update memory so that all message levels fit within their corresponding
        token allotments. If the raw messages themselves are too big, the oldest
        messages will be summarized and archived. Note that this process only 
        summarizes the oldest n_tok_summarize tokens of each level (rounded up to
        the next message), so one or more levels may still be 'over-budget' afterwards.
        """
        # first memory will be in the highest current level
        if len(self.all_memory) > 0:
            current_level = self.all_memory[0]['level']
        else:
            # no memories, so we're at level 0 (raw messages)
            current_level = 0
        
        # tokens occupied by higher levels of summary
        higher_level_tokens = 0
        
        # the index of the first summary we will be summarizing
        # starts at 0, then updated as we finish handling each level
        start_summ_index = 0

        # now iterate through the levels until we hit the raw message level
        while current_level > 0:
            # find index of first summary in this level
            start_summ_index = self._get_index_of_first_summary_in_level(level=current_level)
            # is this level too big?
            level_allowance = self.summary_llm.sampling_options['num_ctx']*self.prop_ctx*self.prop_summary**current_level
            current_level_tokens = self._summary_level_size(level=current_level)
            if (higher_level_tokens + current_level_tokens) >= level_allowance:
                # if the next-to-highest summary level is too big, we include the top level summary
                if current_level == (self.n_levels - 1) and self.all_memory[0]['level'] == self.n_levels:
                    # summarizing both the existing top-level summary and part of the
                    # next level down into a new top-level summary
                    idx_to_summarize = [0]
                    idx_to_summarize.extend(self._get_summary_indices_in_level(level=current_level))
                else:
                    # normally just summarizing within the current level
                    idx_to_summarize = self._get_summary_indices_in_level(level=current_level)
                # get messages within level that we are going to summarize
                lim_idx = self._get_messages_with_token_size(
                    msgs=[self.all_memory[i] for i in idx_to_summarize],
                    n_tok=self.n_tok_summarize
                )
                idx_to_summarize = idx_to_summarize[slice(lim_idx+1)]
                # summarize the messages
                summarized_messages = [self.all_memory[i] for i in idx_to_summarize]
                new_top_summary = self._summarize_messages(
                    messages=summarized_messages,
                    prior_summaries=self.all_memory[:start_summ_index]
                )
                # put old summary messages in archive
                self.archived_memory.extend(summarized_messages)
                # delete from active summaries
                # have to work in reverse order so indices don't change while deleting
                for i in reversed(idx_to_summarize):
                    del self.all_memory[i]
                # insert new summary
                nts_dict = {
                    # make sure updated top-level summaries keep the same level
                    'level': min(current_level + 1, self.n_levels),
                    # last message index of the last summary in this summary
                    'msg_idx': max([s['msg_idx'] for s in summarized_messages]),
                    'content': new_top_summary
                }
                self.all_memory.insert(
                    # replace the first summarized index
                    idx_to_summarize[0],
                    nts_dict
                )
            # add current level's remaining tokens to the cumulative total
            higher_level_tokens += self._summary_level_size(level=current_level)
            # move to next level down
            current_level -= 1

        # now we look at the raw messages
        # index of first summary in this level is the end of the memory list
        start_summ_index = len(self.all_memory)
        # is message level too big?
        level_allowance = self.summary_llm.sampling_options['num_ctx']*self.prop_ctx
        # how long is the current message thread?
        current_level_tokens = 0
        for summary in self.chat_thread.messages:
            current_level_tokens += self._chars_to_tokens(summary['content'])
        # if it is too big
        if (higher_level_tokens + current_level_tokens) >= level_allowance:
            # index of last message that fills up our summarization budget
            lim_idx = self._get_messages_with_token_size(
                msgs=self.chat_thread.messages,
                n_tok=self.n_tok_summarize
            )
            summarized_messages = self.chat_thread.messages[slice(lim_idx+1)]
            # summarize
            new_top_summary = self._summarize_messages(
                messages=summarized_messages,
                prior_summaries=self.all_memory
            )
            # archive these messages from the chat thread
            self.chat_thread.archive_messages(
                start_idx=0,
                stop_idx=lim_idx+1
            )
            # insert new first-level summary
            nts_dict = {
                'level': 1,
                # last message index of the last message in this summary
                'msg_idx': round(summarized_messages[-1]['id']),
                'content': new_top_summary
            }
            # new summary goes at the end
            self.all_memory.append(nts_dict)

    def _summarize_messages(self, messages:list, prior_summaries:list=[]):
        """
        Summarize a list of messages, optionally including a list of older summaries
        as context.

        Args:
        messages (list): a list of messages to summarize
        prior_summaries (list): a list of older summaries to be used as context when summarizing

        Returns: the summary
        """
        # if no prior context, just put 'None' in as a placeholder
        if len(prior_summaries) == 0:
            prior_summaries = [{ 'content': "No prior context." }]
        
        # construct system prompt
        sys_prompt = {
            'role': 'system',
            'content': self.summarization_prompt.format(context="\n\n".join([ps['content'] for ps in prior_summaries]))
        }
        user_prompt = {
            'role': 'user',
            'content': "Please summarize the following messages:\n\n" + "\n\n".join([m['content'] for m in messages])
        }
        # generate the summary
        llm_response = self.summary_llm.generate_instruct(
            messages=[sys_prompt, user_prompt],
            respond=True,
            response_role="assistant",
            stream=False
        )
        # pull the first/only result off the generator and strip whitespace
        return next(llm_response)['response'].strip()

    def _get_index_of_first_summary_in_level(self, level:int):
        if len(self.all_memory) == 0:
            return 0
        for i in range(len(self.all_memory)):
            if self.all_memory[i]['level'] == level:
                return i
        # no summaries at this level
        return -1

    def _get_summaries_in_level(self, level:int):
        """
        Get all the summaries of a given level, in order.

        Returns: a list of all the summaries in that level
        """
        if len(self.all_memory) == 0:
            return []
        level_msgs = []
        for summary in self.all_memory:
            if summary['level'] == level:
                level_msgs.append(summary)
        return level_msgs
    
    def _get_summary_indices_in_level(self, level:int):
        """
        Get the indices of all the summaries of a given level, in order.

        Returns: a list of the summary indicies in the memory list
        """
        if len(self.all_memory) == 0:
            return []
        level_idx = []
        for idx in range(len(self.all_memory)):
            if self.all_memory[idx]['level'] == level:
                level_idx.append(idx)
        return level_idx

    def _get_messages_with_token_size(self, msgs, n_tok:int):
        """
        Calculates the index of the first message in this list in which
        the cumulative number of tokens exceeds n_tok, or len(msgs) if
        the total size of all messages is less than n_tok.
        """
        cum_tokens = 0

        for i in range(len(msgs)):
            cum_tokens += self._chars_to_tokens(text=msgs[i]['content'])
            if cum_tokens >= n_tok:
                # this is the message where we go over the token limit
                return i
        return len(msgs)
    
    def _summary_level_size(self, level:int):
        """
        Estimate the number of tokens in all summaries of a given level.

        Args:
        level (int): the summary level of interest

        Returns: the approximate number of tokens.
        """
        if len(self.all_memory) == 0:
            return 0
        level_size = 0
        for summary in self.all_memory:
            if summary['level'] == level:
                level_size += self._chars_to_tokens(summary['content'])
        return level_size
    
    def _chars_to_tokens(self, text:str):
        """
        Extremely rough estimation of tokens in a string (~3.5 chars/token).

        Args:
        text (str): the text to be estimated

        Returns: the approximate number of tokens
        """
        if isinstance(self.summary_llm, OpenAILLM):
            return self.summary_llm.count_tokens(text)
        return max(1, len(text)/3.5)

    def format_readable(self):
        """
        Convert all active summaries into a human-readable and editable
        format. Summary levels and positions are displayed in curly brackets:
        {{L<level>@<position>}} with summary text following. Leading and trailing
        whitespace are stripped.
        """
        result = ""
        for mem in self.all_memory:
            result += "{{L" + str(mem['level']) + "@" + str(mem['msg_idx']) + "}}\n" + mem['content'] + "\n"
        return result
    
    memory_regex = re.compile(r"{{L(\d+)@(\d+)}}")

    def import_readable(self, formatted_messages:str):
        """
        Parse messages exported by format_readable and use them to replace any
        existing messages in this chat session.

        Args:
        formatted_messages (str): The formatted messages to be parsed.
        """
        # splitting with capturing groups returns the roles too
        msg_parts = self.memory_regex.split(formatted_messages)
        # drop first item, which is blank for some reason
        msg_parts = msg_parts[1:]
        
        # strip out extra whitespace and format
        # should probably pre-allocate this...
        parsed_messages = []
        for i in range(0, len(msg_parts), 3):
            msg_dict = {
                "level": int(msg_parts[i]),
                "msg_idx": int(msg_parts[i + 1]),
                "content": str.strip(msg_parts[i + 2])
            }
            parsed_messages.append(msg_dict)
        self.all_memory = parsed_messages

    def to_json(self):
        """
        Write this object out as a JSON object.

        Returns: a string containing the JSON object
        """
        # define state to save
        settings_to_download = {"summary_llm": self.summary_llm.to_json(),
                                "chat_thread": self.chat_thread.to_json(),
                                "summarization_prompt": self.summarization_prompt,
                                "prop_ctx": self.prop_ctx,
                                "prop_summary": self.prop_summary,
                                "n_levels": self.n_levels,
                                "n_tok_summarize": self.n_tok_summarize,
                                "all_memory": self.all_memory,
                                "archived_memory": self.archived_memory
                                }
        # dump it to a JSON file
        return json.dumps(settings_to_download)

    @classmethod
    def from_json(cls, json_data):
        """
        Load saved session state from a JSON object.
        Args:
        json_data (str): JSON object or file containing session data

        Returns: a new ChatSession object initialized from the JSON data
        """
        # load saved state
        if type(json_data) == str:
            uploaded_settings = json.loads(json_data)
        else:
            uploaded_settings = json.load(json_data)
        # initialize LLM
        llm = OpenAILLM.from_json(uploaded_settings.get('summary_llm'))
        # load associated chat thread
        ct = ChatThread.from_json(uploaded_settings.get('chat_thread'))
        # create new memory object
        new_obj = cls(
            summary_llm=llm,
            chat_thread=ct,
            summary_prompt=uploaded_settings.get('summarization_prompt')
            )
        # load summary sizing parameters
        new_obj.prop_ctx = uploaded_settings["prop_ctx"]
        new_obj.prop_summary = uploaded_settings["prop_summary"]
        new_obj.n_levels = uploaded_settings["n_levels"]
        new_obj.n_tok_summarize = uploaded_settings["n_tok_summarize"]
        # load active summaries
        new_obj.all_memory = uploaded_settings["all_memory"]
        # load archived summaries
        new_obj.archived_memory = uploaded_settings["archived_memory"]
        
        # return object
        return new_obj