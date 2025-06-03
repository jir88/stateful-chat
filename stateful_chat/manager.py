import copy
import uuid
import json
import re
from stateful_chat.llm import OpenAILLM,InstructFormat

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

    def import_readable(self, formatted_messages):
        """
        Parse messages exported by format_readable and use them to replace any
        existing messages in this chat session.

        Args:
        readable_text (str): The formatted messages to be parsed.
        """
        re_role = re.compile(r"{{(.+?)}}")
        # splitting with capturing groups returns the roles too
        msg_parts = re_role.split(formatted_messages)
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

class HierarchicalSummaryMemory(ChatMemory):
    """
    Manages chat memory using a similar mechanism to the one used by perchance.ai.
    The memory structure uses multiple 'levels' of summaries, with the raw messages
    treated as level 0. Each time the length of a summary level plus all higher levels
    exceeds some set proportion of the context window, part of that summary level is
    summarized and added to the next higher summary level. This process is applied to
    all summary levels as needed. The top level is a single summary message. Lower 
    level summaries are merged into this one instead of adding more.
    """

    def __init__(self, llm, prop_ctx=0.8, n_levels=3, prop_level=0.5):
        """
        Construct a new memory object.

        Args:
        llm (LLM): the LLM model to use when generating summaries
        prop_ctx (double): the proportion of maximum allotted context that a summary
            level may use up before triggering a higher-level summary.
        n_levels (int): the number of summary levels to use
        prop_level (double): the proportion of each level allotted to higher-level
            summary messages
        """
        self.llm = llm
        self.prop_ctx = prop_ctx
        self.n_levels = n_levels
        self.prop_level = prop_level
        # messages are stored as a list of dicts with index, role, and content, plus
        # a summary level keyword
        self.all_memory = []

    def add_messages(self, msgs, context=None):
        """
        Add a set of messages to memory. The messages will be summarized together and will
        also be used to update the full summary. Finally, the entity list will be updated.

        Args:
        msgs (list[dict]): a list of dicts, where each dict has the message ('content' key),
            the 'role' key, a 'tokens' key with the number of tokens in the message, plus any other useful metadata.
        context (str): a short description of the chat context. Usually, this is the system
            message of the chat thread.
        """
        pass