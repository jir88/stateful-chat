import re
import json
import ollama
import chromadb
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChatSession:
    """
    A class to represent a chat session.

    Attributes:
    session_id (str): The unique identifier of the chat session.
    messages (list): A list of messages sent in the chat session.
    stop_words (list): A list of stop words to filter out from the messages.
    llm (object): An LLM object for language processing tasks.
    """

    def __init__(self, session_id):
        """
        Initializes the chat session with a given session ID.

        Args:
        session_id (str): The unique identifier of the chat session.
        """
        self.session_id = session_id
        self.instruct_format = InstructFormat(name="Basic Chat",
                                              message_template="{role}:\n{content}",
                                              end_of_turn="\n\n",
                                              continue_template="{role}:\n")
        self.system_prompt = None
        self.messages = []
        self.stop_words = []
        self.llm = None
        self.user_role = "user"
        self.ai_role = "assistant"
        # memory
        self.archived_messages = []
        self.message_summaries = []
        self.full_summary = None
        self.entity_list = None
        # memory prompts
        self.prompt_msg_summary = "Concisely summarize these messages. Include all relevant details. Reference context from prior summaries where relevant, but focus on the most recent messages. Match the tense and perspective of the story."
        self.prompt_full_summary = "Concisely summarize the plot of the story so far. Include all relevant details. Mention any unresolved plot threads. Match the tense and perspective of the story."
        self.prompt_entity_list = "Provide a list of all entities mentioned thus far and a brief description of each. Include entities that have been mentioned but have not yet appeared. For character entities, include a brief description of their personality. Write more detailed descriptions for more important entities and characters."
        # prepare for lazy vector database connection
        self.vec_db_col = None
        # specify embedding model
        self.embed_mdl = None
        # FIXME: specify as metadaata in the collection itself?
        self.sampling_options = {
            "num_predict": 512,
            "num_ctx": 3072,
            #"temperature": 0.8,
            #"top_p": 0.92,
            #"top_k": 100,
            #"repeat_penalty": 1.12,
            "temperature": 1.2,
            "min_p": 0.1,
            "stop": self.stop_words
        }

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

    def messages_to_summary(self, start_idx, stop_idx):
        """
        Generate a summary of some messages and add it to
        the list of message summaries.

        Args:
        start_idx (int): index of first message to summarize (inclusive)
        end_idx (int): index of last message to summarize (exclusive)
        """
        # make list of messages to summarize
        summ_idx = list(range(start_idx, stop_idx))
        # format messages
        fmt_msgs = "<|begin_of_text|>"
        # start with the system prompt, if any
        full_sys_prompt = self.compile_system_prompt().strip()
        if len(full_sys_prompt) > 0:
            #fmt_msgs += self.instruct_format.format_message(role=self.instruct_format.system_role,
            #                                                content = full_sys_prompt, eot=True)
            fmt_msgs += "<|start_header_id|>system<|end_header_id|>\n\n" + full_sys_prompt + "<|eot_id|>"
        # now add the actual messages
        # fmt_msgs += self.instruct_format.format_messages(self.messages[start_idx:stop_idx], append_continuation=False, eot=True)
        for i in summ_idx:
            fmt_msgs += "<|start_header_id|>" + self.messages[i]['role'] + "<|end_header_id|>\n\n" + self.messages[i]['content'] + "<|eot_id|>"
        # add summarize prompt
        #fmt_msgs += self.instruct_format.format_messages(
        #    {
        #        'role': self.instruct_format.system_role,
        #        'content': self.prompt_msg_summary
        #    },
        #    append_continuation=True, continue_role=self.ai_role)
        fmt_msgs += "<|start_header_id|>system<|end_header_id|>\n\n"
        fmt_msgs += self.prompt_msg_summary
        # add tags to prompt for LLM response using current AI role
        fmt_msgs += "<|start_header_id|>" + self.ai_role + "<|end_header_id|>\n\n"
        # generate summary
        o_options = self.sampling_options.copy()
        o_options["temperature"] = 0.8 # lower than for generating new text
        o_options["stop"] = self.stop_words
        o_gen = ollama.generate(model=self.llm, 
                                prompt=fmt_msgs,
                                stream=False, raw=True, options=o_options)
        # add summary to list of summaries
        msg_summ = {
            "id": len(self.message_summaries),
            "content": o_gen['response']
        }
        self.message_summaries.extend([msg_summ])

    def update_full_summary(self, start_idx, stop_idx):
        """
        Generate or update a complete summary of past summarized message
        chunks, adding information from a set of messages. Also updates
        the summary list of entities mentioned thus far.

        Args:
        start_idx (int): index of first message to summarize (inclusive)
        end_idx (int): index of last message to summarize (exclusive)
        """
        # make list of messages to summarize
        summ_idx = list(range(start_idx, stop_idx))
        # format messages
        fmt_msgs = "<|begin_of_text|>"
        # start with the system prompt, if any
        full_sys_prompt = self.compile_system_prompt().strip()
        if len(full_sys_prompt) > 0:
            #fmt_msgs += self.instruct_format.format_message(role=self.instruct_format.system_role,
            #                                                content = full_sys_prompt)
            fmt_msgs += "<|start_header_id|>system<|end_header_id|>\n\n" + full_sys_prompt + "<|eot_id|>"
        # now add the actual messages
        # fmt_msgs += self.instruct_format.format_messages(self.messages[start_idx:stop_idx], append_continuation=False)
        for i in summ_idx:
            fmt_msgs += "<|start_header_id|>" + self.messages[i]['role'] + "<|end_header_id|>\n\n" + self.messages[i]['content'] + "<|eot_id|>"
        # add summarize prompt
        #fmt_msgs += self.instruct_format.format_messages(
        #    {
        #        'role': self.instruct_format.system_role,
        #        'content': self.prompt_full_summary
        #    },
        #    append_continuation=True, continue_role=self.instruct_format.ai_role)
        fmt_msgs += "<|start_header_id|>system<|end_header_id|>\n\n"
        fmt_msgs += self.prompt_full_summary
        # add tags to prompt for LLM response using current AI role
        fmt_msgs += "<|start_header_id|>" + self.ai_role + "<|end_header_id|>\n\n"
        # generate summary
        o_options = self.sampling_options.copy()
        o_options["temperature"] = 0.8 # lower than for generating new text
        o_options["stop"] = self.stop_words
        o_gen = ollama.generate(model=self.llm, 
                                prompt=fmt_msgs,
                                stream=False, raw=True, options=o_options)
        # replace top-level summary with updated one
        msg_summ = o_gen['response']
        self.full_summary = msg_summ
        # now generate list of entities
        entity_prompt = fmt_msgs + msg_summ
        # add entity prompt
        #entity_prompt += self.instruct_format.format_messages(
        #    {
        #        'role': self.instruct_format.system_role,
        #        'content': self.prompt_entity_list
        #    },
        #    append_continuation=True, continue_role=self.instruct_format.ai_role)
        entity_prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
        entity_prompt += self.prompt_entity_list
        # add tags to prompt for LLM response using current AI role
        entity_prompt += "<|start_header_id|>" + self.ai_role + "<|end_header_id|>\n\n"
        # generate entity list
        o_gen = ollama.generate(model=self.llm, 
                                prompt=entity_prompt,
                                stream=False, raw=True, options=o_options)
        # replace current entity list with updated one
        entity_list = o_gen['response']
        self.entity_list = entity_list

    def initialize_vector_db(self):
        """
        Set up the vector database for archiving messages and summaries.
        """
        # set up vector database connection
        if self.vec_db_col is None:
            vec_db = chromadb.PersistentClient(path="./.chroma")
            # define embedding function
            self.embed_mdl = SentenceTransformerEmbeddingFunction(model_name="nomic-ai/nomic-embed-text-v1.5",
                                                                  trust_remote_code=True)
            self.vec_db_col = vec_db.get_or_create_collection(name=self.session_id, 
                                                              embedding_function = self.embed_mdl,
                                                             metadata={"hnsw:space": "cosine"})
        else:
            print("Vector database has already been initialized!")

    def embed_text(self, msg_dict, msg_type):
        # initialize the database if needed
        if self.vec_db_col is None:
            self.initialize_vector_db()
        # construct metadata
        msg_meta = {'index': msg_dict['id'],
                    'type': msg_type
                    }
        # if this is a message, add role
        if msg_type == "message":
            msg_meta['role'] = msg_dict['role']
        # for unique ID, combine type and index
        msg_id = msg_type + str(msg_dict['id'])
        # add to the database
        self.vec_db_col.upsert(ids = msg_id, documents = "search_document: " + msg_dict['content'], metadatas=msg_meta)
        # split text and embed that too 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 50,
            length_function = len,
            is_separator_regex = False
        )
        chunks = text_splitter.split_text(msg_dict['content'])
        # only do this if there is more than one chunk!
        if len(chunks) > 1:
            for i in range(0, len(chunks)):
                msg_meta['chunk_index'] = i
                msg_meta['parent'] = msg_id
                chunk_id = msg_id + "c" + str(i)
                self.vec_db_col.upsert(ids = chunk_id, documents = "search_document: " + chunks[i], metadatas=msg_meta)

    def update_full_vector_db(self):
        """
        Clean up vector database and embed again.
        """
        # initialize the database if needed
        if self.vec_db_col is None:
            self.initialize_vector_db()
        else:
            # otherwise nuke the whole collection and start over
            vec_db = chromadb.PersistentClient(path="./.chroma")
            vec_db.delete_collection(name=self.session_id)
            self.vec_db_col = None
            # initialize the new database
            self.initialize_vector_db()
        # FIXME: clear out old embeddings here!
        # embed all archived messages
        for msg in self.archived_messages:
            self.embed_text(msg, "message")
        # embed all messages
        for msg in self.messages:
            self.embed_text(msg, "message")
        # embed all message summaries
        for msg in self.message_summaries:
            self.embed_text(msg, "message_summary")
        # embed entity list
        pass
    
    def embed_messages(self, idx_start, idx_stop):
        """
        Embed one or more messages and add them to the associated vector database.

        Args:
        idx_start (int): Index of first message to embed (inclusive)
        idx_stop (int): Index of last message to embed (exclusive)
        """
        # initialize the database if needed
        if self.vec_db_col is None:
            self.initialize_vector_db()
        for i in range(idx_start, idx_stop):
            msg = self.messages[i]
            #emb = embeddings.embed_query(msg['content'])
            msg_meta = {'index': i,
                        'type': "message",
                        'role': msg['role']
                       }
            self.vec_db_col.upsert(ids = "msg" + str(i), documents = "search_document: " + msg['content'], metadatas=msg_meta)

    def embed_message_summaries(self, idx_start, idx_stop):
        """
        Embed one or more message summaries and add them to the associated vector database.

        Args:
        idx_start (int): Index of first message summary to embed (inclusive)
        idx_stop (int): Index of last message summary to embed (exclusive)
        """
        # initialize the database if needed
        if self.vec_db_col is None:
            self.initialize_vector_db()
        for i in range(idx_start, idx_stop):
            msg = self.message_summaries[i]
            #emb = embeddings.embed_query(msg['content'])
            msg_meta = {'index': i,
                        'type': "message_summary"
                       }
            self.vec_db_col.upsert(ids = "msg_sum" + str(i), documents = "search_document: " + msg['content'], metadatas=msg_meta)

    def query_vector_db(self, query_text, n_results):
        # initialize the database if needed
        if self.vec_db_col is None:
            self.initialize_vector_db()
        return self.vec_db_col.query(query_texts="search_query: " + query_text, n_results=n_results)
    
    def get_response(self, stream=True):
        o_options = self.sampling_options.copy()
        o_options["stop"] = self.stop_words
        o_gen = ollama.generate(model=self.llm, 
                                prompt=self.format_instruct(),
                                stream=stream, raw=True, options=o_options)
        # return the generator or result, depending on stream parameter
        return o_gen

    def continue_response(self, stream=True):
        """
        Continue generating from the end of the most recent message.
        """
        o_options = self.sampling_options.copy()
        o_options["stop"] = self.stop_words
        # generate prompt
        cont_prompt = self.format_instruct()
        # cont_prompt += self.instruct_format.continue_template.format(role=self.instruct_format.ai_role)
        rem_len = len("<|eot_id|><|start_header_id|>" + self.ai_role + "<|end_header_id|>\n\n")
        # "<|eot_id|>"
        # add tags to prompt for LLM response using current AI role
        #result += "<|start_header_id|>" + self.ai_role + "<|end_header_id|>\n\n"
        o_gen = ollama.generate(model=self.llm, 
                                prompt=cont_prompt[:-rem_len],
                                stream=stream, raw=True, options=o_options)
        # return the generator or result, depending on stream parameter
        return o_gen

    def get_rag_response(self, n_msg=2, rag_thresh=0.3, n_results=1, stream=True):
        """
        Inject RAG results and generate AI response.
        n_msg: How many messages to use when querying database
        rag_thresh: Maximum distance threshold for results. Results with greater distance will not be returned.
        n_results: Maximum number of results to return.
        """
        context_docs = []
        # for each query message
        for msg in self.messages[-n_msg:]:
            # get results
            msg_res = self.query_vector_db(msg['content'], n_results)
            # append results within threshold to dict
            for i in range(0, len(msg_res.get('documents'))):
                if msg_res.get('distances')[i] < rag_thresh:
                    context_docs.append(msg_res.get('documents')[i])
        # get instruct formatted history
        cont_prompt = self.format_instruct()
        # append formatted RAG results
        cont_prompt += "The following context may be relevant to your response:\n"
        for cd in context_docs:
            cont_prompt += cd + "\n"
        cont_prompt += "Response:\n"
        # generate response
        o_gen = ollama.generate(model=self.llm, 
                                prompt=cont_prompt[:-rem_len],
                                stream=stream, raw=True, options=o_options)
        # return the generator or result, depending on stream parameter
        return o_gen

    def append_message(self, role, content):
        # give unique ID based on order of generation
        idx = len(self.archived_messages) + len(self.messages)
        self.messages.append({"id": idx, "role": role, "content": content})

    def compile_system_prompt(self):
        """
        Combine raw prompt, the most recent message summary, and the entity list
        into a full system prompt.
        """
        # start with the system prompt, if any
        full_sys_prompt = ""
        if self.system_prompt is not None:
            full_sys_prompt += self.system_prompt.strip()
        # add top-level summary, if any
        if self.full_summary is not None:
            full_sys_prompt += "\n\nComplete summary of all previous messages:\n" + self.full_summary
        # add entity list, if any
        if self.entity_list is not None:
            full_sys_prompt += "\n\nEntitites mentioned previously:\n" + self.entity_list
        # add latest message summary, if any
        if len(self.message_summaries) > 0:
            full_sys_prompt += "\n\nSummary of recent previous messages:\n" + self.message_summaries[-1]['content']
        return full_sys_prompt

    def format_instruct(self, msg_idx=None):
        """
        Combine messages from this session into a chat prompt. Subclasses can
        override this method to do more complex chat formatting or use 
        specific instruct formats. Defaults to the Llama 3(.1) format.

        Formatting will automatically inject the system prompt and the most
        recent chat summary.

        Args:
        msg_idx (list): A list of message indices to include, or None to use all.
        """
        # if no messages specified, use all of them
        if msg_idx is None:
            msg_idx = range(0, len(self.messages))
        result = "<|begin_of_text|>"
        # start with the system prompt, if any
        full_sys_prompt = self.compile_system_prompt().strip()
        if len(full_sys_prompt) > 0:
            result += "<|start_header_id|>system<|end_header_id|>\n\n" + full_sys_prompt + "<|eot_id|>"
        # now add the actual messages
        for i in msg_idx:
            result += "<|start_header_id|>" + self.messages[i]['role'] + "<|end_header_id|>\n\n" + self.messages[i]['content'] + "<|eot_id|>"
        # add tags to prompt for LLM response using current AI role
        result += "<|start_header_id|>" + self.ai_role + "<|end_header_id|>\n\n"
        return result

    def format_readable(self):
        """
        Convert all messages in this session into a human-readable and editable
        format. Message roles are displayed in curly brackets: {{role}} with
        message text following. Leading and trailing whitespace are ignored.
        """
        result = ""
        for i in range(0, len(self.messages)):
            result += "{{" + self.messages[i]['role'] + "}}\n" + self.messages[i]['content'] + "\n"
        return result

    def import_readable(self, readable_text):
        """
        Parse messages exported by format_readable and use them to replace any
        existing messages in this chat session.

        Args:
        readable_text (str): The formatted messages to be parsed.
        """
        re_role = re.compile(r"{{(.+?)}}")
        # splitting with capturing groups returns the roles too
        msg_parts = re_role.split(readable_text)
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
                                "instruct_format": self.instruct_format.to_json(),
                                "system_prompt": self.system_prompt,
                                "messages": self.messages,
                                "stop_words": self.stop_words,
                                "llm": self.llm,
                                "user_role": self.user_role,
                                "ai_role": self.ai_role,
                                "archived_messages": self.archived_messages,
                                "message_summaries": self.message_summaries,
                                "full_summary": self.full_summary,
                                "entity_list": self.entity_list,
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
        uploaded_settings = json.load(json_data)
        # create new session object
        new_session = cls(session_id=uploaded_settings["session_id"])
        # load the instruct format
        new_session.instruct_format = uploaded_settings.get("instruct_format")
        if new_session.instruct_format is None:
            # need to add default, which is Llama 3
            new_session.instruct_format = InstructFormat("Llama 3 Chat",
                                                         "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>",
                                                         "<|start_header_id|>{role}<|end_header_id|>\n\n")
        else:
            # parse existing format
            new_session.instruct_format = InstructFormat.from_json(new_session.instruct_format)
        # load baseline system prompt
        new_session.system_prompt = uploaded_settings["system_prompt"]
        # load prompt text
        new_session.messages = uploaded_settings["messages"]
        # load stop words
        new_session.stop_words = uploaded_settings["stop_words"]
        # load LLM name
        new_session.llm = uploaded_settings["llm"]
        # load user role
        new_session.user_role = uploaded_settings["user_role"]
        # load AI role
        new_session.ai_role = uploaded_settings["ai_role"]
        # load archived messages
        new_session.archived_messages = uploaded_settings["archived_messages"]
        # add archived message IDs if they are missing
        if len(new_session.archived_messages) > 0 and new_session.archived_messages[0].get('id') is None:
            print("Fixing archived message IDs.")
            for i in range(0, len(new_session.archived_messages)):
                new_session.archived_messages[i]['id'] = i
        # also add current message IDs if those are missing
        if len(new_session.messages) > 0 and new_session.messages[0].get('id') is None:
            print("Fixing current message IDs.")
            for i in range(0, len(new_session.messages)):
                new_session.messages[i]['id'] = i + len(new_session.archived_messages)
        # load message summaries
        new_session.message_summaries = uploaded_settings["message_summaries"]
        # if summaries are stored as strings, update to dicts with indices
        if len(new_session.message_summaries) > 0 and str(new_session.message_summaries[0].__class__) != "<class 'dict'>":
            print(str(new_session.message_summaries[0].__class__))
            for i in range(0, len(new_session.message_summaries)):
                new_session.message_summaries[i] = { "id": i, "content": new_session.message_summaries[i] }
        # load full summary
        new_session.full_summary = uploaded_settings["full_summary"]
        # load entity list
        new_session.entity_list = uploaded_settings["entity_list"]
        # load prompts
        new_session.prompt_msg_summary = uploaded_settings.get("prompt_msg_summary")
        if new_session.prompt_msg_summary is None:
            new_session.prompt_msg_summary = "Concisely summarize these messages. Include all relevant details. Reference context from prior summaries where relevant, but focus on the most recent messages. Match the tense and perspective of the story."
        new_session.prompt_full_summary = uploaded_settings.get("prompt_full_summary")
        if new_session.prompt_full_summary is None:
            new_session.prompt_full_summary = "Concisely summarize the plot of the story so far. Include all relevant details. Mention any unresolved plot threads. Match the tense and perspective of the story."
        new_session.prompt_entity_list = uploaded_settings.get("prompt_entity_list")
        if new_session.prompt_entity_list is None:
            new_session.prompt_entity_list = "Provide a list of all entities mentioned thus far and a brief description of each. Include entities that have been mentioned but have not yet appeared. For character entities, include a brief description of their personality. Write more detailed descriptions for more important entities and characters."
        # return object
        return new_session

    @classmethod
    def update_outdated_session(cls, json_data):
        """
        Fix
        """
        pass

class InstructFormat:
    """
    A class to represent the formatting expected by an instruct-tuned LLM.
    """
    
    def __init__(self, name, message_template, end_of_turn, continue_template):
        """
        Initializes the instruct format.

        Args:
        session_id (str): The unique identifier of the chat session.
        """
        self.name = name
        self.message_template = message_template
        self.end_of_turn = end_of_turn
        self.continue_template = continue_template

    def format_message(self, role, content, eot=True):
        fmt_txt = self.message_template.format(role=role, content=content)
        if eot:
            fmt_txt += self.end_of_turn
        return self.message_template.format(role=role, content=content)

    def format_messages(self, messages, eot=True, append_continuation=True, continue_role=None):
        """
        Format a list of message dicts. Each message must have keys 'role' and 'content'.
        
        append_continuation: Should start of next turn be appended? If true, this overrides 
            paramter 'eot' and always includes the end-of-turn on the previous message.
        """
        fmt_msgs = ""
        for msg in messages[:-1]:
            fmt_msgs += self.message_template.format(role=msg['role'], content=msg['content'])
            # always put end-of-turn between messages
            fmt_msgs += self.end_of_turn
        fmt_msgs += self.message_template.format(role=messages[-1]['role'], content=messages[-1]['content'])
        # add end-of-turn tokens after last message, if requested
        if eot:
            fmt_msgs += self.end_of_turn
        # add the beginning of an AI response
        if append_continuation:
            # force end-of-turn
            if not eot:
                fmt_msgs += self.end_of_turn
            # if no role specified, default to AI
            if continue_role is None:
                continue_role = self.ai_role
            fmt_msgs += self.continue_template.format(role=continue_role)
        return fmt_msgs

    def to_json(self):
        """
        Write this object out as a JSON object.

        Returns: a string containing the JSON object
        """
        # define state to save
        settings_to_download = {
            "name": self.name,
            "message_template": self.message_template,
            "end_of_turn": self.end_of_turn,
            "continue_template": self.continue_template
            }
        # dump it to a JSON file
        return json.dumps(settings_to_download)

    @classmethod
    def from_json(cls, json_data):
        """
        Load saved instruct format from a JSON object.
        Args:
        json_data (str): JSON object or file containing instruct formatting data

        Returns: a new InstructFormat object initialized from the JSON data
        """
        # load saved state
        # if data is a string
        if type(json_data) == str:
            uploaded_settings = json.loads(json_data)
        else:
            uploaded_settings = json.load(json_data)
        # create new session object
        new_fmt = cls(name=uploaded_settings["name"],
                      message_template=uploaded_settings["message_template"],
                      end_of_turn=uploaded_settings["end_of_turn"],
                     continue_template=uploaded_settings["continue_template"]
                     )
        # return object
        return new_fmt

