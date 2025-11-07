import json
# import ollama
# import llama_cpp
import openai
import requests
from abc import ABC
from typing import Union,ClassVar,Dict,Optional,Any,Literal
from pydantic import BaseModel,TypeAdapter,Field

# boilerplate to allow automatically casting serialized LLMs to the proper class
# use the @register decorator to register classes with this registry
# llm_class_registry = {}

# def register(cls):
#     llm_class_registry[cls.__fields__['llm_class'].default] = cls
#     return cls

class InstructFormat(BaseModel):
    """
    A class to represent the formatting expected by an instruct-tuned LLM.
    """

    name: str = Field(..., description="User-friendly name for this format.")
    message_template: str = Field(..., description="Formattable string giving the layout of a single message.")
    begin_of_text: str = Field(
        default="",
        description="Special text to put at the beginning of a series of instruct messages."
    )
    end_of_turn: str = Field(..., description="Special text to put at the end of every individual message.")
    continue_template: str = Field(..., description="Formattable string for when the LLM should continue the last message.")

    def format_message(self, role, content, eot=True):
        fmt_txt = self.message_template.format(role=role, content=content)
        if eot:
            fmt_txt += self.end_of_turn
        return self.message_template.format(role=role, content=content)

    def format_messages(self, messages, bot=True, eot=True, append_continuation=True, continue_role=None):
        """
        Format a list of message dicts. Each message must have keys 'role' and 'content'.

        Args:
        bot (bool): Should the beginning-of-turn text be added before the messages?
        eot (bool): Should the end-of-turn text be added after the last message? Set to false if
            you want the LLM to continue generating from the end of the last message.
        append_continuation (bool): Should start of next turn be appended? If true, this overrides 
            paramter 'eot' and always includes the end-of-turn on the previous message.
        continue_role (str): If appending start of next turn, what role should be used?
        """
        if bot:
            fmt_msgs = self.begin_of_text
        else:
            fmt_msgs = ""
        if len(messages) > 1:
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

class LLM(BaseModel, ABC):
    """
    Generic large language model interface. Extend this class to implement specific LLM providers.
    """

    llm_class:Literal['base'] = "base"

    model: str = Field(..., description="Name of the LLM to use.")
    sampling_options: Optional[Dict[str, Any]] = Field(
        default={
            "num_predict": 1024,
            "num_ctx": 8192,
            "temperature": 1.0,
            "min_p": 0.1,
            "keep_alive": "15m"
        },
        description="Dictionary of OpenAI-compatible sampling parameters to use."
    )
    instruct_format: Optional[InstructFormat] = Field(
        default=InstructFormat(
            name="Basic Chat",
            begin_of_text = "",
            message_template="{role}:\n{content}",
            end_of_turn="\n\n",
            continue_template="{role}:\n"
            ),
        description="The instruct format this LLM expects."
    )

    class Config:
        arbitrary_types_allowed = True

    def generate(self, prompt, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        prompt (str): The prompt that the LLM should respond to
        stream (bool): Whether the response should be streamed as it is generated

        Returns:
        A generator function if stream is true, otherwise a string containing the response.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")

    def generate_instruct(self, messages, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        messages (str): The chat messages that the LLM should respond to
        stream (bool): Whether the response should be streamed as it is generated

        Returns:
        A generator function if stream is true, otherwise a string containing the response.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")
    
    def count_tokens(self, text:str):
        """
        Count (or estimate) the number of tokens in a given string.

        Args:
        text (str): The input string to tokenize.

        Returns:
        int: The (approximate) number of tokens in the input string.
        """
        raise NotImplementedError("Method must be implemented in a subclass!")

# class OllamaLLM(LLM):
#     """
#     Uses Ollama backend to interact with LLMs.
#     """

#     def __init__(self, model, sampling_options=None, instruct_fmt=None):
#         """
#         Create a new LLM provided by Ollama.

#         Args:
#         model (str): Name of the LLM that Ollama should use.
#         sampling_options (dict): Dictionary of Ollama sampling parameters to use.
#         instruct_fmt (InstructFormat): The instruct format this LLM expects.
#         """
#         self.model = model
#         # if no options specified, we provide some reasonable defaults
#         if sampling_options is None:
#             self.sampling_options = {
#                 "num_predict": 512,
#                 "num_ctx": 4096,
#                 "temperature": 1.0,
#                 "min_p": 0.1,
#                 #"stop": self.stop_words,
#                 "keep_alive": "15m"
#             }
#         else: # use the user-supplied options
#             self.sampling_options = sampling_options
#         # use generic instruct format if none specified
#         if instruct_fmt is None:
#             self.instruct_format = InstructFormat(name="Basic Chat",
#                                                   begin_of_text = "",
#                                                   message_template="{role}:\n{content}",
#                                                   end_of_turn="\n\n",
#                                                   continue_template="{role}:\n")
#         else:
#             self.instruct_format = instruct_fmt

#     def generate(self, prompt, stream=True):
#         """
#         Generate a response to a given text prompt. If stream is true, function returns a generator
#         that yields the response chunks as they become available. Otherwise, the full response is
#         returned as a string.

#         Args:
#         prompt (str): The prompt that the LLM should respond to
#         stream (bool): Whether the response should be streamed as it is generated

#         Returns:
#         A generator function if stream is true, otherwise a string containing the response.
#         """
#         o_options = self.sampling_options.copy()
#         #o_options["stop"] = self.stop_words
#         o_gen = ollama.generate(model=self.model, 
#                                 prompt=prompt,
#                                 stream=stream, raw=True, options=o_options)
#         # return the generator or result, depending on stream parameter
#         return o_gen

#     def generate_instruct(self, messages, respond=True, response_role=None, stream=True):
#         """
#         Generate a response to a given text prompt. If stream is true, function returns a generator
#         that yields the response chunks as they become available. Otherwise, the full response is
#         returned as a string.

#         Args:
#         messages (list[dict]): The chat messages that the LLM should respond to
#         respond (bool): If true, LLM will respond to last message. If false, LLM will
#             continue generating from the end of the last message.
#         response_role (str): The role LLM should use when responding.
#         stream (bool): Whether the response should be streamed as it is generated

#         Returns:
#         A generator function if stream is true, otherwise a string containing the response.
#         """
#         if respond and response_role is None:
#             raise ValueError("Response role must be set in order for LLM to respond!")
#         # format messages
#         if respond:
#             fmt_msgs = self.instruct_format.format_messages(messages, bot=True, eot=True,
#                                                             append_continuation=True,
#                                                             continue_role=response_role)
#         else:
#             fmt_msgs = self.instruct_format.format_messages(messages, bot=True, eot=False,
#                                                             append_continuation=False)
#         return self.generate(prompt=fmt_msgs, stream=stream)

#     def to_json(self):
#         """
#         Write this object out as a JSON object.

#         Returns: a string containing the JSON object
#         """
#         # define state to save
#         settings_to_download = {"model": self.model,
#                                 "sampling_options": self.sampling_options,
#                                 "instruct_format": self.instruct_format.to_json()
#                                 }
#         # dump it to a JSON file
#         return json.dumps(settings_to_download)

#     @classmethod
#     def from_json(cls, json_data):
#         """
#         Load saved session state from a JSON object.
#         Args:
#         json_data (str): JSON object or file containing session data

#         Returns: a new ChatSession object initialized from the JSON data
#         """
#         # load saved state
#         if type(json_data) == str:
#             uploaded_settings = json.loads(json_data)
#         else:
#             uploaded_settings = json.load(json_data)
#         # get model name
#         model_name = uploaded_settings.get('model')
#         # pull sampling options
#         samp_opts = uploaded_settings.get('sampling_options')
#         # read instruct format
#         inst_fmt = InstructFormat.from_json(uploaded_settings.get('instruct_format'))
#         # create new LLM object
#         new_obj = cls(model=model_name, sampling_options=samp_opts, instruct_fmt=inst_fmt)
#         # return object
#         return new_obj


# class LlamaCppPythonLLM(LLM):
#     """
#     Uses the python interface to llama.cpp to interact with LLMs.
#     """

#     def __init__(self, model, sampling_options=None, instruct_fmt=None):
#         """
#         Create a new LLM provided by llama.cpp.

#         Args:
#         model (str): Name of the LLM that Ollama should use.
#         sampling_options (dict): Dictionary of Ollama sampling parameters to use.
#         instruct_fmt (InstructFormat): The instruct format this LLM expects.
#         """
#         self.model = model
#         # if no options specified, we provide some reasonable defaults
#         if sampling_options is None:
#             self.sampling_options = {
#                 "num_predict": 512,
#                 "num_ctx": 4096,
#                 "temperature": 1.0,
#                 "min_p": 0.1,
#                 #"stop": self.stop_words,
#                 "keep_alive": "15m"
#             }
#         else: # use the user-supplied options
#             self.sampling_options = sampling_options
#         # use generic instruct format if none specified
#         if instruct_fmt is None:
#             self.instruct_format = InstructFormat(name="Basic Chat",
#                                                   begin_of_text = "",
#                                                   message_template="{role}:\n{content}",
#                                                   end_of_turn="\n\n",
#                                                   continue_template="{role}:\n")
#         else:
#             self.instruct_format = instruct_fmt

#     def generate(self, prompt, stream=True):
#         """
#         Generate a response to a given text prompt. If stream is true, function returns a generator
#         that yields the response chunks as they become available. Otherwise, the full response is
#         returned as a string.

#         Args:
#         prompt (str): The prompt that the LLM should respond to
#         stream (bool): Whether the response should be streamed as it is generated

#         Returns:
#         A generator function if stream is true, otherwise a string containing the response.
#         """
#         o_options = self.sampling_options.copy()
#         #o_options["stop"] = self.stop_words
#         o_gen = ollama.generate(model=self.model, 
#                                 prompt=prompt,
#                                 stream=stream, raw=True, options=o_options)
#         # return the generator or result, depending on stream parameter
#         return o_gen

#     def generate_instruct(self, messages, respond=True, response_role=None, stream=True):
#         """
#         Generate a response to a given text prompt. If stream is true, function returns a generator
#         that yields the response chunks as they become available. Otherwise, the full response is
#         returned as a string.

#         Args:
#         messages (list[dict]): The chat messages that the LLM should respond to
#         respond (bool): If true, LLM will respond to last message. If false, LLM will
#             continue generating from the end of the last message.
#         response_role (str): The role LLM should use when responding.
#         stream (bool): Whether the response should be streamed as it is generated

#         Returns:
#         A generator function if stream is true, otherwise a string containing the response.
#         """
#         if respond and response_role is None:
#             raise ValueError("Response role must be set in order for LLM to respond!")
#         # format messages
#         if respond:
#             fmt_msgs = self.instruct_format.format_messages(messages, bot=True, eot=True,
#                                                             append_continuation=True,
#                                                             continue_role=response_role)
#         else:
#             fmt_msgs = self.instruct_format.format_messages(messages, bot=True, eot=False,
#                                                             append_continuation=False)
#         return self.generate(prompt=fmt_msgs, stream=stream)

#     def to_json(self):
#         """
#         Write this object out as a JSON object.

#         Returns: a string containing the JSON object
#         """
#         # define state to save
#         settings_to_download = {"model": self.model,
#                                 "sampling_options": self.sampling_options,
#                                 "instruct_format": self.instruct_format.to_json()
#                                 }
#         # dump it to a JSON file
#         return json.dumps(settings_to_download)

#     @classmethod
#     def from_json(cls, json_data):
#         """
#         Load saved session state from a JSON object.
#         Args:
#         json_data (str): JSON object or file containing session data

#         Returns: a new ChatSession object initialized from the JSON data
#         """
#         # load saved state
#         if type(json_data) == str:
#             uploaded_settings = json.loads(json_data)
#         else:
#             uploaded_settings = json.load(json_data)
#         # get model name
#         model_name = uploaded_settings.get('model')
#         # pull sampling options
#         samp_opts = uploaded_settings.get('sampling_options')
#         # read instruct format
#         inst_fmt = InstructFormat.from_json(uploaded_settings.get('instruct_format'))
#         # create new LLM object
#         new_obj = cls(model=model_name, sampling_options=samp_opts, instruct_fmt=inst_fmt)
#         # return object
#         return new_obj

class OpenAILLM(LLM):
    """
    Interact with any OpenAI compatible backend.
    """
    # type name for deserialization
    llm_class:Literal['OpenAILLM'] = "OpenAILLM"

    api_key: str = Field(
        default="sk_fake",
        description="The API key to use; can use an arbitrary string for local endpoints that do not require a key."
    )
    base_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="The URL of the API endpoint."
    )
    # client field is only populated at runtime
    client: Optional[Any] = Field(default=None, exclude=True)

    def model_post_init(self, context:Any) -> None:
        """
        Called to set up the OpenAI client object once the object is initialized.
        """
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=1200,
            max_retries=10
        )

    def generate(self, prompt, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        prompt (str): The prompt that the LLM should respond to
        stream (bool): Whether the response should be streamed as it is generated

        Returns:
        A generator function if stream is true, otherwise a string containing the response.
        """
        # ollama generates dicts with keys 'response' (the text), eval_count, eval_duration (tokens generated and time it took in ms)
        # prompt_eval_count (how much prompt was sent and processed)
        # OpenAI format puts the text in response['choices'][0]['message']['content']
        # TODO: fix the call to pass sampling parameters correctly
        response = self.client.completions.create(
            model=self.model, 
            prompt=prompt, 
            stream=stream,
            # try shoving all sampling parameters through this mechanism to avoid manually
            # specifying the canonical OpenAI ones
            extra_body=self.sampling_options
        )
        
        if not stream:
            ol_dict = {
                'response': response.choices[0].text
            }
            # add generation speed if available
            if response.usage is not None:
                ol_dict['prompt_eval_count'] = response.timings['prompt_n']
                ol_dict['eval_count'] = response.timings['predicted_n']
                # ollama outputs times in nanoseconds for some reason...
                ol_dict['eval_duration'] = response.timings['predicted_ms']*1.0e6
            yield ol_dict
        else:
            for chunk in response:
                ol_dict = {
                    'response': chunk.choices[0].text
                }
                # add generation speed if available
                if chunk.usage is not None:
                    ol_dict['prompt_eval_count'] = chunk.timings['prompt_n']
                    ol_dict['eval_count'] = chunk.timings['predicted_n']
                    # ollama outputs times in nanoseconds for some reason...
                    ol_dict['eval_duration'] = chunk.timings['predicted_ms']*1.0e6
                yield ol_dict
        # if not stream:
        #     return response['choices'][0]['message']['content']
        # else:
        #     for chunk in response:
        #         ol_dict = {
        #             'response': chunk['choices'][0]['message']['content']
        #         }
        #         yield ol_dict

    def generate_instruct(self, messages, respond=True, response_role=None, stream=True):
        """
        Generate a response to a given text prompt. If stream is true, function returns a generator
        that yields the response chunks as they become available. Otherwise, the full response is
        returned as a string.

        Args:
        messages (list[dict]): The chat messages that the LLM should respond to
        respond (bool): If true, LLM will respond to last message. If false, LLM will
            continue generating from the end of the last message.
        response_role (str): The role LLM should use when responding.
        stream (bool): Whether the response should be streamed as it is generated

        Returns:
        A generator function if stream is true, otherwise a string containing the response.
        """
        # TODO: convert to optionally use the actual chat API?
        if respond and response_role is None:
            raise ValueError("Response role must be set in order for LLM to respond!")
        # format messages
        if respond:
            fmt_msgs = self.instruct_format.format_messages(messages, bot=True, eot=True,
                                                            append_continuation=True,
                                                            continue_role=response_role)
        else:
            fmt_msgs = self.instruct_format.format_messages(messages, bot=True, eot=False,
                                                            append_continuation=False)
        # now generate using regular completion endpoint
        result = self.generate(prompt=fmt_msgs, stream=stream)
        return result

    def count_tokens(self, text:str):
        """
        Count the number of tokens in a given string using the /tokenize upstream endpoint, if available on this server.
        This only really works with llama-swap.

        Args:
        text (str): The input string to tokenize.

        Returns:
        int: The number of tokens in the input string.
        """
        headers = {'Content-Type': 'text/plain'}

        # Create the request payload
        payload = {'content': text}
        # get the upstream URL, llama-swap doesn't support directly
        llm_url = self.client.base_url
        tk_url = llm_url.scheme + "://" + llm_url.netloc.decode() + "/upstream/" + self.model + "/tokenize"
        # Send the POST request
        response = requests.post(tk_url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract the number of tokens from the response
            tokens = data['tokens']

            return len(tokens)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

# a union type covering the possible LLM types
# you can discriminate it by using Field(discriminator='llm_class')
LLMType = Union[LLM, OpenAILLM]

if __name__ == "__main__":
    # print(str(llm_class_registry))

    # test loading an instruct format from file
    with open("./instruct_formats/gemma_chat.json", mode="r") as file:
        inst_data = file.read()
    inst_fmt = InstructFormat.model_validate_json(inst_data)
    # inst_fmt = InstructFormat.from_json(open("./instruct_formats/gemma_chat.json", mode="r"))
    print("Instruct format:\n" + inst_fmt.model_dump_json(indent=2))

    # test OpenAILLM
    samp_params = {
        "temperature": 1.6,
        "min_p": 0.01,
        "max_tokens": 12
    }
    llm = OpenAILLM(
        model="gemma-3-4B-it-UD-Q4_K_XL-cpu",
        sampling_options=samp_params,
        instruct_fmt=inst_fmt
    )
    # test converting to JSON
    llm_txt = llm.model_dump_json(indent=2)
    print(llm_txt)

    # test converting back
    rehydrated_llm = OpenAILLM.model_validate_json(llm_txt)
    
    # print("Generating in instruct mode...")
    # test_messages = [
    #     {
    #         "role": "user",
    #         "content": "I'm a cat! What are you?"
    #     }
    # ]
    # response = llm.generate_instruct(
    #     messages=test_messages,
    #     respond=True,
    #     response_role="assistant",
    #     stream=False
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
       
    # print("Generating in raw mode...") 
    # response = llm.generate(
    #     prompt="I'm a cat, what are you?",
    #     stream=False
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
    
    # # Streaming output

    # print("Streaming in instruct mode...")
    # test_messages = [
    #     {
    #         "role": "user",
    #         "content": "I'm a cat! What are you?"
    #     }
    # ]
    # response = llm.generate_instruct(
    #     messages=test_messages,
    #     respond=True,
    #     response_role="assistant",
    #     stream=True
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
       
    # print("Streaming in raw mode...") 
    # response = llm.generate(
    #     prompt="I'm a cat, what are you?",
    #     stream=True
    # )
    # print(response)
    # for chunk in response:
    #     print(chunk)
    
    