#!/usr/bin/env python3
"""
saved_chat_upgrade.py

Reads older chat data formats and upgrades them to the latest format version.

Usage:
    python saved_chat_upgrade.py input.json output.json
"""

import argparse
import json
import sys
from pathlib import Path

from typing import Dict,Any
from pydantic import ValidationError

from stateful_chat.entity import SimpleEntityManager
from stateful_chat.llm import LLM,OpenAILLM,InstructFormat
from stateful_chat.manager import HierarchicalSummaryMemory,HierarchicalSummaryManager,ChatThread

# def chat_manager_from_json(cls, json_data):
#     """
#     Load saved default session manager state from a JSON object.
#     Args:
#     json_data (str): JSON object or file containing session data

#     Returns: a new StatefulChatManager object initialized from the JSON data
#     """
#     # load saved state
#     uploaded_settings = json.load(json_data)
#     # if this is an old format, try to recover it
#     if uploaded_settings.get('chat_thread') is None:
#         return StatefulChatManager._recover_old_json_format(uploaded_settings)
#     # initialize LLM
#     # TODO: use some dynamic loading to handle other classes
#     llm = OpenAILLM.from_json(uploaded_settings.get('llm'))
#     # create new memory object
#     new_obj = cls(llm=llm)
#     # load chat thread
#     new_obj.chat_thread = ChatThread.from_json(uploaded_settings.get('chat_thread'))
#     # load chat memory
#     new_obj.chat_memory = LLMSummaryMemory.from_json(uploaded_settings.get('chat_memory'))
#     # return object
#     return new_obj

# def _recover_old_json_format(cls, uploaded_settings):
#     """
#     Upgrade an old JSON file to the current format.
#     """
#     # load instruct format
#     inst_fmt = uploaded_settings.get('instruct_format')
#     if inst_fmt is None:
#         # need to add default, which is Llama 3
#         inst_fmt = InstructFormat(name="Llama 3 Chat",
#                                     message_template="<|start_header_id|>{role}<|end_header_id|>\n\n{content}",
#                                     begin_of_text="",
#                                     end_of_turn="<|eot_id|>",
#                                     continue_template="<|start_header_id|>{role}<|end_header_id|>\n\n")
#     else:
#         # parse existing format
#         inst_fmt = json.loads(inst_fmt)
#         inst_fmt = InstructFormat(name=inst_fmt['name'],
#                                     message_template=inst_fmt['message_template'],
#                                     # old versions didn't have BoT
#                                     begin_of_text="",
#                                     end_of_turn=inst_fmt['end_of_turn'],
#                                     continue_template=inst_fmt['continue_template'])
    
#     # initialize LLM
#     llm = OpenAILLM(model=uploaded_settings["llm"],
#                     # default sampling options
#                     sampling_options=None,
#                     instruct_fmt=inst_fmt
#                     )
#     # create new chat manager
#     new_manager = cls(llm=llm)
    
#     # set up chat thread
#     new_manager.chat_thread = ChatThread(session_id=uploaded_settings["session_id"])
#     new_manager.chat_thread.system_prompt = uploaded_settings["system_prompt"]
#     new_manager.chat_thread.messages = uploaded_settings["messages"]
#     new_manager.chat_thread.user_role = uploaded_settings["user_role"]
#     new_manager.chat_thread.ai_role = uploaded_settings["ai_role"]
#     new_manager.chat_thread.archived_messages = uploaded_settings["archived_messages"]
#     # add archived message IDs if they are missing
#     if len(new_manager.chat_thread.archived_messages) > 0 and new_manager.chat_thread.archived_messages[0].get('id') is None:
#         print("Fixing archived message IDs.")
#         for i in range(0, len(new_manager.chat_thread.archived_messages)):
#             new_manager.chat_thread.archived_messages[i]['id'] = i
#     # also add current message IDs if those are missing
#     if len(new_manager.chat_thread.messages) > 0 and new_manager.chat_thread.messages[0].get('id') is None:
#         print("Fixing current message IDs.")
#         for i in range(0, len(new_manager.chat_thread.messages)):
#             new_manager.chat_thread.messages[i]['id'] = i + len(new_manager.chat_thread.archived_messages)
    
#     # set up session memory using the main LLM
#     # use a copy, though, so we can use different settings for them in the future
#     new_manager.chat_memory = LLMSummaryMemory(llm=copy.deepcopy(llm))
#     # import message summaries
#     new_manager.chat_memory.message_summaries = uploaded_settings["message_summaries"]
#     # if summaries are stored as strings, update to dicts with indices
#     if len(new_manager.chat_memory.message_summaries) > 0 and str(new_manager.chat_memory.message_summaries[0].__class__) != "<class 'dict'>":
#         print(str(new_manager.chat_memory.message_summaries[0].__class__))
#         for i in range(0, len(new_manager.chat_memory.message_summaries)):
#             new_manager.chat_memory.message_summaries[i] = { 
#                 "id": i, 
#                 "content": new_manager.chat_memory.message_summaries[i]
#             }
#     # add full summary
#     new_manager.chat_memory.full_summary = uploaded_settings["full_summary"]
#     # load entity list
#     new_manager.chat_memory.entity_list = uploaded_settings["entity_list"]
#     # memory prompts
#     new_manager.chat_memory.init_sys_prompt = "You are an expert summarizer. You will summarize the following messages. You will also use the messages to update a running summary of the whole previous exchange. The following messages are a conversation between {ai} and {user}.\n\nContext:\n"
#     new_manager.chat_memory.prompt_msg_summary = uploaded_settings.get("prompt_msg_summary")
#     if new_manager.chat_memory.prompt_msg_summary is None:
#         new_manager.chat_memory.prompt_msg_summary = "Concisely summarize these messages. Include all relevant details. Reference context from prior summaries where relevant, but focus on the most recent messages. Match the tense and perspective of the story."
#     new_manager.chat_memory.prompt_full_summary = uploaded_settings.get("prompt_full_summary")
#     if new_manager.chat_memory.prompt_full_summary is None:
#         new_manager.chat_memory.prompt_full_summary = "Concisely summarize all messages so far. Base this summary on the previous full summary. Include all relevant details. Mention any unresolved discussion topics."
#     new_manager.chat_memory.prompt_entity_list = uploaded_settings.get("prompt_entity_list")
#     if new_manager.chat_memory.prompt_entity_list is None:
#         new_manager.chat_memory.prompt_entity_list = "Provide a list of all entities mentioned thus far and a brief description of each. For people, include a brief description of their personalities. Write more detailed descriptions for more important entities."
#     # return object
#     return new_manager

def instruct_format_from_json(json_data):
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
    new_fmt = InstructFormat(
        name=uploaded_settings["name"],
        begin_of_text=uploaded_settings["begin_of_text"],
        message_template=uploaded_settings["message_template"],
        end_of_turn=uploaded_settings["end_of_turn"],
        continue_template=uploaded_settings["continue_template"]
    )
    # return object
    return new_fmt

def llm_from_json(json_data):
    """
    Load saved LLM from a JSON object.
    Args:
    json_data (str): JSON object or file containing session data

    Returns: a new ChatSession object initialized from the JSON data
    """
    # load saved state
    if type(json_data) == str:
        uploaded_settings = json.loads(json_data)
    else:
        uploaded_settings = json.load(json_data)
    # get model name
    model_name = uploaded_settings.get('model')
    # pull sampling options
    samp_opts = uploaded_settings.get('sampling_options')
    # read instruct format
    inst_fmt = instruct_format_from_json(uploaded_settings.get('instruct_format'))
    # create new LLM object
    new_obj = OpenAILLM(model=model_name, sampling_options=samp_opts, instruct_fmt=inst_fmt)
    # return object
    return new_obj

def chat_thread_from_json(json_data):
    """
    Load saved chat thread from a JSON object.
    Args:
    json_data (str): JSON object or file containing the data

    Returns: a new ChatThread object initialized from the JSON data
    """
    # load saved state
    if type(json_data) == str:
        uploaded_settings = json.loads(json_data)
    else:
        uploaded_settings = json.load(json_data)
    # create new thread object
    new_obj = ChatThread(
        session_id=uploaded_settings.get('session_id'),
        system_prompt = uploaded_settings.get('system_prompt'),
        messages = uploaded_settings.get('messages'),
        archived_messages = uploaded_settings.get('archived_messages'),
        user_role = uploaded_settings["user_role"],
        ai_role = uploaded_settings["ai_role"]
    )
    # return object
    return new_obj

def hier_mem_from_json(json_data):
    """
    Load saved hierarchical summary memory from a JSON object.
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
    try:
        llm = OpenAILLM.model_validate_json(uploaded_settings.get('summary_llm'))
    except ValidationError as e:
        print("LLM is stored in old format, attempting to recover...")
        llm = llm_from_json(uploaded_settings.get('summary_llm'))
    # load associated chat thread
    ct = chat_thread_from_json(uploaded_settings.get('chat_thread'))
    # check what type of entity manager we have
    entity_manager = uploaded_settings.get('entity_manager', None)
    entity_list = uploaded_settings.get('entity_list', None)
    if entity_list is None and entity_manager is None:
        # no manager, just make a default one
        print("No entity list found. Making blank one...")
        entity_list = SimpleEntityManager(llm=llm)
    elif entity_list is None and isinstance(entity_manager, str):
        # new version with serialized JSON in 'entity_manager' field
        print("Entity list in JSON format. Importing with pydantic...")
        entity_list = SimpleEntityManager.model_validate_json(entity_manager)
    elif isinstance(entity_list, str):
        # raw string entity list, so we'll put it in a manager
        print("Raw text entity list. Converting to entity manager...")
        el_obj = SimpleEntityManager(
            llm=llm,
            entity_list=entity_list
        )
        entity_list = el_obj
        # get custom prompt, if any, stored alongside the string entity list
        entity_list.prompt_entity_list = uploaded_settings.get('prompt_entity_list', entity_list.prompt_entity_list)
    else:
        print("WARNING: unexpected entity list format!\n\n" + str(entity_list))
    
    # check entity list
    print("Entity list:\n" + entity_list.model_dump_json(indent=2))

    # create new memory object
    new_obj = HierarchicalSummaryMemory(
        summary_llm=llm,
        chat_thread=ct,
        entity_manager=entity_list,
        summary_prompt=uploaded_settings.get('summarization_prompt'),
        prop_ctx = uploaded_settings["prop_ctx"],
        prop_summary = uploaded_settings["prop_summary"],
        n_levels = uploaded_settings["n_levels"],
        n_tok_summarize = uploaded_settings["n_tok_summarize"],
        all_memory = uploaded_settings["all_memory"],
        archived_memory = uploaded_settings["archived_memory"]
    )
    # return object
    return new_obj

def convert_chat(input_data: Dict[str, Any]) -> str:
    # if this is an old non-hierarchical format, try to recover it
    # TODO: convert regular managers into hierarchical ones with no summaries?
    if input_data.get('chat_thread') is None:
        return StatefulChatManager._recover_old_json_format(uploaded_settings)
    # initialize LLM
    # TODO: use some dynamic loading to handle other classes
    llm = llm_from_json(input_data.get('llm'))
    print("Main LLM: " + str(llm))
    # load chat memory, which has required parameters for manager construction
    new_chat_memory = hier_mem_from_json(input_data.get('chat_memory'))
    # create new manager object
    new_obj = HierarchicalSummaryManager(
        llm=llm,
        chat_memory = new_chat_memory
        )

    # convert to JSON
    output_json = new_obj.model_dump_json(indent=2)
    print("JSON from upgraded object:\n\n")
    print(new_obj.chat_memory.entity_manager.model_dump_json(indent=2))
    # test loading from JSON
    print("Testing output JSON...")
    test_obj = HierarchicalSummaryManager.model_validate_json(output_json)
    print("Original entity manager class: " + str(type(new_obj.chat_memory.entity_manager)))
    print("Reconstituted entity manager class: " + str(type(test_obj.chat_memory.entity_manager)))
    print(str(test_obj.chat_memory.entity_manager.entity_list))
    # print(test_obj.chat_memory.entity_manager.model_dump_json(indent=2))
    # return object
    return output_json

def parse_args():
    parser = argparse.ArgumentParser(description="Reads older chat data formats and upgrades them to the latest format version.")
    parser.add_argument("input", type=Path, help="Path to the input JSON file")
    parser.add_argument("output", type=Path, help="Path to the output JSON file")
    return parser.parse_args()

def main():
    args = parse_args()

    # Check input exists
    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    try:
        with args.input.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: failed to parse JSON from {args.input}: {e}", file=sys.stderr)
        sys.exit(3)
    except OSError as e:
        print(f"Error: could not read {args.input}: {e}", file=sys.stderr)
        sys.exit(4)

    # convert the data to latest format
    upgraded_data = convert_chat(input_data=data)

    try:
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            f.write(upgraded_data)
            f.write("\n")
    except OSError as e:
        print(f"Error: could not write to {args.output}: {e}", file=sys.stderr)
        sys.exit(5)

    print(f"Upgraded JSON from {args.input} saved to {args.output}")

if __name__ == "__main__":
    main()