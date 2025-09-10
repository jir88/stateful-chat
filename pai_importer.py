import stateful_chat
import stateful_chat.manager as scm
import stateful_chat.llm as scl

session_file = open("", mode='r', encoding="utf-8")

all_lines = session_file.readlines()

system_prompt = ""
all_messages = []
all_memories = []
msg_idx = 0
in_ai_msg = False

for line in all_lines:
    line = line.strip()
    # determine line type
    line_type = "assistant"
    if len(line) == 0:
        line_type = "blank"
    elif line[0] == ">":
        line_type = "user"
    elif line.startswith("SUMMARY"):
        line_type = "summary"
    elif line.startswith("SYSTEM"):
        line_type = "system"
    
    # if in AI message and line is message
    if in_ai_msg and line_type == "assistant":
        # append to previous message, separated with two newlines
        all_messages[-1]['content'] += "\n\n" + line
    # else if is AI message
    elif not in_ai_msg and line_type == "assistant":
        # add new AI message
        all_messages.append({'role':'assistant', 'content':line})
        # set in message to true
        in_ai_msg = True
        # increment message index
        msg_idx += 1
    # if this is a user message
    elif line_type == "user":
        in_ai_msg = False
        all_messages.append({'role':'user', 'content':line[2:]})
        msg_idx += 1
    # else if is summary
    elif line_type == "summary":
        # in AI message is false
        in_ai_msg = False
        # parse summary level "SUMMARY^1: The"
        level = int(line[8])
        # remove prefix
        line = line[11:]
        # add new summary message
        all_memories.append({
            'level': level,
            'msg_idx': msg_idx,
            'content': line
        })
    # else if is system prompt
    elif line_type == "system":
        # in AI message is false
        in_ai_msg = False
        # remove prefix
        line = line[8:]
        # set system prompt
        system_prompt = line

# 
# n_arch_msgs = len(orig_chat_session.chat_thread.archived_messages)
# n_msgs = len(orig_chat_session.chat_thread.messages)
# print(f"Session has {n_arch_msgs} archived messages and {n_msgs} active messages.")

# set up LLM backend
inst_fmt = scl.InstructFormat.from_json(open("./instruct_formats/gemma_chat.json", mode='r'))
sampling_options = {
    "num_predict": 1024,
    "num_ctx": 16384,
    "temperature": 1.0,
    "min_p": 0.05,
    "top_k": 0,
    "top_p": 1.0,
    "keep_alive": "15m"
}
llm = scl.OpenAILLM(model="gemma-3n-E4B-it-UD-Q5_K_XL-cpu", instruct_fmt=inst_fmt, sampling_options=sampling_options)

# construct a heirarchical chat and memory
h_manager = scm.HierarchicalSummaryManager(
    llm=llm,
    summary_llm=llm,
    prop_ctx=0.8,
    prop_summary=0.5,
    n_levels=3,
    n_tok_summarize=768
)
h_memory = h_manager.chat_memory

# we'll build our own memories instead of using the PAI ones
n_memories = 0
for msg in all_messages:
    print("Adding message:")
    print(msg)
    h_manager.append_message(message=msg)
    h_memory.update_all_memory()
    print(f"Memory has {len(h_memory.all_memory)} active memories and {len(h_memory.archived_memory)} archived memories.")
    if n_memories != (len(h_memory.all_memory) + len(h_memory.archived_memory)):
        mems = [f"Level {m['level']}: {m['content']}" for m in h_memory.archived_memory]
        print("\nArchived memories:\n\n" + "\n".join(mems))
        mems = [f"Level {m['level']}: {m['content']}" for m in h_memory.all_memory]
        print("\nActive memories:\n\n" + "\n".join(mems))
        n_memories = (len(h_memory.all_memory) + len(h_memory.archived_memory))

print("Saving results...")
with open("", mode='w') as f:
    f.write(h_manager.to_json())
print("Done!")