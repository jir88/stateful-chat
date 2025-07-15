import stateful_chat
import stateful_chat.manager as scm
import stateful_chat.llm as scl

session_file = open("C:\\Users\\John Robinson\\Downloads\\logs\\pv-37-b.json", mode='r')
orig_chat_session = scm.StatefulChatManager.from_json(session_file)

n_arch_msgs = len(orig_chat_session.chat_thread.archived_messages)
n_msgs = len(orig_chat_session.chat_thread.messages)
print(f"Session has {n_arch_msgs} archived messages and {n_msgs} active messages.")

# set up LLM backend
inst_fmt = scl.InstructFormat.from_json(open("./instruct_formats/gemma_chat.json", mode='r'))
sampling_options = {
    "num_predict": 512,
    "num_ctx": 10240,
    "temperature": 1.0,
    "min_p": 0.1,
    "keep_alive": "15m"
}
llm = scl.OpenAILLM(model="gemma-3n-E4B-it-UD-Q4_K_XL-cpu", instruct_fmt=inst_fmt, sampling_options=sampling_options)

# construct a heirarchical chat and memory
h_chat = scm.ChatThread(session_id="1")
h_memory = scm.HierarchicalSummaryMemory(
    summary_llm=llm, 
    chat_thread=h_chat,
    prop_ctx=0.8,
    prop_summary=0.5,
    n_levels=3,
    n_tok_summarize=512
)

# combine archived and active messages for this test
all_messages = []
all_messages.extend(orig_chat_session.chat_thread.archived_messages)
all_messages.extend(orig_chat_session.chat_thread.messages)

n_memories = 0
for msg in all_messages:
    print("Adding message:")
    print(msg)
    h_chat.messages.append(msg)
    h_memory.update_all_memory()
    print(f"Memory has {len(h_memory.all_memory)} active memories and {len(h_memory.archived_memory)} archived memories.")
    if n_memories != (len(h_memory.all_memory) + len(h_memory.archived_memory)):
        mems = [f"Level {m['level']}: {m['content']}" for m in h_memory.archived_memory]
        print("\nArchived memories:\n\n" + "\n".join(mems))
        mems = [f"Level {m['level']}: {m['content']}" for m in h_memory.all_memory]
        print("\nActive memories:\n\n" + "\n".join(mems))
        n_memories = (len(h_memory.all_memory) + len(h_memory.archived_memory))

print("Saving results...")
with open("C:\\Users\\John Robinson\\Downloads\\logs\\pv-37-b-h.json", mode='w') as f:
    f.write(h_memory.to_json())
print("Done!")