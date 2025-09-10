import stateful_chat
import stateful_chat.manager as scm
import stateful_chat.llm as scl

session_file = open("", mode='r')
orig_chat_session = scm.StatefulChatManager.from_json(session_file)
# orig_chat_session = scm.HierarchicalSummaryManager.from_json(session_file)

n_arch_msgs = len(orig_chat_session.chat_thread.archived_messages)
n_msgs = len(orig_chat_session.chat_thread.messages)
print(f"Session has {n_arch_msgs} archived messages and {n_msgs} active messages.")

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
n_tok = llm.count_tokens("The quick brown fox jumps over the lazy dog.")
print("Tokens: " + str(n_tok))

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

# combine archived and active messages for this test
all_messages = []
all_messages.extend(orig_chat_session.chat_thread.archived_messages)
all_messages.extend(orig_chat_session.chat_thread.messages)

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