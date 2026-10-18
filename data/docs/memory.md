# Memory

Short-term memory is thread-scoped and provided by a checkpointer. Long-term
memory is cross-thread and provided by a store.

# Checkpointers

MemorySaver does not persist between restarts; use PostgresSaver in production.
