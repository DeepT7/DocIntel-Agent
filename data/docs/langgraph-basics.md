# LangGraph basics

LangGraph models agent workflows as a graph of nodes and edges. A node is a
Python function that receives the state and returns an update.

# State

The state is a TypedDict. Each node returns a partial dict that is merged into
the state.
