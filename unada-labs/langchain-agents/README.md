# AI agents with LangChain

## What it was
Alongside the RAG chatbots, I built AI agents with LangChain. The difference is that a chatbot answers, while an agent can work through a task: decide what to do next, use a tool, look at the result and keep going until it's done.

## How the agents worked
1. The agent gets a task.
2. The LLM reasons about the next step (the ReAct pattern: reason, then act).
3. It calls one of the tools we gave it, like a document retriever, a lookup against data, or an API.
4. It reads what the tool returned and decides whether it needs another step.
5. When it has enough, it gives the final answer.

## Things I learned the hard way
- **Tool descriptions matter more than the prompt.** The LLM picks tools based on their descriptions, so a vague description means the wrong tool gets called.
- **Always cap the number of steps.** Without a limit an agent can loop, calling the same tool again and again.
- **Log every step.** When an agent gives a wrong answer you need to see which step went wrong, not just the final output.

## LangChain vs LangGraph
LangChain gives you the building blocks: model wrappers, prompts, tools and retrievers. LangGraph sits on top and lets you define the agent as a graph with state and loops, which gives you more control once flows get complicated. I've used LangGraph in my [ML On-Call Agent](https://github.com/kanishqtanwar35-hub/ml-oncall-agent) project.

The agent code was written against internal tools and APIs, so it isn't included here.
