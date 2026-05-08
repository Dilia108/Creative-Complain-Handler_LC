# Lab summary

This lab explored how prompt design shapes agent behaviour in a LangChain tool-use setting. 

Starting from a basic `AgentExecutor` wired to four thematic tools, the work progressed through three focused iterations. 
* First, the system prompt was restructured into a few-shot format using `FewShotChatMessagePromptTemplate`, with two example exchanges chosen specifically to model a vivid, urgent investigation tone rather than generic helpfulness. 
* Second, those examples were evaluated against the stated objective — encouraging creative problem-solving — and found to demonstrate style without demonstrating reasoning; the examples were revised so the agent's fictional chain of thought explicitly showed tool results changing the direction of the next tool call, modelling the reframing behaviour that distinguishes creative from mechanical problem-solving. 
* Third, a `ToolUsageTracker` was added as a `BaseCallbackHandler` subclass and wired into the executor via `callbacks`, capturing per-complaint tool sequences automatically without modifying any tool code; this made it possible to measure chaining rate across all four sample complaints and directly verify whether the prompt changes produced observable multi-tool reasoning in practice.


**From the Analysis obtained, like to highlight:**

* Tool usage counts: -> Shows that the agent seems to be overrelying in one tool ('consult_demogorgon')

* Chaining rate -> show that the creative prompt is producing multi-tool reasoning
