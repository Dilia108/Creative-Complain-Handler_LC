# This file tests the agent with a few sample complaints, demonstrating how it uses the tools and few-shot examples to generate creative, story-driven responses.
import os
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# --- Tools (unchanged) ---

@tool
def consult_demogorgon(complaint: str) -> str:
    """Get the Demogorgon's perspective on a complaint about the Upside Down."""
    responses = [
        f"The Demogorgon tilts its head at '{complaint}'. Perhaps you're thinking in three dimensions?",
        f"The Demogorgon suggests the problem might be temporal - things work differently in the Upside Down.",
        f"The Demogorgon doesn't understand '{complaint}' - consistency isn't a priority there.",
    ]
    return random.choice(responses)

@tool
def check_hawkins_records(query: str) -> str:
    """Search Hawkins historical records for information."""
    records = {
        "portal": "Records show portals have opened on various dates with no clear pattern.",
        "monsters": "Creatures from the Upside Down behave differently based on environmental factors.",
        "psychics": "Psychic abilities vary greatly between individuals.",
        "electricity": "Hawkins has a history of electrical anomalies linked to electromagnetic fields.",
    }
    for key, value in records.items():
        if key in query.lower():
            return value
    return f"No specific records for '{query}', but many unexplained events have occurred in Hawkins."

@tool
def cast_interdimensional_spell(problem: str, creativity_level: str = "medium") -> str:
    """Suggest a creative interdimensional spell to fix a problem."""
    creativity_multiplier = {"low": 1, "medium": 2, "high": 3}[creativity_level]
    spells = [
        f"Chant 'Bemca Becma Becma' three times while holding a Walkman to recalibrate frequencies for: {problem}",
        f"Create a salt circle with a compass in the center to stabilize: {problem}",
        f"Play 'Running Up That Hill' backwards at the exact location to fix: {problem}",
        f"Arrange a lighter, compass, and something personal in a triangle while focusing on: {problem}",
    ]
    return "\n".join(random.sample(spells, min(creativity_multiplier, len(spells))))

@tool
def gather_party_wisdom(question: str) -> str:
    """Ask the D&D party (Mike, Dustin, Lucas, Will) for their collective wisdom."""
    party_responses = {
        "portal": "Mike: 'Portals open near strong emotional events.' Dustin: 'They follow the Mind Flayer's activity.'",
        "monsters": "Lucas: 'Demogorgons are territorial.' Will: 'They sense fear and strong emotions.'",
        "psychics": "Mike: 'El's powers connect to her emotions.' Dustin: 'Limited by mental energy.'",
        "electricity": "Lucas: 'The Upside Down disrupts electrical systems.' Dustin: 'It's like a feedback loop.'",
    }
    for key, response in party_responses.items():
        if key in question.lower():
            return response
    return "The party huddles. Mike: 'Tough one.' Dustin: 'Need more info.' Lucas: 'Think it through.' Will: 'Consult other sources?'"


tools = [
    consult_demogorgon,
    check_hawkins_records,
    cast_interdimensional_spell,
    gather_party_wisdom,
]


# --- Few-shot examples ---
# Each example shows the tone: dramatic framing, tool reasoning made explicit,
# and a punchy, story-driven final answer.

examples = [
    {
        "input": "My compass keeps spinning whenever I walk past the old Hawkins quarry.",
        "output": (
            "Fascinating — a spinning compass is rarely just a hardware fault in Hawkins. "
            "I cross-referenced the quarry's coordinates against the Hawkins historical records "
            "and found a cluster of electromagnetic anomalies dating back to the lab's early experiments. "
            "The Upside Down leaks magnetic interference the same way a broken pipe leaks water: "
            "steadily, invisibly, until something forces it into the open. "
            "My recommendation: plant a salt circle ten feet from the quarry's edge and approach only "
            "with a personal object — something with emotional weight acts as a stabiliser. "
            "Do NOT go alone after dark. The anomaly is a beacon, and beacons attract attention from both sides."
        ),
    },
    {
        "input": "Will keeps drawing the same spiral shape in his sleep and won't remember it in the morning.",
        "output": (
            "This is urgent — involuntary drawings during sleep are a classic sign of psychic bleed-through. "
            "I consulted the party immediately: Will's episodes have historically been tied to "
            "the Mind Flayer's attempt to re-establish a sensory link. The spiral almost certainly "
            "maps to a location — possibly one that's shifting between dimensions right now. "
            "Photograph every drawing and overlay them: if they're identical, it's a message; "
            "if they drift slightly each night, it's a moving target. "
            "Either way, keep Will away from basements and anywhere with standing water until we know more. "
            "I've already begun preparing a frequency-disruption ritual — we may need to act fast."
        ),
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


# --- Full prompt ---

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a creative interdimensional investigator solving mysteries related to "
        "the Upside Down in Hawkins, Indiana. Approach every problem with curiosity and "
        "imagination. Combine tools in unexpected ways and suggest unconventional solutions. "
        "The Upside Down doesn't follow normal rules — so neither should your problem-solving! "
        "Always explain your reasoning and make the investigation feel exciting and urgent."
    ),
    few_shot_prompt,                                   # injected between system and human turn
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# --- Agent ---

agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

print("Agent created with few-shot prompt")
print(f"Tools available: {[t.name for t in tools]}")


# --- Testing ---
 
complaints = [
    "Why do demogorgons sometimes eat people and sometimes don't?",
    "The portal opens on different days—is there a schedule?",
    "Why can some psychics see the Upside Down and others can't?",
    "Why do creatures and power lines react so strangely together?",
]
 
 
def handle_complaint(complaint: str) -> str:
    """Handle a single complaint through the agent."""
    print(f"\n{'='*60}")
    print(f"COMPLAINT: {complaint}")
    print(f"{'='*60}\n")
    result = agent_executor.invoke({"input": complaint})
    return result["output"]
 
 
if __name__ == "__main__":
    print("Testing agent with sample complaints...\n")
    for complaint in complaints[:2]:  # Test first 2 — expand slice to test more
        response = handle_complaint(complaint)
        print(f"\nRESPONSE: {response}\n")