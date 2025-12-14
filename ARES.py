import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import OpenAI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = OpenAI(model="gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOllama(model="llama3.2:1b")



# CREATE ANALYSTS: human in the loop
from typing import List
from typing import TypedDict
from pydantic import BaseModel, Field




class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    


@property
def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions


from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
                        1. First, review the research topic: {topic}
                        2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: {human_analyst_feedback}
                        3. Determine the most interesting themes based upon documents and / or feedback above.
                        4. Pick the top {max_analysts} themes.
                        5. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback, 
        max_analysts=max_analysts)

    # Generate question 
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    
    # Write the list of analysis to state
    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    
    # Otherwise end
    return END

# Add nodes and edges 
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# Input
max_analysts = 3 
topic = "The benefits of adopting LangGraph as an agent framework"
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream({"topic":topic,"max_analysts":max_analysts,}, thread, stream_mode="values"):
    # Review
    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)  

'''SO NOW 3 ANALYST ARE GENERATED AND NOW IN NEXT STEP THERE WILL NE THE HUMAN FEEDBACK'''

while True:
    user_feedback = input("\nEnter feedback for analysts (press Enter to finish): ")

    # Update the state at the human_feedback node
    graph.update_state(
        thread,
        {"human_analyst_feedback": user_feedback if user_feedback else None},
        as_node="human_feedback"
    )

    if not user_feedback:
        break

    print("\n=======================   Regenerating analysts...   =======================\n")

    # Continue graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)

final_state = graph.get_state(thread)
analysts = final_state.values.get('analysts')

print("\n\n============================== 🤖🤖  FINAL ANALYSTS  🤖🤖 ==============================")

for analyst in analysts:
    print("\n")
    print(f"Name: {analyst.name}")
    print(f"Affiliation: {analyst.affiliation}")
    print(f"Role: {analyst.role}")
    print(f"Description: {analyst.description}")
    print("-" * 50)














# -----------------------------  PHASE 2: INTERVIEW SYSTEM  -----------------------------

import operator
from typing import Annotated, List
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
from pydantic import BaseModel, Field


# --------------------------- FIXED ANALYST CLASS WITH PROPERTY ---------------------------

class Analyst(BaseModel):
    affiliation: str
    name: str
    role: str
    description: str

    @property
    def persona(self):
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Affiliation: {self.affiliation}\n"
            f"Description: {self.description}\n"
        )


# --------------------------- INTERVIEW STATE ---------------------------

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query to run on Wikipedia")


# --------------------------- QUESTION GENERATION ---------------------------

question_instructions = """You are an analyst conducting an interview.

Stay in character as described here:
{goals}

Ask insightful, specific questions about your topic.

When you feel the interview is complete, end by saying:
"Thank you so much for your help!"
"""

def generate_question(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]

    system_msg = SystemMessage(content=question_instructions.format(goals=analyst.persona))
    response = llm.invoke([system_msg] + messages)

    return {"messages": [response]}


# --------------------------- WIKIPEDIA SEARCH ---------------------------

search_system_msg = SystemMessage(content="""
You will receive the conversation between an analyst and expert.
Rewrite ONLY the final question as a clean, stand-alone search query.
""")

def search_wikipedia(state: InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery)

    sq = structured_llm.invoke([search_system_msg] + state["messages"])

    try:
        docs = WikipediaLoader(query=sq.search_query, load_max_docs=2).load()
    except:
        docs = []

    formatted = []
    for d in docs:
        formatted.append(
            f'<Document source="{d.metadata.get("source","wiki")}" page="{d.metadata.get("page","")}">\n'
            f"{d.page_content}\n</Document>"
        )

    return {"context": ["\n\n---\n\n".join(formatted)]}


# --------------------------- ANSWER GENERATION ---------------------------

answer_instructions = """
You are an expert being interviewed.  
Use ONLY the following context to answer:

{context}

Follow these rules:
1. No external facts.
2. Cite sources like [1], [2], etc.
3. Keep answers focused and specific.
"""

def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    context = state["context"]
    messages = state["messages"]

    sys_msg = SystemMessage(content=answer_instructions.format(
        goals=analyst.persona,
        context=context
    ))

    answer = llm.invoke([sys_msg] + messages)
    answer.name = "expert"

    return {"messages": [answer]}


# --------------------------- ROUTER ---------------------------

def route_messages(state: InterviewState, name="expert"):

    messages = state["messages"]
    max_turns = state["max_num_turns"]

    # How many answers given?
    num_answers = len([m for m in messages if isinstance(m, AIMessage) and m.name == "expert"])

    if num_answers >= max_turns:
        return "save_interview"

    last_question = messages[-2].content if len(messages) >= 2 else ""

    if "Thank you so much for your help" in last_question:
        return "save_interview"

    return "ask_question"


# --------------------------- INTERVIEW SAVER ---------------------------

def save_interview(state: InterviewState):
    log = get_buffer_string(state["messages"])
    return {"interview": log}


# --------------------------- SECTION WRITER ---------------------------

section_writer_instructions = """
You are a technical writer.

Create a clean markdown section based on the interview data.

Use this structure:

## Title
### Summary
### Sources

Make the summary ~300 words and reference sources as [1], [2], etc.
"""

def write_section(state: InterviewState):
    analyst = state["analyst"]
    context = state["context"]

    sys_msg = SystemMessage(content=section_writer_instructions)
    human_msg = HumanMessage(content=f"Use this source to write the section:\n{context}")

    result = llm.invoke([sys_msg, human_msg])

    return {"sections": [result.content]}


# --------------------------- BUILD INTERVIEW GRAPH ---------------------------

interview_builder = StateGraph(InterviewState)

interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge("save_interview", "write_section")

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ["ask_question", "save_interview"])
interview_builder.add_edge("write_section", END)

memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory)



# -----------------------------  PHASE 3: MASTER RESEARCH GRAPH  -----------------------------

from langgraph.types import Send

# 1. Define the Master State
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add] 
    introduction: str
    content: str
    conclusion: str
    final_report: str

# 2. Parallel Execution Node (The "Map" Step)
def initiate_all_interviews(state: ResearchGraphState):
    """ This runs all interviews in parallel using the Send API """    
    
    # Check if we need to go back for feedback
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise, kick off the sub-graph for every analyst
    topic = state["topic"]
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            # We explicitly cast the message to HumanMessage for Ollama stability
            "messages": [HumanMessage(content=f"Start the interview regarding: {topic}")],
            "max_num_turns": 3,
            "context": [],
            "sections": []
        }) for analyst in state["analysts"]
    ]

# 3. Report Writing Nodes (The "Reduce" Step)
report_writer_instructions = """You are a technical writer.
Topic: {topic}
Based on the analyst memos provided, write a comprehensive section.
Do not use "Analyst 1 said..." just synthesize the facts.
Memos: {context}"""

def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Write the main body of the report.")]) 
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    topic = state["topic"]
    instructions = f"Write a catchy 100-word introduction for a report on: {topic}"
    intro = llm.invoke([SystemMessage(content=instructions)]+[HumanMessage(content="Write the introduction.")]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    topic = state["topic"]
    instructions = f"Write a strong 100-word conclusion for a report on: {topic}"
    conclusion = llm.invoke([SystemMessage(content=instructions)]+[HumanMessage(content="Write the conclusion.")]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ Stitch it all together """
    final_report = (
        "# " + state["topic"] + "\n\n" +
        "## Introduction\n" + state["introduction"] + "\n\n" +
        "## Insights\n" + state["content"] + "\n\n" +
        "## Conclusion\n" + state["conclusion"]
    )
    return {"final_report": final_report}

# 4. Build the Master Graph
builder = StateGraph(ResearchGraphState)

# Add Nodes
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_graph) # We use the graph from Phase 2
builder.add_node("write_report", write_report)
builder.add_node("write_introduction", write_introduction)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("finalize_report", finalize_report)

# Add Edges
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")

# Conditional Edge: Either loop back for feedback OR fan-out to interviews
builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])

# Fan-in: Wait for all interviews to finish, then write the 3 parts
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")

# Finalize: Combine the 3 parts
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
memory = MemorySaver()
master_graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# -----------------------------  PHASE 4: EXECUTION  -----------------------------

print("\n🚀 LAUNCHING AUTONOMOUS RESEARCH AGENT...\n")

max_analysts = 3 
topic = "The benefits of adopting LangGraph as an agent framework"
thread = {"configurable": {"thread_id": "1"}}

# Step 1: Generate Analysts
for event in master_graph.stream({"topic":topic, "max_analysts":max_analysts}, thread, stream_mode="values"):
    analysts = event.get('analysts', '')
    if analysts:
        print(f"--> Generated {len(analysts)} analysts.")

# Step 2: Human Feedback (Simulated for automation, or use input())
# You can uncomment the input line below if you want to actually type feedback
# user_feedback = input("\nAny feedback for analysts? (Press Enter to skip): ")
user_feedback = None 

master_graph.update_state(thread, {"human_analyst_feedback": user_feedback}, as_node="human_feedback")

# Step 3: Run Interviews & Write Report
print("\n--> Running Interviews & Writing Report (This may take a moment)...")

final_output = ""

# stream_mode="updates" allows us to see progress node by node
for event in master_graph.stream(None, thread, stream_mode="updates"):
    for node, value in event.items():
        print(f"✅ Finished Node: {node}")
        if node == "finalize_report":
            final_output = value["final_report"]

# -----------------------------  FINAL OUTPUT  -----------------------------

print("\n\n============================== 📝 FINAL COMPILED REPORT 📝 ==============================\n")
print(final_output)

if final_output:
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(final_output)
    print("\n[System] Full report saved to 'research_report.md'")