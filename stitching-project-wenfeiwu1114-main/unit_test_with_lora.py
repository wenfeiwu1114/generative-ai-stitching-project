import time
import os
import torch
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from typing import List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from pprint import pprint
from peft import PeftModel, get_peft_model, LoraConfig


# Load API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

# --- Initialize Base LLM ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# LoRA Configuration
peft_config = LoraConfig(
    r=16, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM"
)

# Load tokenizer (replace with the correct model you're fine-tuning)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Ensure proper precision based on device
dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # Use float32 on CPU
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device).to(dtype)

# --- Load LoRA Fine-Tuned Model ---
# Load environment variables
load_dotenv()
# Load LoRA fine-tuned model
model_name = "./tuned-lora-model"
if not os.path.exists("./tuned-lora-model"):
    print("Error: Fine-tuned LoRA model not found. Ensure training is completed before loading.")
else:
    lora_model = get_peft_model(base_model, peft_config)

lora_model.to(device).to(dtype)

# --- Define Data Models for Grading ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# --- Define Retrieval (Mock for Testing) ---
def mock_retrieve(question):
    responses = {
        "What news about Puerto Rico?": [
            Document(page_content="Puerto Rico was under a hurricane watch."),
            Document(page_content="Puerto Rico's economy is growing."),
        ],
        "What happened during COVID-19?": [
            Document(page_content="COVID-19 caused major lockdowns."),
            Document(page_content="Vaccine rollouts were a major focus."),
        ],
    }
    return responses.get(question, [])

def retrieve(state):
    print("--- RETRIEVING DOCUMENTS ---")
    question = state["question"]
    documents = mock_retrieve(question)
    return {"documents": documents, "question": question}

# --- Define Document Grader ---
def grade_documents(state):
    print("\n--- CHECKING DOCUMENT RELEVANCE ---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = [
        doc for doc in documents if "hurricane" in doc.page_content or "COVID-19" in doc.page_content
    ]

    print(f"--- {len(filtered_docs)} documents deemed relevant ---")
    return {"documents": filtered_docs, "question": question}

# --- LoRA-Enhanced Generation ---
def lora_generate(state):
    print("\n--- GENERATING RESPONSE WITH LORA ---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": documents, "question": question, "generation": "No relevant documents found."}

    context_text = "\n".join([doc.page_content for doc in documents])
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer: "

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = lora_model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"documents": documents, "question": question, "generation": response}

# --- Define Hallucination Grader ---
def hallucination_grader(state):
    print("\n--- CHECKING HALLUCINATIONS ---")
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    hallucination_score = "yes" if any(doc.page_content in generation for doc in documents) else "no"
    print(f"--- HALLUCINATION SCORE: {hallucination_score} ---")

    state["hallucination_score"] = hallucination_score
    return state

# --- Define Answer Grader ---
def answer_grader(state):
    print("\n--- CHECKING ANSWER RELEVANCE ---")
    question = state.get("question", "")
    generation = state.get("generation", "")

    answer_score = "yes" if question in generation else "no"
    print(f"--- ANSWER RELEVANCE SCORE: {answer_score} ---")

    state["answer_score"] = answer_score
    return state

# --- Define Query Rewriting ---
def transform_query(state):
    print("\n--- REWRITING QUERY ---")
    question = state["question"]
    rewritten_question = f"Updated query: {question}"

    return {"documents": state["documents"], "question": rewritten_question}

# --- Decision Logic for Graph ---
def decide_to_generate(state):
    print(f"\n--- DECISION CHECK: Retrieved {len(state['documents'])} documents ---")
    
    if not state["documents"]:
        print("--- DECISION: NO DOCUMENTS FOUND → Transforming Query ---")
        return "not_relevant"
    
    print("--- DECISION: RELEVANT DOCUMENTS FOUND → Proceeding to Generate ---")
    return "relevant"

def grade_generation_v_documents_and_question(state):
    print(f"\n--- CHECKING FINAL RESPONSE ---\n{state.get('generation', 'No response generated.')}\n")

    retry_count = state.get("retry_count", 0)
    hallucination_score = state.get("hallucination_score", "no")
    answer_score = state.get("answer_score", "no")

    print(f"--- HALLUCINATION SCORE: {hallucination_score} ---")
    print(f"--- ANSWER RELEVANCE SCORE: {answer_score} ---")

    if hallucination_score == "yes" and answer_score == "yes":
        print("--- DECISION: RESPONSE IS USEFUL → ENDING WORKFLOW ---")
        return "useful"

    if retry_count >= 2:
        print("--- MAX RETRIES REACHED: ACCEPTING RESPONSE AS FINAL ---")
        return "useful"

    print("--- DECISION: RESPONSE NOT GROUNDED → Retrying ---")
    state["retry_count"] = retry_count + 1
    return "not supported"

# --- Define Graph State ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# --- Build the Multi-Agent Workflow ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_with_lora", lora_generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("hallucination_grader", hallucination_grader)
workflow.add_node("answer_grader", answer_grader)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"not_relevant": "transform_query", "relevant": "generate_with_lora"},
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate_with_lora", "hallucination_grader")
workflow.add_edge("hallucination_grader", "answer_grader")
workflow.add_conditional_edges(
    "answer_grader",
    grade_generation_v_documents_and_question,
    {"not supported": "generate_with_lora", "useful": END},
)

# Compile workflow
app_with_lora = workflow.compile()

# --- Run the Unit Test ---
if __name__ == "__main__":
    test_questions = ["What news about Puerto Rico?", "What happened during COVID-19?"]

    for question in test_questions:
        print(f"\n--- Testing: {question} ---")
        result = app_with_lora.invoke({"question": question}, config={"recursion_limit": 10})
        print(f"Final Response: {result.get('generation', 'No answer generated.')}")
        pprint(result)

    print("\nUnit test completed successfully.")
