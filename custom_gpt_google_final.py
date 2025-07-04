import json
import requests
from typing import List
from newspaper import Article
from pydantic import BaseModel, ValidationError, RootModel
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from lagent.llms import GPTAPI
import os
import pycountry
import tiktoken
# ----- Load API Keys -----
load_dotenv()
GOOGLE_API_KEY = os.getenv("WEB_SEARCH_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----- Token Cost Tracking -----
class TokenCostTracker:
    def __init__(self):
        # GPT-4-turbo pricing (as of 2024)
        self.input_price_per_1k = 0.01  # $0.01 per 1K input tokens
        self.output_price_per_1k = 0.03  # $0.03 per 1K output tokens
        
        # Initialize tokenizer for GPT-4
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for input and output tokens"""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost
    
    def add_call(self, input_tokens: int, output_tokens: int):
        """Add a new API call to the tracker"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        call_cost = self.calculate_cost(input_tokens, output_tokens)
        self.total_cost += call_cost
        self.call_count += 1
        
        print(f"[TokenCost] Call #{self.call_count}:")
        print(f"  Input tokens: {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Call cost: ${call_cost:.4f}")
        print(f"  Running total: ${self.total_cost:.4f}")
    
    def get_summary(self) -> dict:
        """Get cost summary"""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_call": self.total_cost / self.call_count if self.call_count > 0 else 0
        }
    
    def print_final_summary(self):
        """Print final cost summary"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("FINAL TOKEN COST SUMMARY")
        print("="*50)
        print(f"Total API calls: {summary['total_calls']}")
        print(f"Total input tokens: {summary['total_input_tokens']:,}")
        print(f"Total output tokens: {summary['total_output_tokens']:,}")
        print(f"Total cost: ${summary['total_cost']:.4f}")
        print(f"Average cost per call: ${summary['avg_cost_per_call']:.4f}")
        print("="*50)

# Initialize global token cost tracker
token_tracker = TokenCostTracker()

def is_country(name: str) -> bool:
    try:
        pycountry.countries.search_fuzzy(name)
        return True
    except LookupError:
        return False


# ----- JSON Schema -----
class Flashpoint(BaseModel):
    title: str
    description: str
    entities: List[str]

class FlashpointList(RootModel[List[Flashpoint]]):
    pass

# ----- Google Search + Extraction -----
class GoogleSearchAgent:
    def __init__(self, api_key, cx):
        self.api_key = api_key
        self.cx = cx

    def search_news(self, query, num_results=10):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "dateRestrict": "d1",      # Last 1 day
            "sort": "date",
            "num": num_results
        }
        res = requests.get(url, params=params).json()
        return [item["link"] for item in res.get("items", []) if "link" in item]

    def extract_article(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return ""

    def gather_context(self, query: str) -> str:
        urls = self.search_news(query)
        contents = [self.extract_article(u) for u in urls]
        return "\n\n".join(c for c in contents if c.strip())

# ----- Entity Tracker -----
class EntityTracker:
    def __init__(self):
        self.seen_entities = set()
        self.search_run = 0
        self.llm_run = 0

    def is_new_combo(self, entities: List[str]) -> bool:
        return not any(e.lower() in self.seen_entities for e in entities)

    def add(self, entities: List[str]):
        for e in entities:
            self.seen_entities.add(e.lower())
            
    def update_seen_entities(self, entities: List[str]):
        #update entity if it is not in the list
        for e in entities:
            if e.lower() not in self.seen_entities:
                self.seen_entities.add(e.lower())

    def get_exclude_query(self):
        return " ".join(f'-"{e}"' for e in self.seen_entities)

# ----- Prompts -----
SYSTEM_PROMPT = (
    "You are a global strategic signal analyst.\n"
    "Your mission is to detect the top 10 most active or unstable global regions in the last 24 hours, "
    "based on multi-domain tension signals.\n\n"
    "Include ALL of the following domains:\n"
    "- Geopolitical, Military, Economic, Cultural, Religious, Tech, Cybersecurity, Environmental, Demographics, Sovereignty.\n\n"
    "Each output must include:\n"
    "- title: short phrase (e.g., 'US–China Chip War')\n"
    "- description: one factual sentence (≤200 chars)\n"
    "- entities: list of involved countries, organizations, regions, or non-state actors\n\n"
    "Output: JSON list of 10 dictionaries.\n"
    "NO extra text, bullets, or explanation. JUST clean JSON.\n"
    "Example: [{\"title\": \"X\", \"description\": \"Y\", \"entities\": [\"Israel\", \"Iran\"]}]"
)

USER_PROMPT = (
    "List 10 top global flashpoints from the past 24 hours with title, description, and involved entities "
    "(countries, regions, organizations, or non-state actors).\n\n"
    "Return only valid JSON."
)

# ----- LLM Reasoning Agent -----
class FlashpointLLMAgent:
    def __init__(self, key):
        self.llm = GPTAPI(model_type="gpt-4-turbo", key=key, temperature=0)

    def call_llm(self, context: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + f"\n\nContext:\n{context[:15000]}"}
        ]
        
        # Count input tokens
        input_text = "\n".join([msg["content"] for msg in messages])
        input_tokens = token_tracker.count_tokens(input_text)
        
        print(f"[LLMAgent] Making API call with {input_tokens:,} input tokens...")
        
        result = self.llm.chat(messages)
        result = result.strip()
        
        # Count output tokens
        output_tokens = token_tracker.count_tokens(result)
        
        # Track the cost
        token_tracker.add_call(input_tokens, output_tokens)
        
        return result

    def validate_json(self, response: str):
        try:
            data = json.loads(response)
            return FlashpointList.model_validate(data).root
        except (json.JSONDecodeError, ValidationError) as e:
            print("Validation Error:", e)
            return None

# ----- LangGraph Agentic Flow -----
search_agent = GoogleSearchAgent(api_key=GOOGLE_API_KEY, cx=GOOGLE_CX)
llm_agent = FlashpointLLMAgent(key=OPENAI_API_KEY)
entity_tracker = EntityTracker()

def fetch_context(state):
    entity_tracker.search_run += 1
    exclude_terms = entity_tracker.get_exclude_query()
    query = "global tension last 24 hours " + exclude_terms
    print("[SearchAgent] Fetching news context...")
    state["context"] = search_agent.gather_context(query)
    return state

def reason_json(state):
    entity_tracker.llm_run += 1
    print("[LLMAgent] Generating flashpoints...")
    response = llm_agent.call_llm(state["context"])
    validated = llm_agent.validate_json(response)

    if validated:
        existing = state.get("accumulated", [])
        for item in validated:
            overlap_found = False
            for existing_item in existing:
                if set(item.entities) & set(existing_item.entities):
                    existing_item.title += f" / {item.title}"
                    existing_item.description += f" {item.description}"
                    existing_item.entities = list(set(existing_item.entities + item.entities))
                    entity_tracker.update_seen_entities(item.entities)
                    overlap_found = True                   
                    break
                
            # Ensure at least one entity is a recognized country 
            
            has_country = any(is_country(ent) for ent in item.entities)          
                
            if has_country and not overlap_found and entity_tracker.is_new_combo(item.entities):
                entity_tracker.add(item.entities)
                existing.append(item)                

        state["accumulated"] = existing
        state["valid"] = True
    else:
        state["valid"] = False
    return state

def end(state):
    return state

builder = StateGraph(state_schema=dict)
builder.add_node("search", fetch_context)
builder.add_node("reason", reason_json)
builder.add_node("end", end)
builder.set_entry_point("search")
builder.add_edge("search", "reason")
builder.add_conditional_edges(
    "reason",
    lambda s: "done" if len(s.get("accumulated", [])) >= 10 else "search",
    {
        "done": "end",
        "search": "search"
    }
)
graph = builder.compile()

# ----- Run the Agent -----
if __name__ == "__main__":
    print("Starting MindSearch Flashpoint Detection...")
    print("="*50)
    
    final = graph.invoke({})
    
    print(f"\nSearch runs: {entity_tracker.search_run}")
    print(f"LLM runs: {entity_tracker.llm_run}")
    print(json.dumps([fp.model_dump() for fp in final["accumulated"]], indent=2, ensure_ascii=False))
    
    # Print final token cost summary
    token_tracker.print_final_summary()

