# -*- coding: utf-8 -*-
# =============================================================================
# ChefGPT AI - Intelligent Recipe Generator & Cooking Assistant
# =============================================================================
# Built with : Python, Flask, IBM watsonx.ai Studio, IBM Granite Models, RAG
# Purpose    : Agentic AI demonstration for IBM SkillsBuild, hackathons,
#              academic projects, and AI showcases.
# Architecture: 5 Specialized Agents + Master Orchestrator + Document RAG
# =============================================================================

import os
import re
import json
import math
import datetime
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

# ── Optional PDF support ──────────────────────────────────────────────────────
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ── IBM watsonx.ai SDK ────────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    print("[WARN] ibm-watsonx-ai SDK not installed. Running in DEMO mode.")

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
load_dotenv()

WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# =============================================================================
# FLASK APPLICATION
# =============================================================================
app = Flask(__name__)

# In-memory RAG document store (no database required)
RAG_DOCUMENTS     = []   # [{"title": str, "chunks": [str]}]
CONVERSATION_LOG  = []   # simple conversation history

# =============================================================================
# IBM watsonx.ai — MODEL INITIALISATION
# =============================================================================
# All AI reasoning is powered by IBM Granite Models via watsonx.ai Studio.

def get_watsonx_model():
    """
    Initialise IBM Granite 13B Instruct model via watsonx.ai SDK.
    Returns None when credentials are missing (demo mode).

    IBM watsonx.ai Integration Point STAR
    """
    if not WATSONX_AVAILABLE:
        return None
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return None
    try:
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        model = ModelInference(
            model_id   = "ibm/granite-13b-instruct-v2",
            credentials= credentials,
            project_id = WATSONX_PROJECT_ID,
            params     = {
                "max_new_tokens"    : 900,
                "min_new_tokens"    : 30,
                "temperature"       : 0.7,
                "top_p"             : 0.9,
                "repetition_penalty": 1.1,
            }
        )
        return model
    except Exception as exc:
        print(f"[WARN] watsonx.ai init failed: {exc}")
        return None

_MODEL = None

def get_model():
    """Return (lazily-initialised) singleton Granite model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = get_watsonx_model()
    return _MODEL

# =============================================================================
# CORE FUNCTION — generate_response()
# =============================================================================
def generate_response(prompt: str, max_tokens: int = 900) -> str:
    """
    Send a prompt to IBM Granite on watsonx.ai and return the text response.
    Falls back to contextual demo responses when credentials are absent.

    IBM watsonx.ai Integration Point STAR
    """
    model = get_model()
    if model:
        try:
            result = model.generate_text(prompt=prompt)
            return result.strip() if isinstance(result, str) else str(result).strip()
        except Exception as exc:
            print(f"[ERROR] generate_response: {exc}")
            return _demo_response(prompt)
    return _demo_response(prompt)


def _demo_response(prompt: str) -> str:
    """Offline demo fallback with contextually relevant placeholder content."""
    p = prompt.lower()

    if "chocolate cake" in p or "cake" in p:
        return ("Classic Chocolate Cake Recipe:\n"
                "Ingredients: 2 cups flour, 2 cups sugar, 3/4 cup cocoa powder, "
                "2 eggs, 1 cup milk, 1/2 cup oil, 2 tsp vanilla, 2 tsp baking soda.\n"
                "Instructions: Mix dry ingredients. Combine wet ingredients separately. "
                "Fold together. Bake at 350F (175C) for 30-35 minutes.\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    if "pasta" in p or "mushroom" in p:
        return ("Creamy Mushroom Pasta:\n"
                "Ingredients: 300g pasta, 200g mushrooms, 2 cloves garlic, "
                "1 cup heavy cream, parmesan, olive oil, salt, pepper, fresh herbs.\n"
                "Instructions: Cook pasta al dente. Saute mushrooms and garlic in olive oil. "
                "Add cream, simmer 5 minutes. Toss with pasta. Top with parmesan.\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    if "vegan" in p:
        return ("Vegan Adaptation: Replace dairy with plant-based alternatives. "
                "Use coconut milk instead of cream, nutritional yeast instead of parmesan, "
                "flax eggs (1 tbsp flaxseed + 3 tbsp water) instead of eggs. "
                "Replace butter with coconut oil or vegan margarine.\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    if "substitut" in p or "replace" in p or "instead" in p:
        return ("Ingredient Substitutions:\n"
                "Eggs: Use 1/4 cup unsweetened applesauce, 1 mashed banana, "
                "or 1 tbsp flaxseed + 3 tbsp water per egg.\n"
                "Butter: Use equal amounts of coconut oil, applesauce, or Greek yogurt.\n"
                "Milk: Use almond, oat, soy, or coconut milk in equal quantities.\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    if "nutrition" in p or "calorie" in p or "protein" in p:
        return ("Nutritional Analysis (estimated per serving):\n"
                "Calories: 380 kcal | Protein: 8g | Carbohydrates: 52g | "
                "Fat: 16g | Fiber: 3g | Sugar: 28g\n"
                "Health Note: This recipe provides moderate energy. Consider reducing sugar "
                "by 25% for a lighter version. Adding nuts increases protein content.\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    if "shopping" in p or "ingredient" in p or "list" in p:
        return ("Smart Shopping List:\n"
                "Vegetables: mushrooms (200g), garlic (1 head), onion (2)\n"
                "Dairy: heavy cream (1 cup), parmesan (100g), butter (50g)\n"
                "Pantry: pasta (300g), olive oil, salt, black pepper\n"
                "Spices: oregano, basil, thyme\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    if "breakfast" in p or "protein" in p:
        return ("High-Protein Breakfast Bowl:\n"
                "Ingredients: 3 eggs, 1/2 cup Greek yogurt, 1/4 cup oats, "
                "mixed berries, 1 tbsp chia seeds, honey.\n"
                "Instructions: Scramble eggs. Layer yogurt, oats, berries. "
                "Top with chia seeds and drizzle of honey. Protein: ~28g per serving.\n"
                "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

    return ("Here is a delicious recipe suggestion based on your request. "
            "I can help you adapt it for dietary needs, generate a shopping list, "
            "or analyze its nutritional content. What would you like to do next?\n"
            "[DEMO MODE - connect IBM watsonx.ai credentials for live Granite responses]")

# =============================================================================
# LIGHTWEIGHT RAG SYSTEM
# =============================================================================
# Extracts text from uploaded cookbooks/recipe PDFs, chunks content,
# and retrieves relevant passages to ground Granite Model responses.

def extract_text_from_file(file_storage) -> str:
    """Extract raw text from an uploaded TXT or PDF cookbook."""
    filename = file_storage.filename.lower()
    if filename.endswith(".txt"):
        return file_storage.read().decode("utf-8", errors="ignore")
    if filename.endswith(".pdf") and PDF_SUPPORT:
        reader = PyPDF2.PdfReader(file_storage)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_storage.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 70) -> list:
    """Split recipe text into overlapping chunks for better retrieval."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 20]


def _relevance_score(query_words: set, chunk: str) -> float:
    """Lightweight TF-IDF-style keyword relevance scoring — no external libs."""
    chunk_words = chunk.lower().split()
    total       = len(chunk_words) if chunk_words else 1
    score       = 0.0
    for word in query_words:
        tf  = chunk_words.count(word) / total
        idf = math.log(1 + 1 / (1 + chunk_words.count(word)))
        score += tf * idf
    return score


def retrieve_recipe_context(query: str, top_k: int = 3) -> str:
    """
    RAG retrieval — find the most relevant recipe passages for a query.
    Retrieved chunks are injected as context into Granite Model prompts.

    IBM watsonx.ai Integration Point STAR
    """
    if not RAG_DOCUMENTS:
        return ""

    query_words = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
    scored      = []

    for doc in RAG_DOCUMENTS:
        for chunk in doc["chunks"]:
            score = _relevance_score(query_words, chunk)
            scored.append((score, chunk, doc["title"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top or top[0][0] == 0:
        return ""

    return "\n\n".join(f"[Source: {t}]\n{c}" for _, c, t in top)

# =============================================================================
# AGENT 1 — Recipe Retrieval Agent
# =============================================================================
def recipe_retrieval_agent(query: str) -> dict:
    """
    Searches the RAG knowledge base and retrieves the most relevant recipe
    information, ingredients, and cooking notes.

    IBM watsonx.ai Integration Point STAR
    """
    rag_context = retrieve_recipe_context(query)
    ctx_section = f"\n\nRecipe Knowledge Base:\n{rag_context}" if rag_context else ""

    prompt = (
        f"You are an expert chef and recipe specialist with access to a recipe knowledge base.\n"
        f"Your task is to retrieve and present the most relevant recipe information for the user's query.\n"
        f"Include: recipe name, key ingredients, basic instructions, and cooking notes.{ctx_section}\n\n"
        f"User Query: {query}\n\n"
        f"Provide a clear, detailed recipe response:"
    )

    response = generate_response(prompt)

    return {
        "agent"    : "Recipe Retrieval Agent",
        "icon"     : "SEARCH",
        "purpose"  : "RAG-Based Recipe Search",
        "response" : response,
        "rag_used" : bool(rag_context),
        "rag_docs" : len(RAG_DOCUMENTS),
    }

# =============================================================================
# AGENT 2 — Recipe Adaptation Agent
# =============================================================================
def recipe_adaptation_agent(recipe_context: str, dietary_pref: str) -> dict:
    """
    Customises recipes based on dietary constraints using Granite reasoning.
    Supports: vegetarian, vegan, gluten-free, keto, sugar-free, high-protein.

    IBM watsonx.ai Integration Point STAR
    """
    prompt = (
        f"You are a professional culinary nutritionist specialising in dietary adaptations.\n"
        f"Adapt the following recipe to meet the dietary requirement: {dietary_pref}\n\n"
        f"Original Recipe:\n{recipe_context}\n\n"
        f"Provide:\n"
        f"1. Modified Ingredients List (with substitutions highlighted)\n"
        f"2. Updated Step-by-Step Instructions\n"
        f"3. Alternative Options if needed\n"
        f"4. Brief note on how the adaptations preserve flavour\n\n"
        f"Adapted {dietary_pref} Recipe:"
    )

    response = generate_response(prompt)

    return {
        "agent"          : "Recipe Adaptation Agent",
        "icon"           : "ADAPT",
        "purpose"        : f"Dietary Adaptation: {dietary_pref}",
        "dietary_pref"   : dietary_pref,
        "response"       : response,
    }

# =============================================================================
# AGENT 3 — Ingredient Substitution Agent
# =============================================================================
def substitution_agent(ingredient: str, reason: str = "") -> dict:
    """
    Suggests practical ingredient replacements with quantity conversions
    and reasoning, powered by IBM Granite.

    IBM watsonx.ai Integration Point STAR
    """
    context = f" Reason for substitution: {reason}" if reason else ""

    prompt = (
        f"You are an expert culinary advisor specialising in ingredient substitutions.\n"
        f"Suggest the best alternatives for: {ingredient}{context}\n\n"
        f"For each substitute provide:\n"
        f"1. Substitute Name\n"
        f"2. Quantity Conversion (e.g., 1 egg = 1/4 cup applesauce)\n"
        f"3. Why it works (brief culinary reasoning)\n"
        f"4. Best use cases (baking, savory, etc.)\n"
        f"5. Any flavour or texture differences to expect\n\n"
        f"Provide 3-4 practical substitution options:"
    )

    response = generate_response(prompt)

    return {
        "agent"     : "Ingredient Substitution Agent",
        "icon"      : "SWAP",
        "purpose"   : f"Substitutes for: {ingredient}",
        "ingredient": ingredient,
        "reason"    : reason,
        "response"  : response,
    }

# =============================================================================
# AGENT 4 — Nutrition Analysis Agent
# =============================================================================
def analyze_nutrition(recipe_text: str) -> dict:
    """
    Analyses the nutritional profile of a recipe and provides health insights
    using IBM Granite Model reasoning.

    IBM watsonx.ai Integration Point STAR
    """
    prompt = (
        f"You are a certified nutritionist and dietitian.\n"
        f"Analyse the nutritional content of the following recipe and provide estimates per serving.\n\n"
        f"Recipe:\n{recipe_text}\n\n"
        f"Provide a structured nutritional analysis:\n"
        f"1. Calories (kcal)\n"
        f"2. Protein (g)\n"
        f"3. Carbohydrates (g)\n"
        f"4. Fat (g)\n"
        f"5. Fiber (g)\n"
        f"6. Sugar (g)\n"
        f"7. Health Highlights (2-3 key points)\n"
        f"8. Recommendation for improvement\n\n"
        f"Nutritional Analysis:"
    )

    response = generate_response(prompt)

    # Extract approximate numbers for UI display
    nums = re.findall(r'(\d+(?:\.\d+)?)\s*(?:kcal|g\b|cal)', response.lower())

    return {
        "agent"      : "Nutrition Analysis Agent",
        "icon"       : "NUTRITION",
        "purpose"    : "Recipe Nutritional Analysis",
        "response"   : response,
        "calories"   : nums[0] if len(nums) > 0 else "~350",
        "protein"    : nums[1] if len(nums) > 1 else "~8",
        "carbs"      : nums[2] if len(nums) > 2 else "~45",
        "fat"        : nums[3] if len(nums) > 3 else "~14",
    }

# =============================================================================
# AGENT 5 — Shopping List Agent
# =============================================================================
def shopping_list_agent(recipe_text: str) -> dict:
    """
    Extracts ingredients from a recipe and organises them into a smart,
    categorised shopping list using IBM Granite.

    IBM watsonx.ai Integration Point STAR
    """
    prompt = (
        f"You are a smart kitchen assistant and grocery planner.\n"
        f"Extract all ingredients from the recipe below and create an organised shopping list.\n\n"
        f"Recipe:\n{recipe_text}\n\n"
        f"Create a categorised shopping list:\n\n"
        f"VEGETABLES & FRUITS:\n- [items with quantities]\n\n"
        f"PROTEINS (Meat, Fish, Eggs, Legumes):\n- [items with quantities]\n\n"
        f"DAIRY & ALTERNATIVES:\n- [items with quantities]\n\n"
        f"GRAINS & PANTRY:\n- [items with quantities]\n\n"
        f"SPICES & CONDIMENTS:\n- [items with quantities]\n\n"
        f"OTHER:\n- [items with quantities]\n\n"
        f"Shopping List:"
    )

    response = generate_response(prompt)

    return {
        "agent"   : "Shopping List Agent",
        "icon"    : "CART",
        "purpose" : "Smart Shopping List Generator",
        "response": response,
    }

# =============================================================================
# MASTER ORCHESTRATOR AGENT
# =============================================================================

def _classify_recipe_intent(query: str) -> str:
    """
    Rule-based intent classifier for recipe query routing.
    Returns: retrieval | adaptation | substitution | nutrition | shopping | general
    """
    q = query.lower()

    substitution_triggers = [
        "substitute", "replace", "instead of", "without", "no eggs",
        "no butter", "alternative", "swap", "can i use", "what if i don't have"
    ]
    if any(t in q for t in substitution_triggers):
        return "substitution"

    nutrition_triggers = [
        "calorie", "nutrition", "protein", "carb", "fat", "healthy",
        "diet", "macro", "vitamin", "fiber", "how many calories"
    ]
    if any(t in q for t in nutrition_triggers):
        return "nutrition"

    shopping_triggers = [
        "shopping list", "grocery", "buy", "ingredients needed",
        "what do i need", "ingredient list", "shopping"
    ]
    if any(t in q for t in shopping_triggers):
        return "shopping"

    adaptation_triggers = [
        "vegan", "vegetarian", "gluten-free", "gluten free", "keto",
        "sugar-free", "sugar free", "dairy-free", "dairy free",
        "high-protein", "low-calorie", "low calorie", "make it", "adapt"
    ]
    if any(t in q for t in adaptation_triggers):
        return "adaptation"

    return "retrieval"


def _extract_dietary_pref(query: str) -> str:
    """Extract the dietary preference from a query string."""
    q = query.lower()
    prefs = ["vegan", "vegetarian", "gluten-free", "keto",
             "sugar-free", "dairy-free", "high-protein", "low-calorie"]
    for p in prefs:
        if p in q:
            return p.title()
    return "Healthy"


def orchestrate_agents(query: str, dietary_pref: str = "",
                        ingredient_sub: str = "") -> dict:
    """
    Master Orchestrator — the brain of ChefGPT AI.

    Workflow:
    1. Classify query intent
    2. Always run Recipe Retrieval Agent (RAG-backed)
    3. Route to specialist agents based on intent
    4. Run Nutrition + Shopping List Agents on the retrieved recipe
    5. Combine all outputs into a unified cooking guide

    IBM watsonx.ai Integration Point STAR — all agent calls use Granite Models.
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # ── STEP 1: Classify intent ───────────────────────────────────────────────
    intent = _classify_recipe_intent(query)

    # Override with explicit dietary pref if provided
    if dietary_pref and dietary_pref not in ("", "None", "none"):
        intent = "adaptation"

    # ── STEP 2: Always retrieve recipe context (RAG) ──────────────────────────
    retrieval_result = recipe_retrieval_agent(query)
    base_recipe      = retrieval_result["response"]

    # ── STEP 3: Route to specialist agents ────────────────────────────────────
    agent_reason      = ""
    adaptation_result = {}
    substitution_result = {}

    if intent == "adaptation":
        pref = dietary_pref or _extract_dietary_pref(query)
        adaptation_result = recipe_adaptation_agent(base_recipe, pref)
        agent_reason = f"Dietary adaptation requested: {pref}. Routing to Recipe Adaptation Agent."

    elif intent == "substitution":
        sub_ingredient = ingredient_sub or query
        adaptation_result = substitution_agent(sub_ingredient)
        agent_reason = f"Ingredient substitution query detected. Routing to Substitution Agent."

    elif intent == "nutrition":
        agent_reason = "Nutritional query detected. Running full nutrition analysis."

    elif intent == "shopping":
        agent_reason = "Shopping list requested. Extracting and categorising ingredients."

    else:
        agent_reason = "Recipe retrieval query. Searching RAG knowledge base with Granite Models."

    # ── STEP 4: Always run Nutrition + Shopping List Agents ───────────────────
    nutrition_result     = analyze_nutrition(base_recipe)
    shopping_result      = shopping_list_agent(base_recipe)

    # ── STEP 5: Run substitution agent if ingredient_sub provided ─────────────
    if ingredient_sub and intent != "substitution":
        substitution_result = substitution_agent(ingredient_sub)

    # ── STEP 6: Log conversation ──────────────────────────────────────────────
    CONVERSATION_LOG.append({
        "timestamp": timestamp,
        "query"    : query,
        "intent"   : intent,
        "rag_docs" : len(RAG_DOCUMENTS),
    })

    # ── STEP 7: Build agents activated list ───────────────────────────────────
    agents_activated = ["Recipe Retrieval Agent", "Nutrition Analysis Agent", "Shopping List Agent"]
    if adaptation_result:
        agents_activated.insert(1, adaptation_result.get("agent", "Recipe Adaptation Agent"))
    if substitution_result:
        agents_activated.insert(2, substitution_result.get("agent", "Ingredient Substitution Agent"))

    return {
        "timestamp"          : timestamp,
        "query"              : query,
        "intent"             : intent,
        "agent_reason"       : agent_reason,
        "retrieval_result"   : retrieval_result,
        "adaptation_result"  : adaptation_result,
        "substitution_result": substitution_result,
        "nutrition_result"   : nutrition_result,
        "shopping_result"    : shopping_result,
        "agents_activated"   : agents_activated,
        "rag_docs_count"     : len(RAG_DOCUMENTS),
    }

# =============================================================================
# HTML TEMPLATE — Complete Single-Page Application
# =============================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ChefGPT AI - Intelligent Recipe Generator</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    :root {
      --primary   : #e67e22;
      --secondary : #d35400;
      --accent    : #27ae60;
      --light-bg  : #fdf6ec;
      --card-bg   : #ffffff;
      --border    : #f0dfc8;
      --text      : #2c2c2c;
      --muted     : #7a7a7a;
      --shadow    : 0 2px 14px rgba(230,126,34,0.10);
    }
    * { box-sizing: border-box; }
    body {
      background: var(--light-bg);
      font-family: 'Segoe UI', system-ui, sans-serif;
      color: var(--text);
      min-height: 100vh;
    }

    /* HEADER */
    .app-header {
      background: linear-gradient(135deg, #1a0a00 0%, #8b3a00 45%, #e67e22 100%);
      color: white;
      padding: 1.4rem 2rem;
      box-shadow: 0 4px 20px rgba(139,58,0,0.35);
    }
    .brand { font-size: 1.75rem; font-weight: 800; letter-spacing: -0.5px; }
    .sub   { font-size: 0.83rem; opacity: .85; }
    .ibm-badge {
      background: rgba(255,255,255,.15);
      border: 1px solid rgba(255,255,255,.3);
      border-radius: 20px; padding: 3px 13px;
      font-size: 0.75rem; font-weight: 600;
    }

    /* CARDS */
    .panel-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.2rem;
      box-shadow: var(--shadow);
      height: 100%;
    }
    .panel-title {
      font-size: 0.72rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: .9px;
      color: var(--muted);
      border-bottom: 2px solid var(--light-bg);
      padding-bottom: .5rem; margin-bottom: .8rem;
    }

    /* CHAT */
    #chat-box {
      height: 380px; overflow-y: auto;
      padding: 1rem; background: #fdf9f4;
      border-radius: 10px; border: 1px solid var(--border);
      scroll-behavior: smooth;
    }
    .msg-user {
      background: var(--primary); color: white;
      border-radius: 16px 16px 4px 16px;
      padding: 9px 15px; margin: 6px 0 6px auto;
      max-width: 80%; width: fit-content;
      font-size: .9rem; word-break: break-word;
    }
    .msg-ai {
      background: white; color: var(--text);
      border: 1px solid var(--border);
      border-radius: 16px 16px 16px 4px;
      padding: 10px 15px; margin: 6px auto 6px 0;
      max-width: 87%; width: fit-content;
      font-size: .9rem; line-height: 1.65; word-break: break-word;
    }
    .msg-ai strong { color: var(--primary); }

    /* INPUT */
    #user-input {
      border-radius: 25px 0 0 25px;
      border: 1.5px solid var(--border);
      padding: .6rem 1.2rem; font-size: .93rem;
    }
    #user-input:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(230,126,34,.12); outline: none; }
    #send-btn {
      border-radius: 0 25px 25px 0;
      background: var(--primary); border: none;
      color: white; padding: 0 1.3rem;
      font-size: .95rem; font-weight: 700; cursor: pointer;
      transition: background .2s;
    }
    #send-btn:hover { background: var(--secondary); }

    /* AGENT CARDS */
    .agent-card {
      border: 1.5px solid var(--border);
      border-radius: 10px; padding: .8rem 1rem;
      margin-bottom: .55rem; background: #fdf9f4;
      transition: border-color .3s, background .3s, transform .2s;
    }
    .agent-card.active  { border-color: var(--primary); background: #fff4e8; transform: translateX(3px); }
    .agent-card.fired   { border-color: var(--accent);  background: #edf7f0; }
    .agent-name  { font-weight: 700; font-size: .87rem; }
    .agent-role  { font-size: .77rem; color: var(--muted); }
    .agent-badge { font-size: .68rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
    .badge-active { background: #fde8cc; color: #b05a00; }
    .badge-fired  { background: #d5f0e5; color: #1a7a4a; }
    .badge-idle   { background: #ececec; color: #888; }

    /* WORKFLOW STEPS */
    .workflow-step {
      display: flex; align-items: center; gap: 8px;
      padding: 4px 0; font-size: .82rem; color: var(--muted);
    }
    .step-dot { width: 7px; height: 7px; border-radius: 50%; background: #ddd; flex-shrink: 0; }
    .workflow-step.done   .step-dot { background: var(--accent); }
    .workflow-step.active .step-dot { background: var(--primary); }

    /* NUTRITION BARS */
    .nut-bar-wrap { margin-bottom: 8px; }
    .nut-label  { display: flex; justify-content: space-between; font-size: .8rem; margin-bottom: 3px; }
    .nut-bar    { height: 8px; border-radius: 4px; background: #f0e0cc; overflow: hidden; }
    .nut-fill   { height: 100%; border-radius: 4px; transition: width .8s ease; }
    .fill-cal   { background: #e74c3c; }
    .fill-prot  { background: #3498db; }
    .fill-carb  { background: #f39c12; }
    .fill-fat   { background: #9b59b6; }

    /* RECIPE CARD */
    .recipe-section {
      background: white; border-radius: 10px;
      border: 1px solid var(--border); padding: 1rem;
      margin-bottom: .8rem; font-size: .85rem; line-height: 1.7;
      max-height: 220px; overflow-y: auto;
    }

    /* SHOPPING LIST */
    .shop-category {
      font-size: .72rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: .7px; color: var(--primary); margin: 8px 0 4px;
    }
    .shop-item {
      border-left: 3px solid var(--primary); padding: 4px 10px;
      margin-bottom: 4px; background: #fff8f0;
      border-radius: 0 6px 6px 0; font-size: .82rem;
    }

    /* QUICK PROMPTS */
    .quick-btn {
      border: 1.5px solid var(--primary); color: var(--primary);
      background: white; border-radius: 20px;
      padding: 3px 12px; font-size: .78rem;
      cursor: pointer; transition: all .2s; white-space: nowrap;
    }
    .quick-btn:hover { background: var(--primary); color: white; }

    /* THINKING */
    .thinking-indicator {
      display: flex; align-items: center; gap: 8px;
      color: var(--muted); font-size: .84rem; padding: 8px 0;
    }
    .dot-bounce { display: flex; gap: 4px; }
    .dot-bounce span {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--primary); animation: bounce 1.2s infinite ease-in-out;
    }
    .dot-bounce span:nth-child(2) { animation-delay: .2s; }
    .dot-bounce span:nth-child(3) { animation-delay: .4s; }
    @keyframes bounce {
      0%,80%,100% { transform: scale(.7); opacity: .5; }
      40%         { transform: scale(1);  opacity: 1;  }
    }

    /* MISC */
    .section-divider { border: none; border-top: 1.5px solid var(--light-bg); margin: .7rem 0; }
    .rag-badge { font-size: .7rem; background: #fff4e0; color: #a06000;
                 border: 1px solid #f0d090; border-radius: 10px; padding: 2px 8px; }
    .dietary-select {
      border-radius: 8px; border: 1.5px solid var(--border);
      padding: 5px 10px; font-size: .85rem; background: white;
    }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #f5ede0; }
    ::-webkit-scrollbar-thumb { background: #e0b080; border-radius: 3px; }

    @media (max-width:768px) { .brand { font-size: 1.25rem; } #chat-box { height: 260px; } }
  </style>
</head>
<body>

<!-- HEADER -->
<header class="app-header">
  <div class="container-fluid">
    <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
      <div>
        <div class="brand">&#127859; ChefGPT AI</div>
        <div class="sub">Intelligent Recipe Generator &amp; Cooking Assistant</div>
      </div>
      <div class="d-flex gap-2 flex-wrap align-items-center">
        <span class="ibm-badge">&#128311; IBM watsonx.ai Studio</span>
        <span class="ibm-badge">&#11035; IBM Granite Models</span>
        <span class="ibm-badge">&#128196; RAG System</span>
        <span class="ibm-badge">&#129302; Agentic AI</span>
      </div>
    </div>
  </div>
</header>

<div class="container-fluid py-3 px-3 px-md-4">
<div class="row g-3">

  <!-- ====== LEFT: CHAT + AGENTS ====== -->
  <div class="col-12 col-lg-5">

    <!-- CHAT PANEL -->
    <div class="panel-card mb-3">
      <div class="panel-title"><i class="bi bi-chat-dots-fill me-1"></i>Recipe Chat</div>

      <!-- Quick prompts -->
      <div class="d-flex flex-wrap gap-1 mb-2">
        <button class="quick-btn" onclick="setQ('Show me a chocolate cake recipe')">Chocolate Cake</button>
        <button class="quick-btn" onclick="setQ('Vegan pasta recipe')">Vegan Pasta</button>
        <button class="quick-btn" onclick="setQ('What can replace eggs in baking?')">Egg Substitute</button>
        <button class="quick-btn" onclick="setQ('High protein breakfast ideas')">High Protein</button>
        <button class="quick-btn" onclick="setQ('Generate a shopping list for pasta carbonara')">Shopping List</button>
      </div>

      <div id="chat-box">
        <div class="msg-ai">
          <strong>&#127859; ChefGPT AI</strong><br/>
          Welcome! I'm your intelligent cooking assistant powered by
          <strong>IBM Granite Models</strong> on <strong>IBM watsonx.ai Studio</strong>.<br/><br/>
          I can help you find recipes, adapt them for dietary needs, suggest ingredient
          substitutions, analyse nutrition, and generate shopping lists.<br/><br/>
          <em>Upload a cookbook PDF/TXT to activate the RAG knowledge base, or just ask me anything!</em>
        </div>
      </div>

      <!-- Dietary preference selector -->
      <div class="d-flex gap-2 mt-2 mb-2 flex-wrap align-items-center">
        <label style="font-size:.8rem;color:var(--muted);white-space:nowrap;">Dietary Preference:</label>
        <select id="dietary-select" class="dietary-select">
          <option value="">None</option>
          <option value="Vegan">Vegan</option>
          <option value="Vegetarian">Vegetarian</option>
          <option value="Gluten-Free">Gluten-Free</option>
          <option value="Keto">Keto</option>
          <option value="Sugar-Free">Sugar-Free</option>
          <option value="High-Protein">High-Protein</option>
          <option value="Low-Calorie">Low-Calorie</option>
        </select>
        <input id="ingredient-sub" type="text" placeholder="Ingredient to replace (optional)"
               style="border-radius:8px;border:1.5px solid var(--border);padding:5px 10px;font-size:.82rem;flex:1;min-width:140px;"/>
      </div>

      <!-- Input bar -->
      <div class="d-flex mt-1">
        <input id="user-input" type="text" class="form-control"
               placeholder="Ask about recipes, adaptations, nutrition..."
               onkeypress="if(event.key==='Enter') sendMessage()"/>
        <button id="send-btn" onclick="sendMessage()">
          <i class="bi bi-send-fill"></i>
        </button>
      </div>
    </div>

    <!-- AGENT WORKFLOW PANEL -->
    <div class="panel-card">
      <div class="panel-title"><i class="bi bi-diagram-3-fill me-1"></i>Agent Orchestration Workflow</div>

      <div id="workflow-steps" class="mb-2">
        <div class="workflow-step" id="ws1"><div class="step-dot"></div><span>Receive user query</span></div>
        <div class="workflow-step" id="ws2"><div class="step-dot"></div><span>Classify intent &amp; route</span></div>
        <div class="workflow-step" id="ws3"><div class="step-dot"></div><span>RAG document retrieval</span></div>
        <div class="workflow-step" id="ws4"><div class="step-dot"></div><span>IBM Granite generates recipe</span></div>
        <div class="workflow-step" id="ws5"><div class="step-dot"></div><span>Specialist agents process</span></div>
        <div class="workflow-step" id="ws6"><div class="step-dot"></div><span>Combine &amp; return unified guide</span></div>
      </div>

      <hr class="section-divider"/>
      <div id="agent-reason" style="font-size:.82rem;color:var(--muted);min-height:24px;">
        Agent selection reason will appear here after your first query.
      </div>
      <hr class="section-divider"/>

      <!-- Agent cards -->
      <div id="agent-cards">
        <div class="agent-card" id="card-orch">
          <div class="d-flex justify-content-between align-items-center">
            <div><div class="agent-name">&#127919; Master Orchestrator</div><div class="agent-role">Query routing &amp; agent coordination</div></div>
            <span class="agent-badge badge-idle" id="badge-orch">Idle</span>
          </div>
        </div>
        <div class="agent-card" id="card-retrieval">
          <div class="d-flex justify-content-between align-items-center">
            <div><div class="agent-name">&#128269; Recipe Retrieval Agent</div><div class="agent-role">RAG-based recipe search &amp; context</div></div>
            <span class="agent-badge badge-idle" id="badge-retrieval">Idle</span>
          </div>
        </div>
        <div class="agent-card" id="card-adapt">
          <div class="d-flex justify-content-between align-items-center">
            <div><div class="agent-name">&#127807; Recipe Adaptation Agent</div><div class="agent-role">Dietary customisation &amp; modification</div></div>
            <span class="agent-badge badge-idle" id="badge-adapt">Idle</span>
          </div>
        </div>
        <div class="agent-card" id="card-sub">
          <div class="d-flex justify-content-between align-items-center">
            <div><div class="agent-name">&#8645; Substitution Agent</div><div class="agent-role">Ingredient replacement suggestions</div></div>
            <span class="agent-badge badge-idle" id="badge-sub">Idle</span>
          </div>
        </div>
        <div class="agent-card" id="card-nut">
          <div class="d-flex justify-content-between align-items-center">
            <div><div class="agent-name">&#128202; Nutrition Analysis Agent</div><div class="agent-role">Calories, macros &amp; health insights</div></div>
            <span class="agent-badge badge-idle" id="badge-nut">Idle</span>
          </div>
        </div>
        <div class="agent-card" id="card-shop">
          <div class="d-flex justify-content-between align-items-center">
            <div><div class="agent-name">&#128722; Shopping List Agent</div><div class="agent-role">Smart categorised ingredient lists</div></div>
            <span class="agent-badge badge-idle" id="badge-shop">Idle</span>
          </div>
        </div>
      </div>
    </div>

  </div><!-- end left col -->

  <!-- ====== RIGHT: DASHBOARDS ====== -->
  <div class="col-12 col-lg-7">
    <div class="row g-3">

      <!-- RETRIEVED RECIPE -->
      <div class="col-12">
        <div class="panel-card">
          <div class="panel-title"><i class="bi bi-book-fill me-1"></i>Retrieved Recipe
            <span id="rag-indicator" class="rag-badge ms-2" style="display:none;">RAG Active</span>
          </div>
          <div id="recipe-output" class="recipe-section">
            <em style="color:var(--muted);">Ask a recipe question to see results here. Upload a cookbook PDF for RAG-enhanced responses.</em>
          </div>
        </div>
      </div>

      <!-- ADAPTED RECIPE + SUBSTITUTIONS -->
      <div class="col-12 col-md-6">
        <div class="panel-card">
          <div class="panel-title"><i class="bi bi-arrow-repeat me-1"></i>Personalised Recipe</div>
          <div id="adapt-output" class="recipe-section" style="max-height:180px;">
            <em style="color:var(--muted);">Select a dietary preference or ask for an adaptation.</em>
          </div>
        </div>
      </div>

      <!-- SUBSTITUTIONS -->
      <div class="col-12 col-md-6">
        <div class="panel-card">
          <div class="panel-title"><i class="bi bi-shuffle me-1"></i>Ingredient Substitutions</div>
          <div id="sub-output" class="recipe-section" style="max-height:180px;">
            <em style="color:var(--muted);">Type an ingredient to replace in the field above, or ask "What replaces eggs?"</em>
          </div>
        </div>
      </div>

      <!-- NUTRITION PANEL -->
      <div class="col-12 col-md-6">
        <div class="panel-card">
          <div class="panel-title"><i class="bi bi-bar-chart-fill me-1"></i>Nutritional Facts</div>

          <div class="row text-center mb-2 g-1">
            <div class="col-3">
              <div style="font-size:1.3rem;font-weight:800;color:#e74c3c;" id="n-cal">--</div>
              <div style="font-size:.7rem;color:var(--muted);">Calories</div>
            </div>
            <div class="col-3">
              <div style="font-size:1.3rem;font-weight:800;color:#3498db;" id="n-prot">--</div>
              <div style="font-size:.7rem;color:var(--muted);">Protein g</div>
            </div>
            <div class="col-3">
              <div style="font-size:1.3rem;font-weight:800;color:#f39c12;" id="n-carb">--</div>
              <div style="font-size:.7rem;color:var(--muted);">Carbs g</div>
            </div>
            <div class="col-3">
              <div style="font-size:1.3rem;font-weight:800;color:#9b59b6;" id="n-fat">--</div>
              <div style="font-size:.7rem;color:var(--muted);">Fat g</div>
            </div>
          </div>

          <div class="nut-bar-wrap">
            <div class="nut-label"><span>Calories</span><span id="pct-cal">0%</span></div>
            <div class="nut-bar"><div class="nut-fill fill-cal" id="bar-cal" style="width:0%"></div></div>
          </div>
          <div class="nut-bar-wrap">
            <div class="nut-label"><span>Protein</span><span id="pct-prot">0%</span></div>
            <div class="nut-bar"><div class="nut-fill fill-prot" id="bar-prot" style="width:0%"></div></div>
          </div>
          <div class="nut-bar-wrap">
            <div class="nut-label"><span>Carbs</span><span id="pct-carb">0%</span></div>
            <div class="nut-bar"><div class="nut-fill fill-carb" id="bar-carb" style="width:0%"></div></div>
          </div>
          <div class="nut-bar-wrap">
            <div class="nut-label"><span>Fat</span><span id="pct-fat">0%</span></div>
            <div class="nut-bar"><div class="nut-fill fill-fat" id="bar-fat" style="width:0%"></div></div>
          </div>

          <hr class="section-divider"/>
          <div id="nut-detail" style="font-size:.82rem;line-height:1.6;color:var(--text);max-height:100px;overflow-y:auto;">
            <em style="color:var(--muted);">Nutrition analysis will appear here.</em>
          </div>
        </div>
      </div>

      <!-- SHOPPING LIST -->
      <div class="col-12 col-md-6">
        <div class="panel-card">
          <div class="panel-title"><i class="bi bi-cart3 me-1"></i>Smart Shopping List</div>
          <div id="shop-output" style="font-size:.83rem;max-height:280px;overflow-y:auto;">
            <em style="color:var(--muted);">Shopping list will be generated automatically after recipe retrieval.</em>
          </div>
        </div>
      </div>

      <!-- RAG UPLOAD -->
      <div class="col-12">
        <div class="panel-card">
          <div class="panel-title"><i class="bi bi-cloud-arrow-up-fill me-1"></i>RAG Knowledge Base — Upload Cookbook / Recipe Document</div>
          <div class="row align-items-center g-2">
            <div class="col-12 col-md-7">
              <p style="font-size:.82rem;color:var(--muted);margin-bottom:8px;">
                Upload a cookbook PDF or TXT file. The RAG system will extract, chunk, and
                index the content to ground IBM Granite Model responses with your recipe knowledge.
              </p>
              <div class="d-flex gap-2">
                <input type="file" id="rag-file" accept=".txt,.pdf"
                       style="font-size:.8rem;border:1.5px solid var(--border);border-radius:8px;padding:5px 10px;background:white;"/>
                <button class="btn btn-sm btn-outline-warning fw-bold" onclick="uploadDoc()" style="border-radius:8px;font-size:.82rem;">
                  <i class="bi bi-upload me-1"></i>Upload
                </button>
              </div>
            </div>
            <div class="col-12 col-md-5">
              <div style="font-size:.75rem;font-weight:700;color:var(--muted);margin-bottom:5px;">INDEXED DOCUMENTS</div>
              <div id="rag-docs-list" style="font-size:.8rem;max-height:75px;overflow-y:auto;">
                <em style="color:var(--muted);">No documents uploaded yet.</em>
              </div>
            </div>
          </div>
          <div id="rag-status" class="mt-2" style="font-size:.81rem;"></div>
        </div>
      </div>

    </div><!-- end inner row -->
  </div><!-- end right col -->

</div><!-- end main row -->

<!-- FOOTER -->
<div class="text-center mt-4 pt-3" style="border-top:1px solid var(--border);font-size:.75rem;color:var(--muted);">
  <strong>ChefGPT AI</strong> &middot; IBM Granite Models on IBM watsonx.ai Studio &middot;
  Agentic AI + RAG Architecture &middot; For IBM SkillsBuild, Hackathons &amp; Academic Projects
</div>
</div><!-- end container -->

<script>
// =============================================================================
// ChefGPT AI — Frontend JavaScript
// =============================================================================

const chatBox = document.getElementById('chat-box');
let ragDocNames = [];

function setQ(text) {
  document.getElementById('user-input').value = text;
  document.getElementById('user-input').focus();
}

function appendMsg(role, html) {
  const div = document.createElement('div');
  div.className = role === 'user' ? 'msg-user' : 'msg-ai';
  div.innerHTML = role === 'user' ? html :
    '<strong>&#127859; ChefGPT AI</strong><br/>' + html;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function fmt(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br/>');
}

function showThinking() {
  const div = document.createElement('div');
  div.className = 'thinking-indicator'; div.id = 'thinking';
  div.innerHTML = '<span style="color:var(--primary)">&#11035; IBM Granite is cooking up a response</span>' +
    '<div class="dot-bounce"><span></span><span></span><span></span></div>';
  chatBox.appendChild(div); chatBox.scrollTop = chatBox.scrollHeight;
}
function hideThinking() { const t = document.getElementById('thinking'); if (t) t.remove(); }

// ── Workflow animation ────────────────────────────────────────────────────────
function animateWorkflow() {
  const steps = ['ws1','ws2','ws3','ws4','ws5','ws6'];
  steps.forEach(s => {
    const el = document.getElementById(s);
    if (el) el.classList.remove('done','active');
  });
  let delay = 0;
  steps.forEach((s, i) => {
    setTimeout(() => {
      const el = document.getElementById(s); if (!el) return;
      el.classList.add('active');
      if (i > 0) { const prev = document.getElementById(steps[i-1]); if(prev){prev.classList.remove('active');prev.classList.add('done');} }
      if (i === steps.length-1) setTimeout(() => { el.classList.remove('active'); el.classList.add('done'); }, 600);
    }, delay);
    delay += 400;
  });
}

// ── Agent card updates ────────────────────────────────────────────────────────
function updateAgentCards(data) {
  const all = ['orch','retrieval','adapt','sub','nut','shop'];
  all.forEach(k => {
    const c = document.getElementById('card-'+k);
    const b = document.getElementById('badge-'+k);
    if (c) c.classList.remove('active','fired');
    if (b) { b.className='agent-badge badge-idle'; b.textContent='Idle'; }
  });

  // Orchestrator always active
  setCard('orch',  'active', 'Active');
  // Retrieval always fires
  setCard('retrieval', 'fired', 'Fired');
  // Nutrition always fires
  setCard('nut', 'fired', 'Fired');
  // Shopping always fires
  setCard('shop', 'fired', 'Fired');

  // Conditional
  if (data.adaptation_result && data.adaptation_result.agent) setCard('adapt','active','Primary');
  if (data.substitution_result && data.substitution_result.agent) setCard('sub','fired','Fired');

  document.getElementById('agent-reason').textContent =
    data.agent_reason || 'Orchestrator processed your request.';
}
function setCard(key, cls, label) {
  const c = document.getElementById('card-'+key);
  const b = document.getElementById('badge-'+key);
  if (c) { c.classList.remove('active','fired'); c.classList.add(cls); }
  if (b) { b.className='agent-badge '+(cls==='active'?'badge-active':'badge-fired'); b.textContent=label; }
}

// ── Nutrition panel ───────────────────────────────────────────────────────────
function updateNutrition(nr) {
  if (!nr) return;
  const cal  = parseInt(nr.calories)  || 350;
  const prot = parseInt(nr.protein)   || 8;
  const carb = parseInt(nr.carbs)     || 45;
  const fat  = parseInt(nr.fat)       || 14;

  document.getElementById('n-cal').textContent  = cal;
  document.getElementById('n-prot').textContent = prot;
  document.getElementById('n-carb').textContent = carb;
  document.getElementById('n-fat').textContent  = fat;

  const maxCal = 800, maxProt = 50, maxCarb = 100, maxFat = 50;
  setBar('bar-cal',  'pct-cal',  Math.min(100, cal/maxCal*100));
  setBar('bar-prot', 'pct-prot', Math.min(100, prot/maxProt*100));
  setBar('bar-carb', 'pct-carb', Math.min(100, carb/maxCarb*100));
  setBar('bar-fat',  'pct-fat',  Math.min(100, fat/maxFat*100));

  document.getElementById('nut-detail').innerHTML = fmt(nr.response || '');
}
function setBar(barId, pctId, pct) {
  const b = document.getElementById(barId);
  const p = document.getElementById(pctId);
  if (b) b.style.width = pct.toFixed(0) + '%';
  if (p) p.textContent = pct.toFixed(0) + '%';
}

// ── Shopping list formatter ───────────────────────────────────────────────────
function renderShoppingList(text) {
  const lines = text.split('\n');
  let html = '';
  lines.forEach(line => {
    line = line.trim(); if (!line) return;
    if (line.match(/^(VEGETABLES|PROTEINS|DAIRY|GRAINS|SPICES|OTHER|PANTRY)/i)) {
      html += `<div class="shop-category">${line}</div>`;
    } else if (line.startsWith('-') || line.startsWith('*')) {
      html += `<div class="shop-item">${line.slice(1).trim()}</div>`;
    } else {
      html += `<div style="font-size:.82rem;color:var(--muted);margin-bottom:3px;">${line}</div>`;
    }
  });
  return html || fmt(text);
}

// ── MAIN SEND ─────────────────────────────────────────────────────────────────
async function sendMessage() {
  const text    = document.getElementById('user-input').value.trim();
  const dietary = document.getElementById('dietary-select').value;
  const subIng  = document.getElementById('ingredient-sub').value.trim();
  if (!text) return;

  document.getElementById('user-input').value = '';
  appendMsg('user', text + (dietary ? ' <em>['+dietary+']</em>' : ''));
  showThinking();
  animateWorkflow();

  try {
    const resp = await fetch('/api/chef-chat', {
      method : 'POST',
      headers: {'Content-Type':'application/json'},
      body   : JSON.stringify({message: text, dietary_pref: dietary, ingredient_sub: subIng}),
    });
    const data = await resp.json();
    hideThinking();

    if (data.error) { appendMsg('ai','Error: '+data.error); return; }

    // Chat response from retrieval agent
    const rr = data.retrieval_result || {};
    appendMsg('ai', fmt(rr.response || 'Here is your recipe!'));

    // RAG indicator
    const ragInd = document.getElementById('rag-indicator');
    if (ragInd) ragInd.style.display = (rr.rag_used || data.rag_docs_count > 0) ? 'inline-block' : 'none';

    // Panels
    document.getElementById('recipe-output').innerHTML = fmt(rr.response || '');

    const ar = data.adaptation_result || {};
    document.getElementById('adapt-output').innerHTML =
      ar.response ? fmt(ar.response) : '<em style="color:var(--muted)">No dietary adaptation requested.</em>';

    const sr = data.substitution_result || {};
    document.getElementById('sub-output').innerHTML =
      sr.response ? fmt(sr.response) : '<em style="color:var(--muted)">No substitution requested.</em>';

    updateNutrition(data.nutrition_result);

    const shp = data.shopping_result || {};
    document.getElementById('shop-output').innerHTML =
      shp.response ? renderShoppingList(shp.response) : '';

    updateAgentCards(data);

  } catch(err) {
    hideThinking();
    appendMsg('ai', 'Connection error: ' + err.message);
  }
}

// ── RAG Upload ────────────────────────────────────────────────────────────────
async function uploadDoc() {
  const fileInput = document.getElementById('rag-file');
  const statusEl  = document.getElementById('rag-status');
  if (!fileInput.files.length) { statusEl.innerHTML='<span class="text-warning">Please select a file.</span>'; return; }

  const form = new FormData();
  form.append('file', fileInput.files[0]);
  statusEl.innerHTML = '<span class="text-muted">Uploading and indexing document...</span>';

  try {
    const resp = await fetch('/api/upload-recipe-doc', {method:'POST', body: form});
    const data = await resp.json();
    if (data.success) {
      ragDocNames.push(data.filename);
      document.getElementById('rag-docs-list').innerHTML =
        ragDocNames.map(n => `<div>&#9989; <span class="rag-badge">RAG</span> ${n}</div>`).join('');
      statusEl.innerHTML = `<span class="text-success">Indexed ${data.chunks} chunks from ${data.filename}.</span>`;
      document.getElementById('rag-indicator').style.display = 'inline-block';
    } else {
      statusEl.innerHTML = `<span class="text-danger">Error: ${data.error}</span>`;
    }
  } catch(err) {
    statusEl.innerHTML = `<span class="text-danger">Upload failed: ${err.message}</span>`;
  }
  fileInput.value = '';
}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
def index():
    """Serve the ChefGPT AI single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chef-chat", methods=["POST"])
def chef_chat():
    """
    Main chat endpoint — runs the full Agentic AI orchestration pipeline.

    IBM watsonx.ai Integration Point STAR
    """
    data           = request.get_json(force=True)
    user_message   = data.get("message", "").strip()
    dietary_pref   = data.get("dietary_pref", "")
    ingredient_sub = data.get("ingredient_sub", "")

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    try:
        result = orchestrate_agents(user_message, dietary_pref, ingredient_sub)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload-recipe-doc", methods=["POST"])
def upload_recipe_doc():
    """
    Upload and index a cookbook or recipe document into the RAG knowledge base.
    Supports PDF and TXT formats.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file     = request.files["file"]
    filename = file.filename or "recipe_document"

    if not (filename.lower().endswith(".txt") or filename.lower().endswith(".pdf")):
        return jsonify({"success": False, "error": "Only .txt and .pdf files are supported."}), 400

    try:
        text   = extract_text_from_file(file)
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"success": False, "error": "Could not extract text."}), 400

        RAG_DOCUMENTS.append({"title": filename, "chunks": chunks})

        return jsonify({
            "success" : True,
            "filename": filename,
            "chunks"  : len(chunks),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms SDK, credentials, and RAG status."""
    return jsonify({
        "status"           : "healthy",
        "app"              : "ChefGPT AI",
        "watsonx_available": WATSONX_AVAILABLE,
        "credentials_set"  : bool(WATSONX_API_KEY and WATSONX_PROJECT_ID),
        "rag_documents"    : len(RAG_DOCUMENTS),
        "pdf_support"      : PDF_SUPPORT,
        "timestamp"        : datetime.datetime.now().isoformat(),
    })


@app.route("/api/agents", methods=["GET"])
def agents_info():
    """Return metadata about all ChefGPT AI agents."""
    return jsonify({"agents": [
        {"id": "retrieval",  "name": "Recipe Retrieval Agent",       "icon": "SEARCH",    "purpose": "RAG Recipe Search"},
        {"id": "adaptation", "name": "Recipe Adaptation Agent",      "icon": "ADAPT",     "purpose": "Dietary Customisation"},
        {"id": "substitution","name": "Ingredient Substitution Agent","icon": "SWAP",      "purpose": "Ingredient Replacement"},
        {"id": "nutrition",  "name": "Nutrition Analysis Agent",     "icon": "NUTRITION", "purpose": "Nutritional Analysis"},
        {"id": "shopping",   "name": "Shopping List Agent",          "icon": "CART",      "purpose": "Smart Shopping Lists"},
        {"id": "orchestrator","name": "Master Orchestrator",         "icon": "BRAIN",     "purpose": "Agent Coordination"},
    ]})

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 65)
    print("  ChefGPT AI - Intelligent Recipe Generator & Cooking Assistant")
    print("=" * 65)
    print("  IBM watsonx.ai SDK  : " + ("Installed" if WATSONX_AVAILABLE else "Not installed (demo mode)"))
    print("  API Key             : " + ("Set" if WATSONX_API_KEY else "Missing (set in .env)"))
    print("  Project ID          : " + ("Set" if WATSONX_PROJECT_ID else "Missing (set in .env)"))
    print("  watsonx.ai URL      : " + WATSONX_URL)
    print("  PDF Support         : " + ("PyPDF2 available" if PDF_SUPPORT else "PyPDF2 not installed"))
    print("-" * 65)
    print("  Agents: Recipe Retrieval | Adaptation | Substitution")
    print("          Nutrition Analysis | Shopping List | Orchestrator")
    print("  Features: RAG System | Dietary Adaptation | Nutritional Analysis")
    print("-" * 65)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 65)
    app.run(debug=True, host="0.0.0.0", port=5000)
