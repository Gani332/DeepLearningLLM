# Gordon Ramsay AI Cooking Assistant - Project Plan

## Executive Summary

This project develops a Gordon Ramsay-inspired cooking assistant using a **hybrid approach**:
- **Fine-tuning**: For conversational ability and Gordon Ramsay's personality
- **RAG (Retrieval-Augmented Generation)**: For factual recipe knowledge

## Why This Approach?

### Fine-tuning for Personality + RAG for Recipes

**Advantages of RAG for Recipes:**
- ✅ No hallucination of recipes - retrieves actual, tested recipes
- ✅ Easy to update - add new recipes without retraining
- ✅ Scalable - handles millions of recipes efficiently
- ✅ Factual accuracy - preserves exact measurements
- ✅ Saves compute - smaller fine-tuning dataset
- ✅ Better performance - fine-tuning focuses on style, not facts

**What to Fine-tune:**
- Conversational ability
- Gordon Ramsay personality and speaking style

**What to Use RAG For:**
- Recipe database (Recipe NLG + Food.com datasets)
- Factual cooking information

## System Architecture

```
User Query
    ↓
┌─────────────────────────┐
│  Fine-tuned LLM         │
│  (Conversational +      │
│   Gordon Ramsay Style)  │
└─────────────────────────┘
    ↓
    ├──→ Decides if recipe needed
    ↓
┌─────────────────────────┐
│  RAG System             │
│  - Vector DB (recipes)  │
│  - Retrieval mechanism  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Response Generation    │
│  (LLM + Retrieved Data) │
└─────────────────────────┘
```

## Phase 1: Setup & Requirements

### 1.1 Environment Setup
- Python environment with: transformers, datasets, accelerate, peft
- Vector database: ChromaDB / Pinecone / FAISS
- Embeddings: sentence-transformers
- Framework: LangChain / LlamaIndex

### 1.2 Base Model Selection
- **Option 1**: GPT-2 Medium/Large (easier, faster, lower compute)
- **Option 2**: LLaMA-2-7B (better quality, needs more resources)
- **Option 3**: Mistral-7B (best quality-to-size ratio)

### 1.3 Success Metrics
- **Fine-tuned Model**: Perplexity, personality consistency, conversational coherence
- **RAG System**: Retrieval accuracy (Recall@K, MRR), recipe relevance
- **Complete System**: Recipe accuracy, personality authenticity, helpfulness ratings

## Phase 2: Dataset Preparation

### 2.1 Fine-tuning Datasets (Smaller & Focused)

**Dataset 1: Conversational Base (~30-50k examples)**
- OpenAssistant Conversations (OASST1)
- Alpaca dataset
- ShareGPT conversations
- Focus: General Q&A and instruction following

**Dataset 2: Gordon Ramsay Personality (~10-20k examples)**
- Hell's Kitchen, Kitchen Nightmares, MasterChef transcripts
- YouTube subtitle extractions
- Interview transcripts
- Cookbook introductions/commentary
- Format as instruction-response pairs

**Example Training Format:**
```json
{
  "instruction": "User's pasta is mushy",
  "input": "I cooked my pasta for 15 minutes",
  "output": "Fifteen minutes?! That's not pasta, that's baby food! You've murdered it! Pasta should be al dente - firm to the bite. Check the package, usually 8-10 minutes MAX. Taste it a minute before the time's up. This isn't rocket science!"
}
```

### 2.2 RAG Knowledge Base (No Fine-tuning)

**Recipe Databases:**
- Recipe NLG: 2.2M recipes
- Food.com: 500k+ recipes with ratings and reviews

**Processing Pipeline:**
1. Extract: title, ingredients, instructions, cuisine, difficulty, time
2. Create rich text representation
3. Generate embeddings
4. Store in vector database with metadata

**Metadata for Filtering:**
- Cuisine type
- Difficulty level
- Cooking time
- Dietary restrictions
- User ratings

## Phase 3: Model Variants for Comparison

### Model A: Baseline
- Base model fine-tuned on conversational data only
- No personality, no RAG
- Pure instruction-following

### Model B: Personality Only
- Conversational + Gordon Ramsay personality
- No RAG system
- Tests personality without recipe knowledge

### Model C: Conversational + RAG
- Conversational fine-tuning
- RAG for recipes
- No personality (professional chef assistant)

### Model D: Full System (Target)
- Conversational + Gordon Ramsay personality fine-tuning
- RAG for recipes
- Complete Gordon Ramsay cooking assistant

## Phase 4: RAG System Implementation

### 4.1 Vector Database Setup

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load and process recipes
recipes = load_recipes(['recipe_nlg', 'food_com'])

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Build vector store
vector_db = Chroma.from_documents(
    documents=recipes,
    embedding=embeddings,
    metadata_fields=['cuisine', 'difficulty', 'time', 'rating']
)
```

### 4.2 Retrieval Strategy

```python
# Retrieve top-k relevant recipes
retrieved_recipes = vector_db.similarity_search(
    query=user_query,
    k=3,
    filter={'difficulty': 'easy'}  # Optional filtering
)
```

### 4.3 Recipe Representation

```python
recipe_text = f"""
Recipe: {title}
Cuisine: {cuisine}
Difficulty: {difficulty}
Time: {total_time}
Rating: {rating}/5 ({num_reviews} reviews)

Ingredients:
{formatted_ingredients}

Instructions:
{step_by_step_instructions}

Tips: {user_tips}
"""
```

## Phase 5: Model Training

### 5.1 Fine-tuning Configuration

**Training Parameters:**
- LoRA rank: 8-16
- Learning rate: 2e-4
- Epochs: 3-5
- Batch size: 4-8 (with gradient accumulation)
- Gradient accumulation steps: 4
- Max sequence length: 512 tokens
- Warmup steps: 500-1000

**Training Pipeline:**
```
For each model variant (A, B, C, D):
1. Load base model
2. Prepare dataset-specific prompts
3. Apply LoRA/QLoRA for parameter-efficient fine-tuning
4. Save checkpoints at each epoch
5. Evaluate on validation set
```

### 5.2 Response Generation Pipeline

```python
def generate_response(user_query, model, rag_system):
    # 1. Classify query type
    query_type = classify_query(user_query)
    
    # 2. Retrieve recipes if needed
    if requires_recipe(query_type):
        recipes = rag_system.retrieve(user_query, k=3)
        context = format_recipes(recipes)
    else:
        context = None
    
    # 3. Generate with personality
    prompt = build_gordon_ramsay_prompt(user_query, context)
    response = model.generate(prompt)
    
    return response
```

## Phase 6: Evaluation & Benchmarking

### 6.1 Automated Metrics

**For Fine-tuned Model:**
- Perplexity on validation set
- Personality consistency score (custom rubric)
- Conversational coherence
- Instruction following accuracy

**For RAG System:**
- Retrieval accuracy (Recall@K, MRR)
- Recipe relevance (manual evaluation)
- Response time/latency

**For Complete System:**
- Recipe accuracy (are instructions correct?)
- Personality authenticity (sounds like Gordon?)
- Helpfulness ratings
- Hallucination rate

### 6.2 Comparative Analysis

| Metric | Model A | Model B | Model C | Model D |
|--------|---------|---------|---------|---------|
| Conversational Quality | ✓ | ✓ | ✓ | ✓ |
| Personality Strength | ✗ | ✓✓ | ✗ | ✓✓ |
| Recipe Accuracy | ✗ | ✗ | ✓✓ | ✓✓ |
| Hallucination Rate | High | High | Low | Low |
| Overall Helpfulness | Low | Med | High | Highest |

### 6.3 Key Comparisons

- **Model A vs B**: Impact of personality fine-tuning
- **Model B vs D**: Value of RAG for factual accuracy
- **Model C vs D**: Personality doesn't hurt recipe quality
- **Models without RAG vs with RAG**: Hallucination reduction

## Phase 7: Implementation Timeline

### Week 1: Setup + Data Collection
- Set up fine-tuning environment
- Collect conversational + Gordon Ramsay data
- Download recipe datasets (Recipe NLG + Food.com)

### Week 2: Data Preparation
- Process fine-tuning datasets (conversational + personality)
- Create instruction-response pairs
- Build vector database from recipes

### Week 3-4: RAG System
- Implement vector database
- Test retrieval quality
- Optimize embedding + retrieval strategy

### Week 5-6: Model Training
- Train Model A (baseline)
- Train Model B (+ personality)
- Train Model C & D (with RAG integration)

### Week 7: Integration & Testing
- Connect fine-tuned models with RAG
- End-to-end testing
- Fix bugs and optimize

### Week 8: Evaluation
- Run all benchmarks
- Human evaluation study
- Collect comparison metrics

### Week 9-10: Analysis & Documentation
- Statistical analysis
- Write comprehensive report
- Create visualizations and charts

## Phase 8: Report Structure

1. **Introduction & Motivation**
   - Problem statement
   - Objectives and contributions

2. **Related Work**
   - Fine-tuning approaches
   - RAG systems
   - Personality-driven chatbots

3. **Dataset Description & Preparation**
   - Data sources
   - Preprocessing pipeline
   - Dataset statistics

4. **Model Architecture & Training Methodology**
   - Base model selection
   - Fine-tuning strategy
   - RAG implementation

5. **Experimental Results**
   - Quantitative metrics
   - Qualitative analysis
   - Sample outputs

6. **Comparative Analysis**
   - Model comparison tables
   - Ablation studies
   - Discussion of findings

7. **Conclusion & Future Work**
   - Summary of contributions
   - Limitations
   - Future improvements

## Extra Credit Opportunities

### 1. Ablation Studies
- Different retrieval strategies (similarity vs. MMR)
- Varying number of retrieved recipes (k=1, 3, 5)
- Different embedding models comparison
- Dataset mixing ratios

### 2. Advanced Features
- Multi-turn conversation with context memory
- User preference learning
- Dietary restriction filtering
- Recipe customization and substitutions

### 3. Evaluation Innovations
- Human preference study (Gordon vs. non-Gordon)
- Recipe safety validation
- A/B testing with real users
- Blind evaluation studies

## Technical Tips

### For Gordon Ramsay Personality
- Use system prompts defining his character
- Include characteristic phrases in training data
- Balance authenticity with helpfulness
- Don't overdo profanity - keep it PG-13

### For Recipe Integration
- Format recipes consistently (ingredients → steps → tips)
- Include context (difficulty, time, servings)
- Add Gordon's commentary to recipe instructions
- Validate recipe safety and accuracy

### For Best Results
- Start with smaller models and scale up
- Monitor training loss carefully
- Use validation set to prevent overfitting
- Test with diverse queries during development
- Implement proper error handling in RAG retrieval

## Academic Contribution

**Key Insights:**
- Hybrid approach combining personality fine-tuning with factual RAG
- Separating style (fine-tuned) from facts (RAG) improves accuracy
- RAG enables updatable knowledge without retraining
- Personality fine-tuning doesn't require large domain-specific datasets

**Novel Aspects:**
- Personality-driven cooking assistant with factual grounding
- Comparison of different architectural choices
- Demonstration of RAG's advantages over pure fine-tuning for factual domains

## Success Criteria

- ✅ Conversational ability comparable to baseline models
- ✅ Recognizable Gordon Ramsay personality in responses
- ✅ Accurate recipe information with low hallucination
- ✅ Clear performance differences between model variants
- ✅ Comprehensive evaluation and analysis

---

**Project Goal**: Create a helpful, entertaining, and accurate cooking assistant that captures Gordon Ramsay's passionate teaching style while providing reliable culinary information through retrieval-augmented generation.