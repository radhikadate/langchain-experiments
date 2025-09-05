# LangChain AI Experiments

A comprehensive Python project exploring LangChain capabilities with Anthropic's Claude, featuring quote generation, translation, routing, and parsing functionalities.

## Project Structure

### 🎭 Quote Generator
Interactive funny quote generation with translation capabilities.

**Files:**
- `sequential_chains.py` - Command-line quote generator with user input
- `streamlit_funny_quote_generator.py` - Web interface for quote generation and translation

**Features:**
- Generate funny quotes based on topics and characters
- Translate quotes to different languages
- Sequential chain processing (quote → translation)
- Beautiful formatted output

**Usage:**
```bash
cd quote_generator
python sequential_chains.py
# or
streamlit run streamlit_funny_quote_generator.py
```

### 🎨 Joke Caricature Chain
Advanced sequential chain that generates jokes and creates detailed caricature descriptions.

**Files:**
- `sequential_chains.py` - Command-line joke and caricature generator
- `streamlit_joke_caricature_app.py` - Web interface for joke and caricature generation

**Features:**
- Generate funny jokes based on topics and characters
- Create detailed caricature descriptions based on the jokes
- Sequential chain processing (joke → caricature description)
- Visual descriptions suitable for artists

**Usage:**
```bash
cd joke_caricature_chain
python sequential_chains.py
# or
streamlit run streamlit_joke_caricature_app.py
```

### 🔧 Miscellaneous Experiments (`misc/`)

#### `sequential_translation_chain.py`
Demonstrates LangChain's Runnable interface for chained operations.
- Translates text to specified language
- Finds alternative phrasings of the translation
- Uses modern LangChain syntax (`prompt | model | parser`)

#### `prompt_templating_examples.py`
Examples of LangChain prompt templating patterns.
- Code generation requests with structured messaging
- Translation using parameterized templates
- Demonstrates SystemMessage and HumanMessage usage

#### `domain_specific_router_chain.py`
Intelligent routing system that directs questions to specialized expert chains.
- **Experts:** Biology, Math, Astronomy, Travel Agent
- Automatically routes questions to appropriate domain expert
- Fallback to default chain for unmatched topics
- Uses MultiPromptChain for intelligent routing

#### `structured_itinerary_parser.py`
Converts unstructured travel text into structured data.
- Parses travel itineraries into structured format
- Extracts: leave time, departure location, places to visit, restaurants
- Uses LangChain's StructuredOutputParser
- Multi-line input support with beautiful formatted output

#### `basic_translator.py`
Simple translation utility using LangChain.
- Basic text translation functionality
- Demonstrates fundamental LangChain usage

#### `agents_simple.py`
Introduction to LangChain agents and tool usage.
- Agent-based interactions
- Tool integration examples
- **Note:** Currently non-functional - requires debugging and additional dependencies

## Setup

### Prerequisites
- Python 3.8+
- Anthropic API key

### Installation
1. **Install dependencies:**
   ```bash
   pip install langchain langchain-anthropic streamlit python-dotenv
   ```

2. **Create `.env` file:**
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

### Running Applications

**Command Line Tools:**
```bash
python misc/sequential_translation_chain.py
python misc/domain_specific_router_chain.py
python misc/structured_itinerary_parser.py
```

**Web Applications:**
```bash
streamlit run quote_generator/streamlit_funny_quote_generator.py
streamlit run joke_caricature_chain/streamlit_joke_caricature_app.py
```

## Key Features

- **Sequential Chains:** Multi-step processing pipelines
- **Router Chains:** Intelligent question routing to domain experts
- **Structured Parsing:** Convert unstructured text to structured data
- **Prompt Templates:** Reusable, parameterized prompts
- **Web Interfaces:** Interactive Streamlit applications
- **Modern LangChain:** Uses latest Runnable interface patterns

## Technologies Used

- **LangChain:** Framework for LLM applications
- **Anthropic Claude:** AI model for text generation
- **Streamlit:** Web application framework
- **Python:** Core programming language

## Project Goals

This project demonstrates various LangChain patterns and capabilities:
- Chain composition and sequential processing
- Intelligent routing and decision making
- Structured data extraction from unstructured text
- Interactive web interfaces for AI applications
- Modern LangChain development patterns