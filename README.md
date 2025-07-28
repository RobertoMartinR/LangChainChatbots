# LangChain Chatbots

This repository contains various examples and implementations of chatbots using LangChain, demonstrating different techniques and approaches for building conversational AI applications. Everything is based in LangChain official documentation and the purpose is to practice the different methods to create a chatbot and learn the tools. Official documentation reference is at the beggining of each document. 

## 📁 Project Structure

- `1_langchain_bot_using_messages.py` - Basic chatbot implementation using message-based approach
- `2_chatbot_using_prompt_templates.py` - Chatbot using prompt templates for better control
- `3_semantic_search_with_langchain.py` - Semantic search implementation with document retrieval
- `4_tagging.py` - Text classification and tagging using structured output
- `5_Extraction_chain.py` - Information extraction from text using Pydantic models
- `nke-10k-2023.pdf` - Sample PDF document for testing semantic search

## Features

### 1. Message-Based Chatbot
Simple chatbot implementation using LangChain's message system with Google Gemini model.

### 2. Prompt Templates
Advanced chatbot using structured prompt templates for language translation and other tasks.

### 3. Semantic Search
Implements document loading, text splitting, embeddings, and vector-based semantic search using:
- PDF document processing
- Text chunking with overlap
- Google Generative AI embeddings
- Chroma vector store
- Multiple search methods (similarity, async, with scores, by vector)

### 4. Text Classification/Tagging
Text classification system that can identify:
- Sentiment (positive, negative, neutral)
- Aggressiveness level (1-10 scale)
- Language detection
- Support for both basic and detailed classification schemas

### 5. Information Extraction
Structured information extraction from text using Pydantic models to define extraction schemas.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Google API key for Gemini models (Not needed if you use another provider or  local models)
- OpenAI API key (for structured output features)
- LangSmith API key (optional, for tracing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RobertoMartinR/LangChainChatbots.git
cd LangChainChatbots
```

2. Install required dependencies:
```bash
pip install langchain langchain-google-genai langchain-openai langchain-chroma langchain-community python-dotenv pydantic pypdf
```

3. Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=your_project_name
```

## 📖 Usage

### Basic Message Chatbot
```python
python 1_langchain_bot_using_messages.py
```

### Prompt Template Chatbot
```python
python 2_chatbot_using_prompt_templates.py
```

### Semantic Search
```python
python 3_semantic_search_with_langchain.py
```

### Text Classification
```python
python 4_tagging.py
```

### Information Extraction
```python
python 5_Extraction_chain.py
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for Google Gemini models
- `OPENAI_API_KEY`: Required for structured output features
- `LANGSMITH_API_KEY`: Optional, for debugging and tracing
- `LANGSMITH_PROJECT`: Optional, project name for LangSmith
- `LANGSMITH_TRACING`: Set to "true" to enable tracing

### Model Configuration
The examples use different models depending on the functionality:
- **Google Gemini 2.0 Flash**: For general chat and basic operations
- **OpenAI GPT-4o-mini**: For structured output operations (required for `.with_structured_output()`)

## 📚 Key Concepts Demonstrated

### 1. LangChain Fundamentals
- Message-based conversations
- Prompt templates and formatting
- Model initialization and configuration

### 2. Document Processing
- PDF loading and parsing
- Text splitting and chunking
- Embedding generation

### 3. Vector Operations
- Vector store creation and management
- Similarity search techniques
- Retrieval-based question answering

### 4. Structured Output
- Pydantic model definitions
- Schema-based information extraction
- Classification with controlled outputs

### 5. Integration Patterns
- Environment variable management
- Error handling
- Async operations

## ⚠️ Important Notes

- **Structured Output Limitation**: The `.with_structured_output()` method only works with OpenAI models, not with Google Gemini models
- **API Keys**: Make sure to keep your API keys secure and never commit them to version control
- **Rate Limits**: Be aware of API rate limits when running multiple examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API](https://platform.openai.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 📞 Support

If you have questions or need help, please:
1. Check the official LangChain documentation
2. Search existing issues in this repository
3. Create a new issue with detailed information about your problem

---

**Happy coding! 🚀**
