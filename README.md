# MMA Pulse

## Project Overview
Sports Pulse is an AI-powered sports analytics project designed to process and analyze sports data using modern machine learning and natural language processing tools. The project aims to deliver actionable insights and predictions to sports enthusiasts, analysts, and organizations.

## Features
- Cleaned and structured fight data from multiple sources
- Retrieval-Augmented Generation (RAG) powered chatbot
- Integrates [OpenAI](https://openai.com/) models
- Semantic search using [ChromaDB]([https://openai.com/](https://www.trychroma.com/)) and Sentence Transformers
- Uses [Transformers](https://github.com/huggingface/transformers) for NLP
- Supports voice input and transcription 
- Manages environment variables with `python-dotenv`

## Prerequisites

- Python 3.8+
- (Recommended) Create and activate a virtual environment

## Set up and Run Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/amulsapkota1/sports-pulse.git
   cd sports-pulse
   ```
   
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   - If a `requirements.txt` file exists:
     ```bash
     pip install -r requirements.txt
     ```
   - If not, install dependencies manually:
     ```bash
     pip install langchain openai faiss-cpu transformers tqdm python-dotenv
     ```
4. **Set up environment variables**
   - Create a `.env` file in the root directory with your OpenAI API key and other required configurations:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```
5. **Run the project**
   - Launch the main script
     ```bash
     cd sports-pulse/src 
     python main.py
     ```

## Set up and Run Instructions for Google Colab
1. **Open the Notebook**
   - Launch the project notebook: [SPORTPULSE.ipynb](https://colab.research.google.com/drive/14gHDT1qt1tC2KEriZP49omtyKScRYsvj?usp=sharing)
2. **File Storage in Google Drive**
   - Make sure master_data-rabindra-dhant.txt is placed in the root of your Google Drive.
3. **Add Your OpenAI API Key**
   - Store OpenAI API Key in the secrets tab (🔑) of Google Colab
4. **Run All Cells**
5. **Interact with the UI generated at the end**

## Project Structure

```
sports-pulse/
├── src/main.py
├── requirements.txt
├── README.md
└── ...
```

## Dependencies and Tools Used

- **Python 3.8+**
- Pandas – data manipulation and analysis
- NumPy – numerical computing
- Matplotlib & Seaborn – data visualization
- ChromaDB – vector database for semantic search
- Sentence Transformers – text embeddings for retrieval
- OpenAI API – GPT-based language generation
- Gradio – interactive chatbot UI
- Whisper – speech-to-text transcription
- NetworkX – mind map and graph visualization
- orjson – fast JSON parsing
- XGBoost – optional ML model support
- Google Colab – cloud-based development environment

## Team Members and Roles

_Team members:_

- **Amul Sapkota** – AI Engineer/Presenter
- **Bikash Mainali** – Developer/Designer, Documentation
- **Om Nepal** – AI Engineer

---

## License

MIT License

Copyright (c) 2025 NepHacks

Permission is hereby granted, free of charge, to any person obtaining a copy

---
