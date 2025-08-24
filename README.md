# MMA Pulse

## Project Overview
Sports Pulse is an AI-powered sports analytics project designed to process and analyze sports data using modern machine learning and natural language processing tools. The project aims to deliver actionable insights and predictions to sports enthusiasts, analysts, and organizations.

## Features

- Uses [LangChain](https://github.com/hwchase17/langchain) for chain-based AI workflows
- Integrates [OpenAI](https://openai.com/) models
- Implements vector search with [FAISS](https://github.com/facebookresearch/faiss)
- Uses [Transformers](https://github.com/huggingface/transformers) for NLP
- Supports progress tracking with `tqdm`
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
- [LangChain](https://github.com/hwchase17/langchain) – for building AI chains
- [OpenAI](https://openai.com/) – large language models
- [FAISS](https://github.com/facebookresearch/faiss) – fast similarity search
- [Transformers](https://github.com/huggingface/transformers) – NLP models
- [TQDM](https://github.com/tqdm/tqdm) – progress bars
- [python-dotenv](https://github.com/theskumar/python-dotenv) – environment management
- [gradio](https://www.gradio.app/) – User Interface

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
