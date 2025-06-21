# PsAIch Colleague

**PsAIch Colleague** is a modular Python application that provides psychotherapists with an "AI sparring chat partner" for supporting diagnosis and treatment planning. In this project I aim to learn about OOP, LLMs, modular code design, and good ML engineering practices along with git copilot.

## Features

- **Streamlit UI:** Chat interface, history saving/loading (txt/csv), base prompts/contexts, reproducibility via seed, etc.
- **Communicator Module:** Handles user input, LLM output, and context management/warnings.
- **Model Abstraction:** Easily swap or extend the LLM backend (e.g., for RAG or custom knowledge bases).

## Project Structure

```
psaich_colleague/
├── app/
│   ├── interface/         # Streamlit UI
│   ├── communicator/      # Communication logic
│   └── models/            # LLM and future RAG logic
├── tests/                 # Unit/integration tests
├── data/                  # Example histories, prompts, etc.
├── scripts/               # Helper scripts
├── .streamlit/             # Streamlit config
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

1. **Clone the repo**
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **(Optional) Set up a virtual environment**
    ```bash
    python -m venv .venv
    ```
4. **Run the Streamlit app**
    ```bash
    streamlit run app/interface/main.py
    ```

## Tested LLM Backends
[Ollama](https://ollama.com/) (pulled locally via "ollama pull ...")
- mistral (7b)
- llama3:8b
- mixtral 
- llama3:70b (unstable)

## Contributing

Contributions are welcome! Please open issues or pull requests as you enhance the project.

## License

[MIT](LICENSE)

---
*Created by Gerrit Hüller as a learning project.*