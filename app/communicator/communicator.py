import os
import requests
import json
from typing import List, Dict, Optional
from app.models.base_model import BaseModel
from app.rag.pdf_rag import SimplePDFRAG 


class Communicator:
    """
    Handles the interaction between user input, the LLM backend, and the conversation context.
    Sends user prompts to the LLM (e.g., via Ollama), collects responses, and manages history.
    Warns when the context window might be exceeded.
    """

    def __init__(self, 
                 model_url: str = "http://localhost:11434/api/chat", 
                 model_name: str = "mistral",
                 temperature: float = 0.7,
                 enable_pdf_rag: bool = True,  # Enable PDF RAG by default
                 max_context_length: int = 4096, # 4096 for Llama 3, 2048 for Mistral
                 max_summary_length_percentage: float = 0.75):
        self.model_url = model_url
        self.model_name = model_name
        self.temperature = temperature
        self.model = BaseModel(model_name=model_name, temperature=temperature, ollama_url=model_url)
        self.max_context_length = max_context_length
        self.history: List[Dict[str, str]] = []  # Each message: {"role": "user"/"assistant", "content": ...}
        self.summary: Optional[str] = None  # Summary of conversation history
        self.max_summary_length = int(max_context_length * max_summary_length_percentage)
        self.enable_pdf_rag = enable_pdf_rag
        self.pdf_rag = SimplePDFRAG(pdf_path="data/knowledge_folder") if enable_pdf_rag else None 

    def add_message(self, role: str, content: str):
        """Adds a message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def get_context(self) -> str:  
        """
        Returns the current conversation context as a string.
        Warns if the context length exceeds the limit, then summarizes the history within 75% of the max length.
        """
        context = "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)
        if len(context) < self.max_context_length:
            return context
        else:
            warning = f"Warning: Context limit ({self.max_context_length} chars) reached. Summarizing conversation history.\n"
            summary = self.summarize_history()
            self.summary = summary  # Store summary for later
            context_with_summary = warning + "Summary of previous conversation:\n" + summary
            return context_with_summary

    def summarize_history(self) -> str:
        """
        Summarizes the existing conversation history with a very short, precise summary.
        The summary is returned in bold markdown for user review/intervention.
        The summarization prompt/response is NOT added to the conversation history.
        """
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        max_summary_length = min(500, self.max_summary_length)  # Aim for 500 chars or less

        summary_prompt = (
            f"Summarize the following conversation as briefly and precisely as possible, "
            f"highlighting only the most important facts and context. "
            f"Keep the summary under {max_summary_length} characters. "
            f"Be concise and accurate:\n{history_text}"
        )
        # Call the LLM for summarization WITHOUT adding to history
        # Temporarily store current history, clear it, call LLM, then restore
        original_history = self.history
        self.history = []
        summary = self.model.generate_response(
            prompt=summary_prompt,
            system_prompt="You summarize conversations for context pruning. Be concise and precise.",
            history=[]
        )
        self.history = original_history

        # Return summary in bold markdown
        return f"**{summary.strip()}**"
    
    def get_augmented_prompt(self, user_prompt: str) -> str:
        if self.pdf_rag:
            contexts = self.pdf_rag.retrieve(user_prompt)
            if contexts:
                context_text = "\n\n".join(contexts)
                return f"Use the following information from local PDFs as context:\n{context_text}\n\nUser question: {user_prompt}"
        return user_prompt

    def send_to_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send the prompt (and history) to the LLM backend and return the response.
        This uses the Ollama API (http://localhost:11434).
        """
        # Update model parameters in case they changed
        self.model.model_name = self.model_name
        self.model.temperature = self.temperature
        self.model.model_url = self.model_url

        if self.enable_pdf_rag:
            # Augment prompt with PDF RAG context if enabled
            my_prompt = self.get_augmented_prompt(prompt)
        else:
            my_prompt = prompt

        answer = self.model.generate_response(
            prompt=my_prompt,
            system_prompt=system_prompt,
            history=self.history
        )

        # Add latest user and assistant message to history
        self.add_message("user", prompt)
        self.add_message("assistant", answer)
        return answer

    def reset_session(self):
        self.history = []
        self.summary = None

    def export_session(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)