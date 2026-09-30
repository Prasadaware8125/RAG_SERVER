"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 8 - Prompt Builder module to construct structured prompts for Gemini grounding.
Dependencies: utils.logger, utils.helper
"""

import sys
from typing import List, Dict, Any

# Package imports
from utils.logger import setup_logger
from utils.helper import (
    print_phase_header,
    print_loading,
    print_processing,
    print_success,
    print_failure,
    print_statistics,
    PhaseTimer
)

# Initialize logger
logger = setup_logger("phase8_prompt_builder")

class PromptBuilder:
    """
    Builds structured, grounded prompts containing system instructions,
    retrieved contexts (titles, URLs, and text content), and target constraints.
    """

    def __init__(self) -> None:
        """Initializes the prompt builder."""
        logger.debug("PromptBuilder initialized.")

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved text chunks with source title, URL, and an index.
        
        Args:
            retrieved_chunks (List[Dict[str, Any]]): List of matching chunks with metadata.
            
        Returns:
            str: Formatted context block.
        """
        if not retrieved_chunks:
            return "No relevant context found from trusted sources."
            
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            title = chunk.get("title", "Untitled Webpage")
            url = chunk.get("url", "No URL provided")
            text = chunk.get("text", "")
            
            part = (
                f"[Source {idx}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content:\n{text.strip()}\n"
            )
            context_parts.append(part)
            
        return "\n".join(context_parts)

    def build_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Constructs the complete grounded prompt injected with system instructions, context, chat history, and query.
        
        Args:
            query (str): The user topic/question.
            retrieved_chunks (List[Dict[str, Any]]): The retrieved background information chunks.
            chat_history (List[Dict[str, str]]): List of previous messages in the conversation.
            
        Returns:
            str: The fully populated prompt string.
        """
        logger.info(f"Building grounded prompt for query: '{query}' with {len(retrieved_chunks)} context sources.")
        
        # Format the retrieved sources into the context structure
        context_block = self.format_context(retrieved_chunks)
        
        history_block = ""
        if chat_history:
            history_parts = []
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_block = "Conversation History:\n" + "\n".join(history_parts) + "\n\n"
            
        # Build prompt sections with strict RAG and formatting rules
        prompt = (
            "System Instructions:\n"
            "You are an expert Engineering Education AI Assistant. Your task is to generate accurate, "
            "source-backed, clear educational notes and answers based ONLY on the provided retrieved web context. "
            "You can also refer to the conversation history to understand referred topics, but always ground your facts strictly in the Retrieved Context.\n\n"
            
            "Constraint Rules:\n"
            "1. ONLY answer using the facts directly mentioned in the retrieved context. Do NOT assume, "
            "extrapolate, or bring in external knowledge.\n"
            "2. If the context does not contain enough information to address the query, say EXACTLY: "
            "\"I couldn't find enough relevant web content to answer this question reliably.\"\n"
            "3. Never hallucinate or make up facts. Strict compliance is mandatory.\n"
            "4. Always cite your sources. Use bracketed numbers inline matching the Source index (e.g. [1], [2]).\n"
            "5. Structure the generated notes with the following sections where relevant context allows:\n"
            "   - **Definition**\n"
            "   - **Explanation**\n"
            "   - **Formula** (if applicable)\n"
            "   - **Example**\n"
            "   - **Applications**\n"
            "   - **Summary**\n"
            "   - **References** (A list of cited sources formatted as: [Title](URL))\n\n"
            
            "Retrieved Context:\n"
            f"{context_block}\n\n"
            
            f"{history_block}"
            
            "User Question:\n"
            f"{query}\n\n"
            
            "Answer:\n"
        )
        
        logger.debug("Prompt constructed successfully.")
        return prompt

    def build_fallback_prompt(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Constructs a prompt for when no search results could be retrieved from trusted engineering sources.
        Instructs the model to answer from general knowledge but prepend a warning.
        """
        logger.info(f"Building fallback prompt for query: '{query}'")
        
        history_block = ""
        if chat_history:
            history_parts = []
            for msg in chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_block = "Conversation History:\n" + "\n".join(history_parts) + "\n\n"
            
        prompt = (
            "System Instructions:\n"
            "You are an expert Engineering Education AI Assistant. Note that the system was UNABLE to retrieve "
            "any real-time web context from trusted sources for this query. Therefore, you must answer based on your "
            "general pre-trained engineering knowledge.\n\n"
            
            "CRITICAL REQUIREMENT:\n"
            "You MUST prepend a warning banner EXACTLY as follows at the very beginning of your response:\n"
            "**WARNING: No trusted real-time sources could be retrieved for this query. The following response is generated using general pre-trained knowledge and has not been grounded in trusted sources.**\n\n"
            
            f"{history_block}"
            
            "User Question:\n"
            f"{query}\n\n"
            
            "Answer:\n"
        )
        return prompt

def run_phase8(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Orchestrates the Phase 8 prompt construction.
    
    Args:
        query (str): User query.
        retrieved_chunks (List[Dict[str, Any]]): List of context chunks.
        
    Returns:
        str: Grounded prompt.
    """
    print_phase_header("Phase 8: Prompt Builder")
    print_loading("Assembling grounding guidelines...")
    
    builder = PromptBuilder()
    
    with PhaseTimer("Phase 8: Prompt Builder") as timer:
        print_processing("Formatting context chunks and prompt injections...")
        full_prompt = builder.build_prompt(query, retrieved_chunks)
        
    # Statistics
    stats = {
        "User Query": query,
        "Retrieved Sources": len(retrieved_chunks),
        "Prompt Length (chars)": len(full_prompt),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Print the built prompt to CLI for verification
    print("--- GENERATED GROUNDED PROMPT ---")
    print(full_prompt)
    print("-" * 60 + "\n")
    
    return full_prompt

if __name__ == "__main__":
    # Test execution using mock retrieved chunks
    test_query = "Explain Deadlock conditions"
    mock_retrieved_chunks = [
        {
            "id": "https://wikipedia.org/wiki/Deadlock#chunk-2",
            "text": (
                "The four necessary conditions for deadlock are: Mutual Exclusion, Hold and Wait, "
                "No Preemption, and Circular Wait. If any one of these conditions is prevented, "
                "deadlock is avoided."
            ),
            "metadata": {"url": "https://wikipedia.org/wiki/Deadlock", "title": "Deadlock - Wikipedia", "chunk_number": 2},
            "url": "https://wikipedia.org/wiki/Deadlock",
            "title": "Deadlock - Wikipedia"
        }
    ]
    
    try:
        run_phase8(test_query, mock_retrieved_chunks)
    except Exception as exc:
        print_failure(f"Phase 8 execution failed: {exc}")
        sys.exit(1)
