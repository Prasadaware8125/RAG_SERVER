"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Phase 9 - Gemini Client module to query Gemini 2.5 Flash with grounded prompts.
Dependencies: google-genai, config.config, utils.logger, utils.helper
"""

import sys
from typing import Dict, Any, Tuple
from groq import Groq

# Package imports
from config.config import GROQ_API_KEY, GROQ_MODEL_NAME, validate_config
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
logger = setup_logger("phase9_gemini_client")

class GeminiContentGenerator:
    """
    Interfaces with the Groq API to generate text notes.
    Applies strict settings including 0.2 temperature and 4096 token limits.
    """

    def __init__(self, api_key: str = GROQ_API_KEY, model_name: str = GROQ_MODEL_NAME) -> None:
        """
        Initializes the Groq Client.
        
        Args:
            api_key (str): The Groq API key.
            model_name (str): Groq model name.
        """
        if not api_key:
            raise ValueError("Groq API key is missing. Ensure GROQ_API_KEY is defined in your environment or .env file.")
            
        logger.debug("Initializing Groq Client...")
        try:
            self.client = Groq(api_key=api_key)
            self.model_name = model_name
            logger.info(f"Groq client successfully initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise RuntimeError(f"Groq client initialization failed: {e}") from e

    def generate_response_with_tokens(self, prompt: str) -> tuple[str, Dict[str, int]]:
        """
        Submits a grounded prompt to LLM and returns the text output plus token usage dict.
        """
        logger.info(f"Sending generation request to Groq model: {self.model_name}.")
        print_processing(f"Submitting grounded query to Groq ({self.model_name}) API...")
        
        import time
        retries = 5
        backoff = 6.0
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                
                answer_text = response.choices[0].message.content
                if not answer_text:
                    raise ValueError("Groq API returned an empty response.")
                    
                usage = getattr(response, "usage", None)
                if usage:
                    token_dict = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0)
                    }
                else:
                    token_dict = {
                        "input_tokens": len(prompt) // 4,
                        "output_tokens": len(answer_text) // 4,
                        "total_tokens": (len(prompt) + len(answer_text)) // 4
                    }
                    
                logger.info("Successfully received response from Groq API.")
                return answer_text, token_dict
            except Exception as e:
                # Catch rate limits / 429
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < retries - 1:
                        logger.warning(f"Rate limit hit during generation. Retrying in {backoff:.1f}s... (Attempt {attempt+1}/{retries})")
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                logger.error(f"Groq generation request failed: {e}")
                raise RuntimeError(f"Groq API invocation failed: {e}") from e

    def generate_response(self, prompt: str) -> str:
        """
        Submits a grounded prompt to LLM and returns the text output.
        """
        answer_text, _ = self.generate_response_with_tokens(prompt)
        return answer_text

def run_phase9(prompt: str) -> str:
    """
    Orchestrates the Phase 9 Gemini call.
    
    Args:
        prompt (str): Full prompt to submit.
        
    Returns:
        str: Gemini answer.
    """
    print_phase_header("Phase 9: Gemini Content Generation")
    print_loading("Connecting to Gemini API...")
    
    try:
        # Validate base config
        validate_config()
    except ValueError as val_err:
        print_failure(f"Configuration validation failed: {val_err}")
        logger.error(f"Configuration validation failed: {val_err}")
        sys.exit(1)
        
    generator = GeminiContentGenerator()
    
    with PhaseTimer("Phase 9: Gemini Content Generation") as timer:
        answer = generator.generate_response(prompt)
        
    # Analyze statistics
    stats = {
        "Target Model": generator.model_name,
        "Prompt Size (chars)": len(prompt),
        "Response Size (chars)": len(answer),
        "Execution Status": "Success",
        "Duration": f"{timer.elapsed_time:.3f}s"
    }
    print_statistics(stats)
    
    # Print a preview of the response
    print("--- GEMINI RESPONSE PREVIEW ---")
    print(answer)
    print("-" * 60 + "\n")
    
    return answer

if __name__ == "__main__":
    # Test execution using a standard dummy test prompt
    test_prompt = (
        "System Instructions:\n"
        "You are an expert Engineering Education AI Assistant. Answer based ONLY on context.\n\n"
        "Retrieved Context:\n"
        "[Source 1]\n"
        "Title: CPU Scheduling\n"
        "URL: https://wikipedia.org/wiki/CPU_scheduling\n"
        "Content:\n"
        "CPU scheduling is the process by which a process is allocated the CPU for execution. "
        "First-Come, First-Served (FCFS) is the simplest algorithm, executing jobs in arrival order.\n\n"
        "User Question:\n"
        "What is FCFS scheduling?\n\n"
        "Answer:\n"
    )
    
    try:
        run_phase9(test_prompt)
    except Exception as exc:
        print_failure(f"Phase 9 execution failed: {exc}")
        sys.exit(1)
