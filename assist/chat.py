# assist/chat.py

def chat(question: str) -> str:
    """
    Chat plugin stub: echoes back what you asked.
    """
    if not question.strip():
        return "Please enter a question above."
    return f"Chat plugin stub received: “{question}”"
