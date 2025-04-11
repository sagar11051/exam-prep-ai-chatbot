def generate_prompt(question, context_sections):
    """
    Generate a custom prompt by combining the user's question with the retrieved context sections.
    """
    prompt = (
        "You are an expert engineering tutor. "
        "Based on the following context, answer the question:\n\n"
    )
    for section in context_sections:
        prompt += f"{section}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

