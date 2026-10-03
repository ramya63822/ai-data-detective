import ollama

MODEL = "qwen3:4b"


def ask_llm(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_ctx": 2048,
            "num_predict": 300
        },
        think=False
    )

    return response["message"]["content"]