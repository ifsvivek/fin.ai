import os
import openai
import dotenv

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)


def chatGPT(text):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": text}],
        max_tokens=4000,
        temperature=0.6
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    user_input = input("Enter your query: ")
    chatGPT(user_input)