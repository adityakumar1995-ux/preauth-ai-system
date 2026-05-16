import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("=" * 60)
print("Gemini API Test")
print("=" * 60)

print("API key loaded:", bool(api_key))

if api_key:
    print("API key starts with:", api_key[:6])
else:
    print("ERROR: GEMINI_API_KEY was not found.")
    print("Check that .env exists in the same folder as this file.")
    print("Expected .env format:")
    print("GEMINI_API_KEY=your_actual_key_here")
    raise SystemExit


try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents='Return only this JSON: {"status": "working"}'
    )

    print("\nGemini response:")
    print(response.text)

    print("\nGemini test completed successfully.")

except Exception as e:
    print("\nGemini test failed.")
    print("Error:")
    print(e)