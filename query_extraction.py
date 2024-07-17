from process_output import process_llm_response
import re
import ast
import subprocess
import json

with open('metadata.json') as f:
    d = json.load(f)

def generate_md(Question, query):
    prompt = f"{Question}{query}"
    print(f"Prompt: {prompt}")  # Debugging statement
    data = {
        "model": "phi3",
        "temperature": 0.4,
        "n": 1,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False
    }

    result = subprocess.run(
        ['curl', '-X', 'POST', 'http://localhost:11434/v1/chat/completions',
         '-H', 'Content-Type: application/json', '-H', 'Authorization: Bearer nokeyneeded',
         '-d', json.dumps(data)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running curl: {result.stderr}")  # Debugging statement
        return "[]"
    
    try:
        response = json.loads(result.stdout)
        text = response['choices'][0]['message']['content']
        text = process_llm_response(text)
        print(f"LLM Response: {text}")  # Debugging statement

        pattern = r'\["([^"]+)",\s*({[^}]+})\]'
        match = re.search(pattern, text)
        if match:
            output_list = match.group(0)
            return ast.literal_eval(output_list)
        else:
            print("No match found")
            return "[]"
    except Exception as e:
        print(f"Error processing response: {e}")  # Debugging statement
        return "[]"

