from process_output import process_llm_response
import re
import ast
import subprocess
import json

with open('metadata.json') as f:
    d = json.load(f)


def generate_md(Question, query):
    prompt = f"{Question}{query}"
    print(prompt)
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
    
    response = json.loads(result.stdout)
    text = response['choices'][0]['message']['content']
    text = process_llm_response(text)
    print(text)
    pattern = r'\["([^"]+)",\s*({[^}]+})\]'
    match = re.search(pattern, text)
    if match:
        output_list = match.group(0)
        return ast.literal_eval(output_list)
    else:
        print("No match found")
        return "[]"
