from query_extraction import generate_md
import text_split
from model_param import CFG
from embeddings_and_context import make_context
from process_output import llm_ans
from filter_by_metadata import filter_data
from data_preprocess import preprocess
import json
import time 
import warnings
import openai
warnings.filterwarnings("ignore")

with open('metadata.json') as f:
    d = json.load(f)

Question = """Your task is to identify the attributes/features of the metadata from a given user query. The attributes/features you need to identify are:

title
author
abstract
keywords
publication_date
arxiv_id
results

Important Notes:
- All attributes except 'keywords' may or may not be present in the given query.
- If a query specifies a date, include "<", ">", ">=", "<=", "=" to denote before, after, after and on, before and on, and on the publication date, respectively.
- Separate the user query into the main query and the metadata attributes.
- If the query includes a metadata attribute term (e.g., author) without a specific name, include it in the main query instead of identifying it.
- The main query should not contain the identified metadata attributes.
- The 'abstract' attribute should be present in all, but it shouldn't be more than 20 words long.
- Give the answer at all costs.

Examples:
1. Query: "Can you tell me the authors of the paper titled 'An Alternative Source for Dark Energy'?”
   Identified Attributes:
   title: 'An Alternative Source for Dark Energy'
   abstract: 'An Alternative Source for Dark Energy' 
   keywords: 'Alternative Source, Dark Energy'
   Main Query: "Can you tell me the authors of the paper."
   Output: ["Can you tell me the authors of the paper.", {"title": "An Alternative Source for Dark Energy","abstract": "An Alternative Source for Dark Energy" ,"keywords": "Alternative Source, Dark Energy"}]

2. Query: "I need the abstract and results from the recent paper on DNA bending after 27 August 2024.”
   Identified Attributes:
   abstract: 'paper on DNA bending'
   publication_date: '27 August 2024'
   keywords: 'DNA bending'
   Main Query: "I need the abstract and results from the recent paper."
   Output: ["I need the abstract and results from the recent paper.", {"abstract": "paper on DNA bending", "publication_date": ">2024-08-27", "keywords": "DNA bending"}]

3. Query: "I need to know the title and publication date of the recent study by Dr. Williams on gene editing techniques on or before 1st august 2023."
   Identified Attributes:
   author: 'Dr. Williams'
   abstract: 'gene editing techniques'
   keywords: 'gene editing techniques'
   publication_date: '1st august 2023'
   Main Query: "I need to know the title and publication date of the recent study."
   Output: ["I need to know the title and publication date of the recent study.", {"author": "Dr. Williams", "abstract": "gene editing techniques", "keywords": "gene editing techniques", "publication_date": "<=2023-08-01"}]

4. Query: "Can you find the keywords and publication date for the paper titled 'Advances in Machine Learning' authored by Dr. Jane Doe?"
   Identified Attributes:
   title: 'Advances in Machine Learning'
   author: 'Dr. Jane Doe'
   abstract: 'paper showing the advances in Machine Learning'
   Main Query: "Can you find the keywords and publication date for the paper."
   Output: ["Can you find the keywords and publication date for the paper.", {"title": "Advances in Machine Learning", "author": "Dr. Jane Doe", "abstract": "paper showing the advances in Machine Learning"}]

5. Query: "I want the arxiv_id and results of the paper by Dr. Smith on climate change adaptation published on 22 May 2022."
   Identified Attributes:
   author: 'Dr. Smith'
   abstract: 'paper on climate change adaptation'
   keywords: 'climate change adaptation'
   publication_date: '22 May 2022'
   Main Query: "I want the arxiv_id and results of the paper."
   Output: ["I want the arxiv_id and results of the paper.", {"author": "Dr. Smith", "abstract": "paper on climate change adaptation", "keywords": "climate change, adaptation", "publication_date": "2022-05-22"}]

6. Query: "Please give me the abstract of the research on blockchain technology."
    Identified Attributes:
    abstract: 'Research on blockchain technology.'
    keywords: 'blockchain technology'
    Main Query: "Please give me the abstract of the research."
    Output: ["Please give me the abstract of the research.", {"abstract": "Research on blockchain technology.","keywords": "blockchain technology"}]

The answer should only be a list and no other content whatsoever. Please print the Output for the following query:\n"""

list_of_documents = text_split.text_split(d)
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)
def ans(context, question):
   prompt = f"""<|system|>
   
   You are given some extracted parts in a paragraph from research papers along with a question. Everything in the extract may not be important. Choose carefully!

   If you don't know the answer, just say "I don't know." Don't try to make up an answer.

   It is very important that you ALWAYS answer the question in the same language the question is in. Remember to always do that.

   Your answer should not be more than {CFG.max_len} words long.

   The answer should be grammatically correct and start from the beginning of a sentence.

   Use the following pieces of context to answer the question at the end.

   Context: {context}

   Question is below. Remember to answer only in English:

   Question: {question}

   <|end|>

   <|assistant|>

   """

   response = client.chat.completions.create(
   model="phi3",
   temperature=0.4,
   n=1,
   messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
   ],)
   print(f"\n\n\n{prompt}\n\n\n")
   print("Response:")
   print(response.choices[0].message.content)
   print()

query = input("Enter your query here. Write 'stop' to terminate running.")

while (query.lower() != "stop"):
    
    start_time = time.time()
    
    out = generate_md(Question,query)
    
    filtered_metadata = filter_data(d,out[1])
    
    context = preprocess(make_context(list_of_documents, filtered_metadata[0],out))

    print(context)
    ans(llm,context,out[0])
    print("Source Document: "+ filtered_metadata[0]['title'])
    
    print("Time Taken: "+ str(time.time() - start_time))
    query = input("Enter your query here. Write 'stop' to terminate running.")
