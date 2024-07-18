import warnings
warnings.filterwarnings("ignore")
from model_param import embeddings
from embeddings_and_context import make_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from datetime import datetime

def compute_cosine_similarity(text1, text2):
    return cosine_similarity([text1], [text2])[0][0]

def filter_attributes(metadata_entry, key, value):
    if (key=='title'):
        cos_sim = compute_cosine_similarity(metadata_entry['title'], value)
        return cos_sim
    elif (key == 'author'):
        cos_sim = compute_cosine_similarity(metadata_entry['author'], value)
    elif (key == 'abstract'):
        cos_sim = compute_cosine_similarity(metadata_entry['abstract'], value)
        return cos_sim
    elif (key == 'keywords'):
        cos_sim = compute_cosine_similarity(metadata_entry['keywords'], value)
        return cos_sim
    elif (key == 'publication_date'):
        op = value[0] if value[1].isdigit() else value[0:2]
        value = value[len(op):]
        filter_date = datetime.strptime(value, "%Y-%m-%d")
        if metadata_entry['publication_date'] == "N/A":
            return 0.0
        entry_date = datetime.strptime(metadata_entry['publication_date'], "%Y-%m-%d")
        if (op == '>'):
            return 2.0 if entry_date > filter_date else -6.0
        elif (op == '>='):
            return 2.0 if entry_date >= filter_date else -6.0
        elif (op == '<'):
            return 2.0 if entry_date < filter_date else -6.0
        elif (op == '<='):
            return 2.0 if entry_date <= filter_date else -6.0
        else:
            return 2.0 if entry_date == filter_date else -6.0
    elif (key == 'results'):
        if (type(metadata_entry['results'])==list):
            metadata_entry['results'] = " ".join(metadata_entry['results'])
        cos_sim = compute_cosine_similarity(metadata_entry['results'], value)
        return cos_sim
    else:
        return 0.0



