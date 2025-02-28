import re
import os
import json
from PIL import Image
from openai import OpenAI
from pandas import DataFrame
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity


load_dotenv()
# make sure to add an env file with your open ai key 
API_KEY = os.getenv("OPENAI_API_KEY")


# This was run on a mac so no cuda was enabled to keep it consistent
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
openAI_client = OpenAI(api_key=API_KEY)


def extract_image_names(image_data):
    """
    returns just the image names and removes any path data
    """
    return [path.split("/")[-1] for path, _ in image_data]


def extract_json_from_response(response_text: str) -> dict:
    """Helper function to help extract the contents of the json response from gpt"""
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    json_content = match.group(0)
    parsed_json = json.loads(json_content)
    return parsed_json


def get_ranked_images(text_description: str, image_paths: list) -> list:
    """Uses clip to compute similarity scores between a text description and a set of images and returns in descending order."""
    images = [Image.open(img_path) for img_path in image_paths]
    inputs = processor(
        text=[text_description], images=images, return_tensors="pt", padding=True
    )
    outputs = model(**inputs)
    text_embedding, image_embeddings = outputs.text_embeds, outputs.image_embeds
    similarity_scores = cosine_similarity(text_embedding, image_embeddings).squeeze(0)
    image_score_pairs = list(zip(image_paths, similarity_scores.tolist()))
    image_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return image_score_pairs


def check_word_usage(word, sentence):
    """Queries gpt to determine if a word's usage is either literal or idiomatic"""
    
    prompt = (
        f"Determine whether the usage of the word '{word}' in the following sentence is idiomatic or literal. This is important to me: \n"
        f"Sentence: {sentence}\n"
        f"Respond with 'idiomatic' or 'literal' and dont give an explanation"
    )
    response = openAI_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert language analyzer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.5,
    )
    content = response.choices[0].message.content.strip()
    classification = content.split("\n")[0].strip().lower()
    return classification


def get_idiom_meanings(idiomatic_word: str) -> dict:
    """ Gets both the literal and idiomatic meanings of a NC using gpt"""
    
    prompt = f'''
    Analyze the given term: "{idiomatic_word}".
    1. **Literal Meaning**: 
    - Provide a short and descriptive explanation of the literal meaning of "{idiomatic_word}". Include things that can help an AI model find it 
    - Then, generate 5 strongly associated keywords that could strongly describe the literal meaning of the word. Focus on the most relevant and descriptive words.
    - Also generate 3 sentiments that could relate strongly to it based on the sentence given.
    2. **Idiomatic Meaning**:
    - Provide a short and descriptive explanation of the idiomatic meaning of "{idiomatic_word}" including "{idiomatic_word}" in the explanation. Include things that can help an AI model find it.
    - Then, generate 5 descriptive keywords that strongly describe the idiomatic meaning based on the sentence, focusing on its core essence and usage. 
    - Also generate 3 sentiments or emotions that relate strongly to it based on the sentence given.

    Return the result in the following JSON format: 
        "literal_meaning": "<brief explanation of the literal meaning>",
        "literal_keywords": ["<word1>", "<word2>", "<word3>", ...],
        "literal_sentiments" : ["<word1>", "<word2>", "<word3>", ...],
        "idiomatic_meaning": "<brief explanation of the idiomatic meaning>",
        "idiomatic_keywords": ["<word1>", "<word2>", "<word3>", ...]
        "idiomatic_sentiments" : ["<word1>", "<word2>", "<word3>", ...],
        
    This is important to me
    '''
    
    response = openAI_client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides definitions in JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    content = response.choices[0].message.content
    results = extract_json_from_response(content)
    return results


def translate_portuguese_to_english(portuguese_text):
    """Uses GPT to get a direct translation of a portugese sentence in english"""

    response = openAI_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": f"Translate the following Portuguese text to English, return just the translation and nothing else: {portuguese_text}"}
            ]
    )
    english_text = response.choices[0].message.content.strip()
    return english_text


def translate_portugese_dataframe(dataframe: DataFrame):
    """helper fucntion to translate the 'compound' and 'sentence' columns of a Portuguese df"""
    
    translated_dictionary = {}
    for _, rows in dataframe.iterrows():
        compound = rows['compound']
        sentence = rows['sentence']
        translated_compound = translate_portuguese_to_english(compound)
        translated_sentence = translate_portuguese_to_english(sentence)
        translated_dictionary[compound] = {
            "compound": translated_compound,
            "sentence": translated_sentence
        }
        
    return translated_dictionary