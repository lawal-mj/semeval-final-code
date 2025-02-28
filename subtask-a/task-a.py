import os
import pandas as pd
from utils import (
    get_ranked_images, 
    extract_image_names, 
    check_word_usage, 
    get_idiom_meanings, 
    translate_portugese_dataframe
)

# english file information
ENGLISH_FILE_PATH = "english/test/subtask_a_test.tsv"
ENGLISH_OUTPUT_PATH = "submission_EN.tsv"
ENGLISH_PATH_PREFIX = "english/test/"

# portugese file information 
PORTUGESE_FILE_PATH = "portugese/test/subtask_a_test.tsv"
PORTUGESE_OUTPUT_PATH = "submission_PT.tsv"
PORTUGESE_PATH_PREFIX = "portugese/test/"


def handle_portugese_data():
    """reads a Portuguese TSV file, translates each 'compound' and 'sentence' to English, determines the usage as either idiomatic or literal, ranks the images and saves the output"""
    
    final_results = []
    portugese_dataframe = pd.read_csv(PORTUGESE_FILE_PATH, sep='\t')
    translations_dictionary = translate_portugese_dataframe(portugese_dataframe)
    
    for _, row in portugese_dataframe.iterrows():
        portugese_compound = row['compound']
        
        portugese_idiomatic_compound = translations_dictionary[portugese_compound]['compound']
        portugese_sentence = translations_dictionary[portugese_compound]['sentence']
        
        meanings = get_idiom_meanings(idiomatic_word=portugese_idiomatic_compound)
        
        literal_keywords = " ".join(meanings['literal_keywords'])
        literal_sentiments = " ".join(meanings['literal_sentiments'])
        # literal_meaning = meanings['literal_meaning']
        
        idiomatic_keywords = " ".join(meanings['idiomatic_keywords'])
        idiomatic_sentiments = " ".join(meanings['idiomatic_sentiments'])
        # idiomatic_meaning = meanings['idiomatic_meaning']
        
        image_paths = [os.path.join(PORTUGESE_PATH_PREFIX, portugese_compound, row[f'image{i}_name']) for i in range(1, 6)]
        word_usage = check_word_usage(portugese_idiomatic_compound, portugese_sentence)
        
        rankings = []
        if word_usage == "idiomatic":
            rankings = get_ranked_images(idiomatic_keywords + idiomatic_sentiments, image_paths)
            print(portugese_idiomatic_compound, word_usage)
            
        elif word_usage == "literal":
            rankings = get_ranked_images(literal_keywords + literal_sentiments, image_paths)
            print(portugese_idiomatic_compound, word_usage)

        image_ranking_results = extract_image_names(rankings)
        final_results.append({
            "compound": portugese_compound,
            "expected_order": image_ranking_results
        })
        
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(PORTUGESE_OUTPUT_PATH, sep='\t', index=False)
    print(f"Portugese results saved to {PORTUGESE_OUTPUT_PATH}")
    

def handle_english_data():
    """reads an english TSV file,  determines the usage as either idiomatic or literal, ranks the images and saves the output"""
    
    dataframe = pd.read_csv(ENGLISH_FILE_PATH, sep='\t')
    final_results = []

    for _, row in dataframe.iterrows():
        idiomatic_compound = row['compound']
        sentence = row['sentence']
        meanings = get_idiom_meanings(idiomatic_word=idiomatic_compound)
        
        literal_keywords = " ".join(meanings['literal_keywords'])
        literal_sentiments = " ".join(meanings['literal_sentiments'])
        # literal_meaning = meanings['literal_meaning']
        
        idiomatic_keywords = " ".join(meanings['idiomatic_keywords'])
        idiomatic_sentiments = " ".join(meanings['idiomatic_sentiments'])
        # idiomatic_meaning = meanings['idiomatic_meaning']

        image_paths = [os.path.join(ENGLISH_PATH_PREFIX, idiomatic_compound, row[f'image{i}_name']) for i in range(1, 6)]
        word_usage = check_word_usage(idiomatic_compound, sentence)
        
        rankings = []
        if word_usage == "idiomatic":
            rankings = get_ranked_images(idiomatic_keywords+idiomatic_sentiments, image_paths)
            print(idiomatic_compound, word_usage)
            
        elif word_usage == "literal":
            rankings = get_ranked_images(literal_keywords+literal_sentiments, image_paths)
            print(idiomatic_compound, word_usage)

        image_ranking_results = extract_image_names(rankings)
        final_results.append({
            "compound": idiomatic_compound,
            "expected_order": image_ranking_results
        })
        
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(ENGLISH_OUTPUT_PATH, sep='\t', index=False)
    print(f"English results saved to {ENGLISH_OUTPUT_PATH}")
    
    
    
    
# if __name__ == "__main__":
#     handle_english_data()
#     handle_portugese_data()
