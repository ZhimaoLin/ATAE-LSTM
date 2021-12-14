import pandas as pd
import numpy as np
import argparse
import re
from nltk.tokenize import word_tokenize

from global_config import ASPECT_LIST, ASPECT_DICT_TO_DIGIT

SENTENCE_LENGTH = 0



def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop(df.columns[0], axis=1)
    category_df = df.iloc[:, 2:]
    result = df.iloc[:, 0:2]

    aspect_sentiment_df = pd.DataFrame(np.zeros((category_df.shape[0], 2), dtype=int), columns=['aspect', 'sentiment'])

    for column_name, column in category_df.iteritems():
        index_list = column.index[column > 0].tolist()
        aspect_sentiment_df.iloc[index_list , 0] = ASPECT_DICT_TO_DIGIT[column_name]
        aspect_sentiment_df.iloc[index_list , 1] = column[index_list]

    result = pd.concat([result, aspect_sentiment_df], axis=1)

    # remove the texts with no aspect
    index_no_aspect = result.index[result['aspect'] == 0].tolist()
    old_text_number = result.shape[0]
    result = result.drop(index_no_aspect, axis=0)
    result = result.reset_index(drop=True)
    new_text_number = result.shape[0]
    print(f"There are [{old_text_number - new_text_number}] texts with no aspect.")
    print(f"There are [{new_text_number}] texts left to analyze.")

    return result



def remove_email_url_mention(df):
    email_regex = "\w+@\w+\.\w+"
    url_regex = "https?:\/\/\S*"
    mention_regex = "@\w+"

    for index, row in df.iterrows():
        row["text"] = re.sub(email_regex, "", row["text"])
        row["text"] = re.sub(url_regex, "", row["text"])
        row["text"] = re.sub(mention_regex, "", row["text"])
        df.loc[index, "text"] = row["text"].lower()

    return df



def load_glove_embedding(glove_path):
    word_dict = {}

    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            parse = line.strip(" \n").split(" ")
            text = parse[0]
            embedding = parse[1:]
            embedding = np.array(embedding).astype(float)
            word_dict[text] = embedding

    return word_dict



def convert_word_to_embedding(df, word_dict, word_embedding_dim):
    embedding = pd.DataFrame(np.zeros((df.shape[0], 4)), columns=["text_embedding", "aspect_embedding", "sentiment_embedding", "word_count"])
    embedding = embedding.astype(object)
    embedding.loc[:, "text_embedding"] = [[] for _ in range(len(embedding))]
    embedding.loc[:, "aspect_embedding"] = [[] for _ in range(len(embedding))]
    embedding.loc[:, "sentiment_embedding"] = [[] for _ in range(len(embedding))]

    negative = np.array([1,0,0]) # 1
    positive = np.array([0,0,1]) # 3
    neutral = np.array([0,1,0]) # 2
    sentiment_dict = {1:negative, 2:positive, 3:neutral}

    for key in sentiment_dict:
        index_list = df.index[df['sentiment'] == key]
        embedding.loc[index_list , "sentiment_embedding"] = [sentiment_dict[key] for _ in range(len(index_list))]

    for aspect in ASPECT_LIST:
        aspect_embedding = word_dict[aspect]
        index_list = df.index[df["aspect"] == ASPECT_DICT_TO_DIGIT[aspect]].tolist()
        embedding.loc[index_list, "aspect_embedding"] = [aspect_embedding for _ in range(len(index_list))]


    for row_index, row in df.iterrows():
        sentence = word_tokenize(row["text"])
        embedding.at[row_index, "word_count"] = len(sentence)
    max_word_count = embedding["word_count"].max()
    mean_word_count = embedding["word_count"].mean()
    print(f"The maximum length of the sentences is [{max_word_count}]")
    print(f"The average length of the sentences is [{mean_word_count}]")

    global SENTENCE_LENGTH
    SENTENCE_LENGTH = max_word_count

    not_in_dict_count = 0
    padding = np.zeros(word_embedding_dim*2)
    for row_index, row in df.iterrows():
        ae = embedding.loc[row_index, "aspect_embedding"]
        sentence = word_tokenize(row["text"])

        word_embedding = []
        for word in sentence:
            try:
                e = word_dict[word]
                word_embedding.append(np.concatenate((e, ae)))
            except:
                not_in_dict_count +=1
                continue

        difference = max_word_count - len(word_embedding)
        word_embedding = word_embedding + [padding]*difference
        embedding.at[row_index, "text_embedding"] = np.array(word_embedding)

    result = pd.concat([df, embedding], axis=1)
    print(f"There are [{not_in_dict_count}] number of words that are not in the glove library.")
    return result



def process_data(data_path, glove_path, word_embedding_dim):
    df = read_data(data_path)
    df = remove_email_url_mention(df)

    word_dict = load_glove_embedding(glove_path)
    result = convert_word_to_embedding(df, word_dict, word_embedding_dim)

    print("========================================")
    print("Number of entries of each aspect:")
    print("========================================")
    print(result.loc[:, "aspect"].value_counts())
    print("========================================")
    
    print("========================================")
    print("Number of entries of each sentiment:")
    print("========================================")
    print(result.loc[:, "sentiment"].value_counts())
    print("========================================")

    return result, SENTENCE_LENGTH



def main():
    parser = argparse.ArgumentParser(description="Process Data")
    parser.add_argument("--data_path", type=str, required=True, help="path to the data")
    parser.add_argument("--glove_path", type=str, required=True, help="path to the glove embedding")
    parser.add_argument("--word_embedding_dim", type=int, required=True, help="word embedding dimension")

    args, _ = parser.parse_known_args()
    data_path = args.data_path
    glove_path = args.glove_path
    word_embedding_dim = args.word_embedding_dim

    result = process_data(data_path, glove_path, word_embedding_dim)
  

    
if __name__ == '__main__':
    main()

