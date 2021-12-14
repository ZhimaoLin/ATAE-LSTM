import argparse
import numpy as np
import pandas as pd
import process_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from ATAE_LSTM import ATAE_LSTM
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from global_config import ASPECT_DICT_TO_TEXT
BATCH_SIZE = 0
EPOCHS = 0
WORD_EMBEDDING_DIM = 0
HIDDEN_DIM = 0

DEVICE = None



def convert_to_tensor(df):
    x_text = df.loc[:, "text_embedding"].values
    y = df.loc[:, ("sentiment_embedding", "aspect")].values
    # y_sentiment = df.loc[:, "sentiment_embedding"].values
    # aspect = df.loc[:, "aspect"].values


    # x_text_train, x_text_test, y_sentiment_train, y_sentiment_test = train_test_split(x_text, y_sentiment, test_size=0.3, random_state=0)
    x_text_train, x_text_test, y_train, y_test = train_test_split(x_text, y, test_size=0.3, random_state=0)

    # describe_train =  pd.DataFrame(y_train)
    describe_test = pd.DataFrame(y_test, columns=["sentiment_embedding", "aspect"])

    # Data summary
    print("========================================")
    print("Test: Total number of test data:")
    print("========================================")
    print(y_test.shape[0])
    print("========================================")
    print("Test: Number of entries of each aspect:")
    print("========================================")
    print(describe_test.loc[:, "aspect"].value_counts())
    print("========================================")



    # Train data
    tensor_list = []
    for row in x_text_train:
        row_tensor = torch.from_numpy(row).float()
        tensor_list.append(row_tensor)
    x_text_train_tensor = torch.stack(tensor_list)

    tensor_list = []
    for row in y_train:
        row_tensor = torch.from_numpy(row[0]).float()
        tensor_list.append(row_tensor)
    y_sentiment_train_tensor = torch.stack(tensor_list)

    dataset_train = TensorDataset(x_text_train_tensor, y_sentiment_train_tensor)


    # Test data
    tensor_list = []
    for row in x_text_test:
        row_tensor = torch.from_numpy(row).float()
        tensor_list.append(row_tensor)
    x_text_test_tensor = torch.stack(tensor_list)

    sentiment_list = []
    # aspect_list = []
    for row in y_test:
        row_sentiment = torch.from_numpy(row[0]).float()
        sentiment_list.append(row_sentiment)
        # row_aspect = row[1]
        # aspect_list.append(row_aspect)
    y_sentiment_test_tensor = torch.stack(sentiment_list)
    y_aspect_test_tensor = torch.from_numpy(y_test[:, 1].astype(int))

    dataset_test = TensorDataset(x_text_test_tensor, y_sentiment_test_tensor, y_aspect_test_tensor)


    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    return dataloader_train, dataloader_test


def train(dataloader_train, word_embedding_dim, hidden_dim, batch_size, sentence_length, num_class=3):
    model = ATAE_LSTM(word_embedding_dim=word_embedding_dim, hidden_dim=hidden_dim, batch_size=batch_size, sentence_length=sentence_length, num_class=num_class).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    h0, c0 = model.init_prev_hidden()
    h0 = h0.to(DEVICE)
    c0 = c0.to(DEVICE)
    
    loss_list = []
    for epoch in range(EPOCHS):
        for data in dataloader_train:
            X, y = data
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            model.zero_grad()
            output, H = model(X, (h0, c0))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

    plt.figure()    
    plt.plot(loss_list)
    plt.savefig()

    return model


def test(model, dataloader_test):
    # Initialize correct dictionary
    correct_dict = {}
    aspect_count = {}
    for key in ASPECT_DICT_TO_TEXT:
        correct_dict[key] = 0
        aspect_count[key] = 0
    total = 0 

    with torch.no_grad():
        h0, c0 = model.init_prev_hidden()
        h0 = h0.to(DEVICE)
        c0 = c0.to(DEVICE)

        for data in dataloader_test:
            X, y, aspect = data
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            aspect = aspect.to(DEVICE)

            output, H = model(X, (h0, c0))

            for i, k in enumerate(aspect):
                if (torch.argmax(output[i]) == torch.argmax(y[i])):
                    correct_dict[k.item()] += 1
                total += 1
                aspect_count[k.item()] += 1


    total_correct = 0
    for k in correct_dict:
        total_correct += correct_dict[k]

        if aspect_count[k] > 0:
            print("========================================")
            print(f"The accuracy of aspect [{ASPECT_DICT_TO_TEXT[k]}]")
            print("========================================")
            print(f"{correct_dict[k]/aspect_count[k]}")
            print("========================================")

    print("========================================")
    print(f"Total Accuracy: {total_correct/total}")
    print("========================================")



def main():
    # Find the hardware for training
    global DEVICE
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Running on the GPU")
    else:
        print(f"Running on the CPU")

    parser = argparse.ArgumentParser(description="LSTM PyTorch")
    parser.add_argument("--data_path", type=str, required=True, help="path to the data")
    parser.add_argument("--glove_path", type=str, required=True, help="path to the glove embedding")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--epoch", type=int, required=True, help="epoch")
    parser.add_argument("--word_embedding_dim", type=int, required=True, help="word embedding dimension")
    parser.add_argument("--hidden_dim", type=int, required=True, help="hidden dimension")
    # parser.add_argument("--sentence_length", type=int, required=True, help="the max length of the sentence")
    # parser.add_argument("--graph_name", type=str, required=True, help="loss graph name")

    args, _ = parser.parse_known_args()
    data_path = args.data_path
    glove_path = args.glove_path
    batch_size = args.batch_size
    epoch = args.epoch
    word_embedding_dim = args.word_embedding_dim
    hidden_dim = args.hidden_dim
    # sentence_length = SENTENCE_LENGTH

    global BATCH_SIZE 
    global EPOCHS 
    global WORD_EMBEDDING_DIM 
    global HIDDEN_DIM 

    BATCH_SIZE = batch_size
    EPOCHS = epoch
    WORD_EMBEDDING_DIM = word_embedding_dim
    HIDDEN_DIM = hidden_dim
    print("========================================")
    print("Configurations")
    print("========================================")
    print(f"Data path: {data_path}")
    print(f"Glove path: {glove_path}")
    print(f"Batch size: {batch_size}")
    print(f"EPOCH: {EPOCHS}")
    print(f"Word embeddig dimension: {WORD_EMBEDDING_DIM}")
    print(f"Hidden dimension: {HIDDEN_DIM}")

    df, sentence_length = process_data.process_data(data_path, glove_path, WORD_EMBEDDING_DIM)
    dataloader_train, dataloader_test = convert_to_tensor(df)

    train_start = time.time()

    model = train(dataloader_train, word_embedding_dim=WORD_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, sentence_length=sentence_length)

    train_end = time.time()
    train_time = train_end - train_start
    print("========================================")
    print(f"Train Time: {train_time} s")
    print("========================================")

    test(model, dataloader_test)




    
if __name__ == '__main__':
    main()