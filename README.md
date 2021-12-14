# Attention-based LSTM with Aspect Embedding (ATAE-LSTM)

This is an application of "[Attention-based LSTM for Aspect-level Sentiment Classification](https://aclanthology.org/D16-1058.pdf)" by Yequan Wang, Minlie Huang, Li Zhao, and Xiaoyan Zhu.

The Attention-based LSTM with Aspect Embedding (ATAE-LSTM) is implemented using [PyTorch](https://pytorch.org/) in order to conduct sentiment analysis on the Twitter [tweets](https://github.com/thunlp/COVID19-CountryImage) during the Covid-19 pandemic.

For detailed explanation, please refer to [ATAE-LSTM Explain](./ATAE_LSTM.pdf).

## Motivation

The main purpose of this project is to have a hands-on experience with the Aspect-based Sentiment Analysis (ABSA) using PyTorch by evaluating the NLP model in the paper above.

At the beginning, I tried the original [code](https://github.com/mindspore-ai/models/tree/master/research/nlp/atae_lstm) of the paper. However, it was developed based on the [MindSpore](https://www.mindspore.cn/en) framework and [Huawei Ascend 910](https://e.huawei.com/en/products/cloud-computing-dc/atlas/ascend-910) processor. Due to my personal hardware limitation, I did not get the code working on my own Windows 10 or Mac machine by trying the following potential solutions:

- Windows 10 + WSL2 (Ubuntu 20.04) + Cuda 11 + Pip
  - [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
  - [Enable TensorFlow with DirectML in WSL](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-wsl)
- Windows 10 + WSL2 (Ubuntu 20.04) + Cuda 11 + Docker
  - [Using MindSpore on Windows 10](https://zhuanlan.zhihu.com/p/267389196)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Mac + Intel CPU
  - Not supported

In the end, I decided to implement the NLP model using PyTorch due to the learning purpose and the looming deadline.

## Development Environment

|                  |                         |
| :--------------- | :---------------------- |
| Operating System | Windows 10 Version 21H2 |
| PyTorch          | 1.10 + Pip + CUDA 11.3  |
| CUDA             | 11.5                    |

## Word Embedding

The word embedding comes from the pre-trained word vector of [Global Vectors for Word Representation (GloVe)](https://nlp.stanford.edu/projects/glove/). The following pre-trained word embedding was explored:

- ./glove_embedding/glove.twitter.27B.25d.txt
- ./glove_embedding/glove.twitter.27B.200d.txt
- ./glove_embedding/glove.6B.200d.txt
- ./glove_embedding/glove.6B.300d.txt

You need to download those text files from [Global Vectors for Word Representation (GloVe)](https://nlp.stanford.edu/projects/glove/) and pass the path to the `train_and_test.py`.

## How to run the code

### Data

The pre-processed dataset is in the `./data/covid_dataset.csv`. Each sentence may have multiple aspects. However, the ATAE-LSTM model can only take one aspect per sentence. So, if a sentence has multiple aspects, I separate them so that each row only contains one aspect of a sentence.  

### Install Packages

Run `pip install -r requirements.txt` to install all required packages. Use `pip3` if you are using Mac.

### Run a Single Instance

Run `python ./train_and_test.py --data_path=./data/covid_dataset.csv --glove_path=./glove_embedding/glove.twitter.27B.25d.txt --batch_size=10 --epoch=10 --word_embedding_dim=25 --hidden_dim=32` to train and test the model.

### Run Multiple Instances

If you want to experiment with different hyperparameters of the model, you can modify the PowerShell Script `run.ps1` and run it. It will pipe the results into a text file to the `result` folder.

## Results

The best overall accuracy comes from the following configurations:

- Word embedding dimension: 25  
- Hidden dimension: 25
- Batch size: 10
- Epoch: 10

The overall accuracy is 71.48%.
