# Attention-based LSTM with Aspect Embedding (ATAE-LSTM)

This is an application of "[Attention-based LSTM for Aspect-level Sentiment Classification](https://aclanthology.org/D16-1058.pdf)" by Yequan Wang, Minlie Huang, Li Zhao, and Xiaoyan Zhu.

The Attention-based LSTM with Aspect Embedding (ATAE-LSTM) is implemented using [PyTorch](https://pytorch.org/) in order to conduct sentiment analysis on the Twitter [tweets](https://github.com/thunlp/COVID19-CountryImage) during the Covid-19 pandemic.

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

|||
|:---|:---|
|dsfas|dsaf|



## Word Embedding

The word embedding comes from the pre-trained word vector of [Global Vectors for Word Representation (GloVe)](https://nlp.stanford.edu/projects/glove/). The following pre-trained word embedding was explored:

- glove.twitter.27B.25d.txt 
- glove.twitter.27B.50d.txt
- glove.twitter.27B.100d.txt
- glove.twitter.27B.200d.txt
- glove.6B.50d.txt
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt


