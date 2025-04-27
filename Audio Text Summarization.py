{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DkRI1kcXvrbW",
        "outputId": "8443e1fd-f324-4b52-826d-f80d3f897912"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Wed Jan 18 09:35:03 2023       \n",
            "+-----------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |\n",
            "|-------------------------------+----------------------+----------------------+\n",
            "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                               |                      |               MIG M. |\n",
            "|===============================+======================+======================|\n",
            "|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   70C    P0    30W /  70W |      0MiB / 15109MiB |      0%      Default |\n",
            "|                               |                      |                  N/A |\n",
            "+-------------------------------+----------------------+----------------------+\n",
            "                                                                               \n",
            "+-----------------------------------------------------------------------------+\n",
            "| Processes:                                                                  |\n",
            "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
            "|        ID   ID                                                   Usage      |\n",
            "|=============================================================================|\n",
            "|  No running processes found                                                 |\n",
            "+-----------------------------------------------------------------------------+\n"
          ]
        }
      ],
      "source": [
        "!nvidia-smi"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install git+https://github.com/openai/whisper.git \n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wCQ6icKlwEUZ",
        "outputId": "78193267-30de-4dcc-d9de-c95fb00a5b8b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting git+https://github.com/openai/whisper.git\n",
            "  Cloning https://github.com/openai/whisper.git to /tmp/pip-req-build-4rxmxuvg\n",
            "  Running command git clone --filter=blob:none --quiet https://github.com/openai/whisper.git /tmp/pip-req-build-4rxmxuvg\n",
            "  Resolved https://github.com/openai/whisper.git to commit 9d646db9d885d7d86fa5f40aa29a48147379ed70\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from openai-whisper==20230117) (1.21.6)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.8/dist-packages (from openai-whisper==20230117) (1.13.1+cu116)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.8/dist-packages (from openai-whisper==20230117) (4.64.1)\n",
            "Requirement already satisfied: more-itertools in /usr/local/lib/python3.8/dist-packages (from openai-whisper==20230117) (9.0.0)\n",
            "Collecting transformers>=4.19.0\n",
            "  Downloading transformers-4.25.1-py3-none-any.whl (5.8 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m5.8/5.8 MB\u001b[0m \u001b[31m99.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting ffmpeg-python==0.2.0\n",
            "  Downloading ffmpeg_python-0.2.0-py3-none-any.whl (25 kB)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.8/dist-packages (from ffmpeg-python==0.2.0->openai-whisper==20230117) (0.16.0)\n",
            "Collecting huggingface-hub<1.0,>=0.10.0\n",
            "  Downloading huggingface_hub-0.11.1-py3-none-any.whl (182 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m182.4/182.4 KB\u001b[0m \u001b[31m23.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: filelock in /usr/local/lib/python3.8/dist-packages (from transformers>=4.19.0->openai-whisper==20230117) (3.9.0)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.8/dist-packages (from transformers>=4.19.0->openai-whisper==20230117) (2022.6.2)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.8/dist-packages (from transformers>=4.19.0->openai-whisper==20230117) (21.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.8/dist-packages (from transformers>=4.19.0->openai-whisper==20230117) (2.25.1)\n",
            "Collecting tokenizers!=0.11.3,<0.14,>=0.11.1\n",
            "  Downloading tokenizers-0.13.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.6 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.6/7.6 MB\u001b[0m \u001b[31m105.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.8/dist-packages (from transformers>=4.19.0->openai-whisper==20230117) (6.0)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.8/dist-packages (from torch->openai-whisper==20230117) (4.4.0)\n",
            "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from packaging>=20.0->transformers>=4.19.0->openai-whisper==20230117) (3.0.9)\n",
            "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests->transformers>=4.19.0->openai-whisper==20230117) (1.24.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests->transformers>=4.19.0->openai-whisper==20230117) (2022.12.7)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests->transformers>=4.19.0->openai-whisper==20230117) (2.10)\n",
            "Requirement already satisfied: chardet<5,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests->transformers>=4.19.0->openai-whisper==20230117) (4.0.0)\n",
            "Building wheels for collected packages: openai-whisper\n",
            "  Building wheel for openai-whisper (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for openai-whisper: filename=openai_whisper-20230117-py3-none-any.whl size=1178769 sha256=c2106808bc3b21a0d1674a8e9866d926ecdd548bf1f7fd6c55c59b4e4105f74b\n",
            "  Stored in directory: /tmp/pip-ephem-wheel-cache-ii1ylx44/wheels/a7/70/18/b7693c07b1d18b3dafb328f5d0496aa0d41a9c09ef332fd8e6\n",
            "Successfully built openai-whisper\n",
            "Installing collected packages: tokenizers, ffmpeg-python, huggingface-hub, transformers, openai-whisper\n",
            "Successfully installed ffmpeg-python-0.2.0 huggingface-hub-0.11.1 openai-whisper-20230117 tokenizers-0.13.2 transformers-4.25.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import whisper\n",
        "\n",
        "model = whisper.load_model(\"base\")\n",
        "result = model.transcribe(\"maudio.mp3\")\n",
        "\n",
        "print(result[\"text\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e88vH5BBwaZI",
        "outputId": "2f4835a9-025a-4f45-aeb8-a2552dc6a1a7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|███████████████████████████████████████| 139M/139M [00:02<00:00, 52.8MiB/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " There are boardly two types of extractive summarization tasks depending on what the summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection, whether documents, or sets of images, or videos, news stories etc. The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Common systems are able to create both query relevant text summaries and generic machine generated summaries depending on what the user needs. An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents, for example, a cluster of articles on the same topic. This problem is called multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically pulls together news articles on a given topic, from the web, and can precisely represents the latest news as a summary. Image collection summarization is another application example of automatic summarization. It consists in selecting a representative set of images from a larger set of images. 3. A summary in this context is useful to show the most representative images of results in an image collection exploration system. Video summarization is a related domain, where the system automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "text=result[\"text\"]\n"
      ],
      "metadata": {
        "id": "piTQXTrOw2nF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import spacy\n",
        "from spacy.lang.en.stop_words import STOP_WORDS\n",
        "from string import punctuation"
      ],
      "metadata": {
        "id": "Qhl04ItqxTaM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "stopwords = list(STOP_WORDS)"
      ],
      "metadata": {
        "id": "mivAkgVbxUkQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "nlp = spacy.load('en_core_web_sm')"
      ],
      "metadata": {
        "id": "t7h7uG5yxWgt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "doc = nlp(text)"
      ],
      "metadata": {
        "id": "3lfaJCdExX8i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(text)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FU-ngU-_xZu1",
        "outputId": "ef9f96c1-d37d-4c27-c380-27296e9b77da"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " There are boardly two types of extractive summarization tasks depending on what the summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection, whether documents, or sets of images, or videos, news stories etc. The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Common systems are able to create both query relevant text summaries and generic machine generated summaries depending on what the user needs. An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents, for example, a cluster of articles on the same topic. This problem is called multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically pulls together news articles on a given topic, from the web, and can precisely represents the latest news as a summary. Image collection summarization is another application example of automatic summarization. It consists in selecting a representative set of images from a larger set of images. 3. A summary in this context is useful to show the most representative images of results in an image collection exploration system. Video summarization is a related domain, where the system automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "tokens = [token.text for token in doc]\n",
        "print(tokens)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wxxNwPUVxb7i",
        "outputId": "75980976-0d0c-4c48-a8b3-871229fe5f66"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[' ', 'There', 'are', 'boardly', 'two', 'types', 'of', 'extractive', 'summarization', 'tasks', 'depending', 'on', 'what', 'the', 'summarization', 'program', 'focuses', 'on', '.', 'The', 'first', 'is', 'generic', 'summarization', ',', 'which', 'focuses', 'on', 'obtaining', 'a', 'generic', 'summary', 'or', 'abstract', 'of', 'the', 'collection', ',', 'whether', 'documents', ',', 'or', 'sets', 'of', 'images', ',', 'or', 'videos', ',', 'news', 'stories', 'etc', '.', 'The', 'second', 'is', 'query', 'relevant', 'summarization', ',', 'sometimes', 'called', 'query', '-', 'based', 'summarization', ',', 'which', 'summarizes', 'objects', 'specific', 'to', 'a', 'query', '.', 'Common', 'systems', 'are', 'able', 'to', 'create', 'both', 'query', 'relevant', 'text', 'summaries', 'and', 'generic', 'machine', 'generated', 'summaries', 'depending', 'on', 'what', 'the', 'user', 'needs', '.', 'An', 'example', 'of', 'a', 'summarization', 'problem', 'is', 'document', 'summarization', ',', 'which', 'attempts', 'to', 'automatically', 'produce', 'an', 'abstract', 'from', 'a', 'given', 'document', '.', 'Sometimes', 'one', 'might', 'be', 'interested', 'in', 'generating', 'a', 'summary', 'from', 'a', 'single', 'source', 'document', ',', 'while', 'others', 'can', 'use', 'multiple', 'source', 'documents', ',', 'for', 'example', ',', 'a', 'cluster', 'of', 'articles', 'on', 'the', 'same', 'topic', '.', 'This', 'problem', 'is', 'called', 'multi', '-', 'document', 'summarization', '.', 'A', 'related', 'application', 'is', 'summarizing', 'news', 'articles', '.', 'Imagine', 'a', 'system', ',', 'which', 'automatically', 'pulls', 'together', 'news', 'articles', 'on', 'a', 'given', 'topic', ',', 'from', 'the', 'web', ',', 'and', 'can', 'precisely', 'represents', 'the', 'latest', 'news', 'as', 'a', 'summary', '.', 'Image', 'collection', 'summarization', 'is', 'another', 'application', 'example', 'of', 'automatic', 'summarization', '.', 'It', 'consists', 'in', 'selecting', 'a', 'representative', 'set', 'of', 'images', 'from', 'a', 'larger', 'set', 'of', 'images', '.', '3', '.', 'A', 'summary', 'in', 'this', 'context', 'is', 'useful', 'to', 'show', 'the', 'most', 'representative', 'images', 'of', 'results', 'in', 'an', 'image', 'collection', 'exploration', 'system', '.', 'Video', 'summarization', 'is', 'a', 'related', 'domain', ',', 'where', 'the', 'system', 'automatically', 'creates', 'a', 'trailer', 'of', 'a', 'long', 'video', '.', 'This', 'also', 'has', 'applications', 'in', 'consumer', 'or', 'personal', 'videos', ',', 'where', 'one', 'might', 'want', 'to', 'skip', 'the', 'boring', 'or', 'repetitive', 'actions', '.', 'Similarly', ',', 'in', 'surveillance', 'videos', ',', 'one', 'would', 'want', 'to', 'extract', 'important', 'and', 'suspicious', 'activity', ',', 'while', 'ignoring', 'all', 'the', 'boring', 'and', 'redundant', 'frames', 'captured', '.']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "punctuation = punctuation + '\\n'\n",
        "punctuation"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "STQPvORyxgGV",
        "outputId": "e86cc9c2-31a4-4529-90b6-9d1784dcb006"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~\\n'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~\\n'\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "B-5O0qJxxiYE",
        "outputId": "1ca853cd-d4bd-4fdb-8755-f6a91304165c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~\\n'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "word_frequencies = {}\n",
        "for word in doc:\n",
        "    if word.text.lower() not in stopwords:\n",
        "        if word.text.lower() not in punctuation:\n",
        "            if word.text not in word_frequencies.keys():\n",
        "                word_frequencies[word.text] = 1\n",
        "            else:\n",
        "                word_frequencies[word.text] += 1\n",
        "                \n",
        "print(word_frequencies)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p4B05A5rxl-M",
        "outputId": "c3d9cbe5-f5f6-4fed-9d80-6866c1e1a222"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{' ': 1, 'boardly': 1, 'types': 1, 'extractive': 1, 'summarization': 11, 'tasks': 1, 'depending': 2, 'program': 1, 'focuses': 2, 'generic': 3, 'obtaining': 1, 'summary': 4, 'abstract': 2, 'collection': 3, 'documents': 2, 'sets': 1, 'images': 4, 'videos': 3, 'news': 4, 'stories': 1, 'etc': 1, 'second': 1, 'query': 4, 'relevant': 2, 'called': 2, 'based': 1, 'summarizes': 1, 'objects': 1, 'specific': 1, 'Common': 1, 'systems': 1, 'able': 1, 'create': 1, 'text': 1, 'summaries': 2, 'machine': 1, 'generated': 1, 'user': 1, 'needs': 1, 'example': 3, 'problem': 2, 'document': 4, 'attempts': 1, 'automatically': 3, 'produce': 1, 'given': 2, 'interested': 1, 'generating': 1, 'single': 1, 'source': 2, 'use': 1, 'multiple': 1, 'cluster': 1, 'articles': 3, 'topic': 2, 'multi': 1, 'related': 2, 'application': 2, 'summarizing': 1, 'Imagine': 1, 'system': 3, 'pulls': 1, 'web': 1, 'precisely': 1, 'represents': 1, 'latest': 1, 'Image': 1, 'automatic': 1, 'consists': 1, 'selecting': 1, 'representative': 2, 'set': 2, 'larger': 1, '3': 1, 'context': 1, 'useful': 1, 'results': 1, 'image': 1, 'exploration': 1, 'Video': 1, 'domain': 1, 'creates': 1, 'trailer': 1, 'long': 1, 'video': 1, 'applications': 1, 'consumer': 1, 'personal': 1, 'want': 2, 'skip': 1, 'boring': 2, 'repetitive': 1, 'actions': 1, 'Similarly': 1, 'surveillance': 1, 'extract': 1, 'important': 1, 'suspicious': 1, 'activity': 1, 'ignoring': 1, 'redundant': 1, 'frames': 1, 'captured': 1}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "max_frequency = max(word_frequencies.values())\n",
        "max_frequency"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gdCc_ehdxolz",
        "outputId": "dd07e0b9-b6d7-4a36-a77f-1f36625b094f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "11"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for word in word_frequencies.keys():\n",
        "    word_frequencies[word] = word_frequencies[word]/max_frequency\n",
        "\n",
        "print(word_frequencies)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ogsx5b2mxsKj",
        "outputId": "5f1878e8-7067-415e-f102-9a2b5de1512d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{' ': 0.09090909090909091, 'boardly': 0.09090909090909091, 'types': 0.09090909090909091, 'extractive': 0.09090909090909091, 'summarization': 1.0, 'tasks': 0.09090909090909091, 'depending': 0.18181818181818182, 'program': 0.09090909090909091, 'focuses': 0.18181818181818182, 'generic': 0.2727272727272727, 'obtaining': 0.09090909090909091, 'summary': 0.36363636363636365, 'abstract': 0.18181818181818182, 'collection': 0.2727272727272727, 'documents': 0.18181818181818182, 'sets': 0.09090909090909091, 'images': 0.36363636363636365, 'videos': 0.2727272727272727, 'news': 0.36363636363636365, 'stories': 0.09090909090909091, 'etc': 0.09090909090909091, 'second': 0.09090909090909091, 'query': 0.36363636363636365, 'relevant': 0.18181818181818182, 'called': 0.18181818181818182, 'based': 0.09090909090909091, 'summarizes': 0.09090909090909091, 'objects': 0.09090909090909091, 'specific': 0.09090909090909091, 'Common': 0.09090909090909091, 'systems': 0.09090909090909091, 'able': 0.09090909090909091, 'create': 0.09090909090909091, 'text': 0.09090909090909091, 'summaries': 0.18181818181818182, 'machine': 0.09090909090909091, 'generated': 0.09090909090909091, 'user': 0.09090909090909091, 'needs': 0.09090909090909091, 'example': 0.2727272727272727, 'problem': 0.18181818181818182, 'document': 0.36363636363636365, 'attempts': 0.09090909090909091, 'automatically': 0.2727272727272727, 'produce': 0.09090909090909091, 'given': 0.18181818181818182, 'interested': 0.09090909090909091, 'generating': 0.09090909090909091, 'single': 0.09090909090909091, 'source': 0.18181818181818182, 'use': 0.09090909090909091, 'multiple': 0.09090909090909091, 'cluster': 0.09090909090909091, 'articles': 0.2727272727272727, 'topic': 0.18181818181818182, 'multi': 0.09090909090909091, 'related': 0.18181818181818182, 'application': 0.18181818181818182, 'summarizing': 0.09090909090909091, 'Imagine': 0.09090909090909091, 'system': 0.2727272727272727, 'pulls': 0.09090909090909091, 'web': 0.09090909090909091, 'precisely': 0.09090909090909091, 'represents': 0.09090909090909091, 'latest': 0.09090909090909091, 'Image': 0.09090909090909091, 'automatic': 0.09090909090909091, 'consists': 0.09090909090909091, 'selecting': 0.09090909090909091, 'representative': 0.18181818181818182, 'set': 0.18181818181818182, 'larger': 0.09090909090909091, '3': 0.09090909090909091, 'context': 0.09090909090909091, 'useful': 0.09090909090909091, 'results': 0.09090909090909091, 'image': 0.09090909090909091, 'exploration': 0.09090909090909091, 'Video': 0.09090909090909091, 'domain': 0.09090909090909091, 'creates': 0.09090909090909091, 'trailer': 0.09090909090909091, 'long': 0.09090909090909091, 'video': 0.09090909090909091, 'applications': 0.09090909090909091, 'consumer': 0.09090909090909091, 'personal': 0.09090909090909091, 'want': 0.18181818181818182, 'skip': 0.09090909090909091, 'boring': 0.18181818181818182, 'repetitive': 0.09090909090909091, 'actions': 0.09090909090909091, 'Similarly': 0.09090909090909091, 'surveillance': 0.09090909090909091, 'extract': 0.09090909090909091, 'important': 0.09090909090909091, 'suspicious': 0.09090909090909091, 'activity': 0.09090909090909091, 'ignoring': 0.09090909090909091, 'redundant': 0.09090909090909091, 'frames': 0.09090909090909091, 'captured': 0.09090909090909091}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sentence_tokens = [sent for sent in doc.sents]\n",
        "len(sentence_tokens)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jHIJBx4ZxuKZ",
        "outputId": "60e83aa3-331c-4dc5-abdb-a66ed722c8c8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "16"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sentence_scores = {}\n",
        "for sent in sentence_tokens:\n",
        "    for word in sent:\n",
        "        if word.text.lower() in word_frequencies.keys():\n",
        "            if sent not in sentence_scores.keys():\n",
        "                sentence_scores[sent] = word_frequencies[word.text.lower()]\n",
        "            else:\n",
        "                sentence_scores[sent] += word_frequencies[word.text.lower()]\n",
        "                \n",
        "sentence_scores"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IPdrEYTBxwUe",
        "outputId": "07f8e178-1d47-4bcd-f0ef-9854376e578c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{ : 0.09090909090909091,\n",
              " There are boardly two types of extractive summarization tasks depending on what the summarization program focuses on.: 2.818181818181818,\n",
              " The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection, whether documents, or sets of images, or videos, news stories etc.: 4.090909090909091,\n",
              " The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query.: 3.909090909090909,\n",
              " Common systems are able to create both query relevant text summaries and generic machine generated summaries depending on what the user needs.: 2.090909090909091,\n",
              " An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document.: 3.9999999999999996,\n",
              " Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents, for example, a cluster of articles on the same topic.: 2.545454545454545,\n",
              " This problem is called multi-document summarization.: 1.8181818181818183,\n",
              " A related application is summarizing news articles.: 1.0909090909090908,\n",
              " Imagine a system, which automatically pulls together news articles on a given topic, from the web, and can precisely represents the latest news as a summary.: 2.727272727272727,\n",
              " Image collection summarization is another application example of automatic summarization.: 2.909090909090909,\n",
              " It consists in selecting a representative set of images from a larger set of images.: 1.5454545454545454,\n",
              " 3. A summary in this context is useful to show the most representative images of results in an image collection exploration system.: 2.0,\n",
              " Video summarization is a related domain, where the system automatically creates a trailer of a long video.: 2.2727272727272725,\n",
              " This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions.: 1.1818181818181817,\n",
              " Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.: 1.4545454545454544}"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from heapq import nlargest"
      ],
      "metadata": {
        "id": "WlqiXd7pxyWH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "select_length = int(len(sentence_tokens)*0.3)\n",
        "select_length"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "siuThcS2x2H2",
        "outputId": "2aac72cf-c163-4052-f54e-0ae01662c3dc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "4"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)\n",
        "summary"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rV8xWtfjx3zx",
        "outputId": "8fbc6fba-b40d-48e0-d636-aff95bd9c2df"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection, whether documents, or sets of images, or videos, news stories etc.,\n",
              " An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document.,\n",
              " The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query.,\n",
              " Image collection summarization is another application example of automatic summarization.]"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "final_summary = [word.text for word in summary]\n",
        "summary = ' '.join(final_summary)"
      ],
      "metadata": {
        "id": "Cov692S6x5QK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(text)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-eQuu366x7DO",
        "outputId": "4ebf7c20-f2f1-477e-9f3c-93edd50e883b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " There are boardly two types of extractive summarization tasks depending on what the summarization program focuses on. The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection, whether documents, or sets of images, or videos, news stories etc. The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Common systems are able to create both query relevant text summaries and generic machine generated summaries depending on what the user needs. An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. Sometimes one might be interested in generating a summary from a single source document, while others can use multiple source documents, for example, a cluster of articles on the same topic. This problem is called multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically pulls together news articles on a given topic, from the web, and can precisely represents the latest news as a summary. Image collection summarization is another application example of automatic summarization. It consists in selecting a representative set of images from a larger set of images. 3. A summary in this context is useful to show the most representative images of results in an image collection exploration system. Video summarization is a related domain, where the system automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important and suspicious activity, while ignoring all the boring and redundant frames captured.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(summary)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UIVfVpB2x8mH",
        "outputId": "3da50117-eded-44f8-a784-18cd314fb6f9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection, whether documents, or sets of images, or videos, news stories etc. An example of a summarization problem is document summarization, which attempts to automatically produce an abstract from a given document. The second is query relevant summarization, sometimes called query-based summarization, which summarizes objects specific to a query. Image collection summarization is another application example of automatic summarization.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "len(text)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B1J3_QABx-hP",
        "outputId": "2859ff83-c7cc-449e-f84f-e543790d2732"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1861"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "len(summary)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OuPaa9xHyCgT",
        "outputId": "05867d86-61d5-4b14-b3d9-d490417c292d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "542"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "8RdYrVBNyD4E"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
