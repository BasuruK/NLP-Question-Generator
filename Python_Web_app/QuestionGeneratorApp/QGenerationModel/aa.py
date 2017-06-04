import string
from nltk.corpus import stopwords
from nltk.tag import CRFTagger
import pycrfsuite
from nltk import word_tokenize, sent_tokenize
import pandas as pd
import numpy as np
import spacy
import time

# Import PyTorch Framework
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

ct = CRFTagger()
nlp = spacy.load('en')


# Read in the train set data
# train_set  = pd.read_csv('Train_and_test_data/train.txt',sep=' ',names=['word','Brill','tag']).drop('Brill',1)
# Format train set data to tuples of a list of lists
# train_set = [[tuple(x) for x in train_set.values]]

# Training Code, Keep it commented unless retraining
# ct.train(train_set,'model.crf.tagger')

class QuestionGenerator:

    def generate(self, path_to_file):

        # Read the Test data from txt
        sentence = ' '.join(open(path_to_file, 'r').readlines()).rstrip("\n")

        def process_text(passage):
            """
            1.Remove Punctuations and Special characters appear in book chapters
            2.Remove Stopwords
            3.Return Clean list of words
            """
            exclude_set = set(['“', '”', ':'])

            no_punctuation = [char for char in passage if char not in string.punctuation + "“”."]
            no_punctuation = ''.join(no_punctuation)
            no_punctuation = [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]

            return no_punctuation

        # remove stop words and punctuation
        sentence = process_text(sentence)
        sentence = ' '.join(sentence)
        sentences = sent_tokenize(sentence)
        word_list = [[]]
        word_list.clear()
        wlist = []
        i = 0
        j = 0

        for sent in sentences:
            for word in sent.split():
                wlist.append(word)
                j += 1

            wlist.copy
            word_list.append(wlist.copy())
            wlist.clear()
            i += 1

        # Set the model of previously trained data set
        ct.set_model_file(model_file='model.crf.tagger')
        word_list = ct.tag_sents(word_list)

        # test_set = pd.read_csv('Train_and_test_data/test.txt', sep=' ', names=['words','Brill','tag'], index_col=False).drop(['Brill','tag'],1)
        # test_set = test_set.values.tolist()
        # ct.tag_sents(test_set)

        # test_setEval = pd.read_csv('Train_and_test_data/test.txt', sep=' ', names=['words','Brill','tag']).drop('Brill',1)
        # test_setEval = [[tuple(x) for x in test_setEval.values]]
        # test the accuracy of the POS
        # ct.evaluate(test_setEval)

        # Convert list of tuples to a numpy array
        word_list = np.array(word_list)
        word_list = np.reshape(word_list, (-1, 2))

        # Find S -> NP VP NP
        # Returns if a subject is found
        # pattern variable = [pat0, pat1, pat2]
        def findSubject(pattern):
            # print('{} -> {} {}'.format(pattern[0], pattern[1], pattern[2]))
            if pattern[0] == 'NP' and pattern[1] == 'VP' and pattern[2] == 'NP':
                # if true pattern[0] is the subject of a sentence
                return True
            else:
                return False

        # Extract the Subject -> NP VP NP to find the subjects in a sentence
        # using Markov Chain Model

        pattern = []
        subjects = []
        word_list_length = len(word_list)

        for i in range(0, len(word_list)):
            # print(word_list[i][1].split('-')[1])
            try:
                if ((word_list_length - i == 2)):
                    break
                else:
                    pattern = [word_list[i][1].split('-')[1], word_list[i + 1][1].split('-')[1],
                               word_list[i + 2][1].split('-')[1]]
                    # Call the function findSubject to identify potential subject elements and save them in an array
                    if findSubject(pattern):
                        # If returns true consider 1st element as a potential Subject
                        print('Potential Subject Found at index {}'.format(i))
                        print(
                            'Subject -> {} => {} {}'.format(word_list[i][0], word_list[i + 1][0], word_list[i + 2][0]))
                        # Put the phrases in to a sentece and append it to an array.
                        sub = [word_list[i][0], word_list[i + 1][0], word_list[i + 2][0]]
                        subjects.append(' '.join(sub))
                        print('--------------------')

            except IndexError:
                # Break the loop if IndexError occurs
                # Fail safe
                print("Out of Index")
                break

        # Find PERSONs in the filtered sentence using spaCy NER and build a DataFrame Object based on the data
        # DataFrame structure -> Word | POS_TAG | Person

        # Convert the subjects array to a 2D numpy array
        subjects = np.array(subjects)
        subjects = np.reshape(subjects, (-1, 1))

        refferingDataFrame = pd.DataFrame(columns=('Word_POS-TAG_Person', 'nullColumn'))

        for i in range(0, len(subjects)):
            for j in range(0, len(subjects[i][0].split(" "))):
                wordBag = subjects[i][0].split(" ")
                # Find POS_TAG
                pos = ct.tag([wordBag[j]])

                # Find IF word is Person
                person = [ent.label_ for ent in nlp(pos[0][0]).ents]
                person = ' '.join(person)
                if not person:
                    person = ""

                # Add a row to the DataFrame with the retrived data
                refferingDataFrame.loc[len(refferingDataFrame)] = [
                    (pos[0][0] + "_" + pos[0][1].split('-')[1] + "_" + person).rstrip('_'), ""]

        # # Create the LSTM RNN

        # The RNN is created using PyTorch Framework with Cuda disabled. The RNN will clasify the Question according the the catogiry
        # To enable cuda add .cuda() method to torch.randn() method
        lstm = nn.LSTM(3, 3)
        inputs = [autograd.Variable(torch.randn(1, 3)) for _ in range(5)]
        hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3)))

        for i in inputs:
            out, hidden = lstm(i.view(1, 1, -1), hidden)

        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3)))
        out, hidden = lstm(inputs, hidden)

        def prepare_sequence(seq, to_ix):
            idxs = [to_ix[w] for w in seq]
            tensor = torch.LongTensor(idxs)
            return autograd.Variable(tensor)

        # Read training data from the csv
        # Training data format => [(["Phrase"], ["Question"])]
        unprocessed_data = pd.read_csv('Train_and_test_data/LSTM_train_set.csv', header=None)

        # Transform the data in to a processable format
        t_data_for_lstm = []
        training_data = []
        for phrase in unprocessed_data.itertuples():
            t_data_for_lstm.append(list(zip([[phrase[1]]], [[phrase[2]]])))

        for i in range(0, len(t_data_for_lstm)):
            training_data.append(t_data_for_lstm[i][0])

        word_to_ix = {}
        tag_to_ix = {}

        for sent, tags in training_data:
            for word in sent:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
            for word1 in tags:
                if word1 not in tag_to_ix:
                    tag_to_ix[word1] = len(tag_to_ix)

        EMBEDDING_DIM = 6
        HIDDEN_DIM = 6

        # Create the model
        class LSTMTagger(nn.Module):
            def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
                super(LSTMTagger, self).__init__()
                self.hidden_dim = hidden_dim

                self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
                self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
                self.hidden = self.init_hidden()

            def init_hidden(self):
                return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                        autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

            def forward(self, sentence):
                embeds = self.word_embeddings(sentence)
                lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
                tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
                tag_scores = F.log_softmax(tag_space)
                return tag_scores

        # set the variables to Train the model
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Values Before Training
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        t0 = time.time()
        iterations = 300
        progress_after = iterations / 4

        # Train the RNN for 300 iterations
        for epoch in range(0, iterations):
            for sentence, tags in training_data:
                model.zero_grad()
                model.hidden = model.init_hidden()
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = prepare_sequence(tags, tag_to_ix)
                # Calculate the % of completion
                if (progress_after == epoch):
                    print("Training {}% Completed".format((progress_after / iterations) * 100))
                    progress_after = progress_after + (iterations / 4)

                tag_scores = model(sentence_in)

                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()

        print("Training Complete!\nTotal Training time :", round(time.time() - t0, 2), "s\n")

        # #Values after training
        # inputs = prepare_sequence(training_data[0][0], word_to_ix)
        # tag_scores = model(inputs)

        # print("Values After training")
        # print(tag_scores)


        # Format the result generated by the LSTM by adding the relelevent Noun Phrases
        def generate_questions(model, ref_data_frame):
            question = ""
            sentence = ""
            index = 0
            # for all the entries in the ref_data_frame predict a question
            for index, word_phrase in ref_data_frame.iterrows():

                # Predict the potentian sentence structure that can be used to generate the question.
                try:
                    inputs = prepare_sequence([word_phrase[0]], word_to_ix)
                    tag_scores = model(inputs)
                except KeyError:
                    # ignore
                    pass

                # Take the maximum probalilty, and based on the probabilty find index of value of the dictonary
                maxVal = max(tag_scores.data.numpy()[0])
                index_loc = 0
                probability_tag_scores = tag_scores.data.numpy().ravel()

                for i in range(0, len(probability_tag_scores)):
                    if (probability_tag_scores.ravel()[i] == maxVal):
                        index_loc = i

                # Travers the dictonary and identify the key value based on the probability predicted
                for key, value in tag_to_ix.items():
                    if (value == index_loc):
                        # format the Key to generate a meaningfull question
                        # extract subject and object of the tested sentence
                        sentence = word_phrase[0].split('_')
                        # Check whether the question needs modification
                        if "NP" in sentence:
                            question = key.replace("NP", sentence[0])
                            print(question)
                        elif "VP" in sentence:
                            if "NP" in key and "N1" in key:
                                # Find the complete sentence matching for the verb
                                # Find the index in subjects which matches for verb
                                index = [i for i, j in enumerate(subjects.ravel()) if sentence[0] in j]
                                ref_subject = subjects.ravel()[index][0].split()
                                # Replace for words NP and NP1
                                question = key.replace("NP", ref_subject[0])
                                question = question.replace("N1", ref_subject[-1])
                                print(question)

                            else:
                                # If "VP" but only one "NP"
                                index = [i for i, j in enumerate(subjects.ravel()) if sentence[0] in j]
                                ref_subject = subjects.ravel()[index][0].split()
                                question = key.replace("NP", ref_subject[0])
                                print(question)

            generate_questions(model, refferingDataFrame)


x = QuestionGenerator()
x.generate("/home/basuruk/Desktop/DRF/QuestionGenerationUsingNLP/Text_Passages1.txt")
