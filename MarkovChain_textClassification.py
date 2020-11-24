# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:26:50 2020

@author: Anna
"""

from collections import defaultdict 
import csv
from lxml import html
import re
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt

usecaseNameDict = defaultdict(list)
usecaseDescDict = defaultdict(list)
usecaseCount = defaultdict(int)

with open("UC_formData.csv", "r", encoding = "utf-8") as file:
    data = csv.reader(file)
    for idx, line in enumerate(data):
        adminid, usecase, formname, formdesc = line
        if idx == 0:
            continue
        else:
            usecaseCount[usecase] += 1
            #strip html from description text
            if len(formdesc) > 0:
                formdesc = str(html.fromstring(formdesc).text_content()).lower()
            formname = formname.lower()
            #split name and description into list of words and punctuation
            formname = re.findall(r"[\w']+|[.,!?;]", formname)
            formdesc = re.findall(r"[\w']+|[.,!?;]", formdesc)
            usecaseNameDict[usecase].extend(formname)
            usecaseDescDict[usecase].extend(formdesc)
        
class FirstOrderMarkovModel(object):
    """
    Implements a bigram model for text generation and classification

    Attributes:
        terminal_characters: a set of punctuation that shouldn't fall at the
                             beginning of a sentence, but should end one.

    Methods:
        build_transition_matrices: builds a dictionary of word frequency
        generate_phrase: generates a phrase by randomly sampling from 
                         word frequencies
        compute_log_likelihood: computes the logarithm of the likelihood
                                for a specified phrase

    """


    terminal_characters = ['.','?','!']

    def __init__(self,sequence):
        """
        sequence: an ordered list of words corresponding to the training set
        """

        self.order=1
        self.sequence = sequence
        self.sequence_length = len(sequence)
        self.transitions = [{}]
        for i in range(self.order):
            self.transitions.append({})
        

    def build_transition_matrices(self):
        """
        Builds a set of nested dictionaries of word probabilities
        """

        for i in range(self.sequence_length):
            word = self.sequence[i]
            if word in self.transitions[0]:
                self.transitions[0][word] += 1
            else:
                self.transitions[0][word] = 1

        transition_sum = float(sum(self.transitions[0].values()))
        for k,v in self.transitions[0].items():
            self.transitions[0][k] = v/transition_sum

        for i in range(self.sequence_length-1):
            word = self.sequence[i]
            next_word = self.sequence[i+1]
            if word in self.transitions[1]:
                if next_word in self.transitions[1][word]:
                    self.transitions[1][word][next_word] += 1
                else:
                    self.transitions[1][word][next_word] = 1
            else:
                self.transitions[1][word] = {}
                self.transitions[1][word][next_word] = 1

        for k_1,tdict in self.transitions[1].items():
            key_sum = float(sum(self.transitions[1][k_1].values()))
            for k_2,v in tdict.items():
                self.transitions[1][k_1][k_2] = v/key_sum 

    def generate_phrase(self):
        """
        Take a random sample from the probability distribution.  Terminate
        when a period, question mark, or exclamation point is drawn.
        """

        w_minus_1 = '?'
        while w_minus_1 in self.terminal_characters:
            w_minus_1 = np.random.choice([*self.transitions[0].keys()],replace=True,p=[*self.transitions[0].values()])
        phrase = w_minus_1+' '
        while w_minus_1 not in self.terminal_characters:
            w_minus_1 = np.random.choice([*self.transitions[1][w_minus_1].keys()],replace=True,p=[*self.transitions[1][w_minus_1].values()])
            phrase += w_minus_1+' '
        return phrase

    def compute_log_likelihood(self,phrase,lamda=0.0,unknown_probability=1e-5):
        """
        Return the log-probability of a given phrase (entered as a string)
        lambda: regularization factor for unseen transitions
        unknown_probability: probability mass of a word not in the dictionary.
        """

        words_in = phrase.split()
        
        w_i = words_in[0]
        try: 
            log_prob = np.log(self.transitions[0][w_i])
        except KeyError:
            log_prob = np.log(unknown_probability)
        for w in words_in[1:]:
            try:
                fjk = 0
                if w in self.transitions[1][w_i]:
                    fjk = self.transitions[1][w_i][w]
                log_prob += np.log((1-lamda)*fjk + lamda*self.transitions[0][w])
                w_i = w
            except KeyError:
                log_prob += np.log(unknown_probability)
        return log_prob

#create transition matrix for each use case name and description data
nameModelDict = {}
descModelDict = {}        

for uc in usecaseNameDict:
    nameModel = FirstOrderMarkovModel(usecaseNameDict[uc])
    nameMatrix = nameModel.build_transition_matrices()
    nameModelDict[uc] = [nameModel, nameMatrix]
    descModel = FirstOrderMarkovModel(usecaseDescDict[uc])
    descMatrix = descModel.build_transition_matrices()
    descModelDict[uc] = [descModel, descMatrix]


testNameDict = defaultdict(list)
testDescDict = defaultdict(list)

#bring in test set
with open("UC_testData.csv", "r", encoding = "utf-8") as file:
    data = csv.reader(file)
    for idx,line in enumerate(data):
        adminid, usecase, formname, formdesc = line
        if idx == 0:
            continue
        else:
            #strip html from description text
            if len(formdesc) > 0:
                formdesc = str(html.fromstring(formdesc).text_content()).lower()
            formname = formname.lower()
            #split name and description into list of words and punctuation
            if adminid in testNameDict:
                adminid = adminid +'b'
            testNameDict[adminid].extend([usecase, formname])
            testDescDict[adminid].extend([usecase, formdesc])    

#calculate log likelihood that each form name and description come from each usecase, then select the max as the predicted label
for form in testNameDict:
    model_fit = {}
    form_text = testNameDict[form][1]
    for uc in nameModelDict:
        model =  nameModelDict[uc][0]
        model_fit[uc]= model.compute_log_likelihood(form_text,lamda=0.01,unknown_probability=1e-10)
    testNameDict[form].append(model_fit)
    v=list(model_fit.values())
    k=list(model_fit.keys())
    best = k[v.index(max(v))]
    testNameDict[form].append(best)

for form in testDescDict:
    model_fit = {}
    form_text = testDescDict[form][1]
    for uc in descModelDict:
        model =  descModelDict[uc][0]
        model_fit[uc]= model.compute_log_likelihood(form_text,lamda=0.01,unknown_probability=1e-10)
    testDescDict[form].append(model_fit)
    v=list(model_fit.values())
    k=list(model_fit.keys())
    best = k[v.index(max(v))]
    testDescDict[form].append(best)

def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.8
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
 
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
 
    ax.set_xticklabels(labels, rotation = 45, ha = 'right')
    ax.set_yticklabels(labels, rotation = 45)

 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

 
usecases = sorted([uc for uc in usecaseNameDict])    
nameActual = [testNameDict[i][0] for i in testNameDict]
namePredict = [testNameDict[i][3] for i in testNameDict]
nameConfusion = confusion_matrix(nameActual, namePredict)
plot_confusion_matrix(nameConfusion, usecases , "nameConfusion.png")

nameAcc = sum([1 for i in testNameDict if testNameDict[i][0] == testNameDict[i][3]])/len(testNameDict)
  
descActual = [testDescDict[i][0] for i in testDescDict]
descPredict = [testDescDict[i][3] for i in testDescDict]
descConfusion = confusion_matrix(descActual, descPredict)
plot_confusion_matrix(descConfusion, usecases , "descConfusion.png")

descAcc = sum([1 for i in testDescDict if testDescDict[i][0] == testDescDict[i][3] ])/len(testDescDict)

#try again with grouped categories to make training data for usecases more equal in size
group1 = ['Residency', 'Internships', 'Scholarships', 'Fellowship', 'Award/Nomination']
group2 = ['Job applications', 'Internal Use', 'Admissions', 'Audition']
group3 = ['Festival or Event', 'Contest', 'Conference', 'Exhibition']

newUCNameDict = defaultdict(list)
newUCDescDict = defaultdict(list)

for uc in usecaseNameDict:
    if uc in group1:
        newUCDescDict['group1'].extend(usecaseDescDict[uc])
        newUCNameDict['group1'].extend(usecaseNameDict[uc])
    elif uc in group2:
        newUCDescDict['group2'].extend(usecaseDescDict[uc])
        newUCNameDict['group2'].extend(usecaseNameDict[uc])
    elif uc in group3:
        newUCDescDict['group3'].extend(usecaseDescDict[uc])
        newUCNameDict['group3'].extend(usecaseNameDict[uc])
    else:
        newUCDescDict[uc].extend(usecaseDescDict[uc])
        newUCNameDict[uc].extend(usecaseNameDict[uc])
        
#create transition matrix for each new group's name and description data
newNameModelDict = {}
newDescModelDict = {}        

for uc in newUCNameDict:
    nameModel = FirstOrderMarkovModel(newUCNameDict[uc])
    nameMatrix = nameModel.build_transition_matrices()
    newNameModelDict[uc] = [nameModel, nameMatrix]
    descModel = FirstOrderMarkovModel(newUCDescDict[uc])
    descMatrix = descModel.build_transition_matrices()
    newDescModelDict[uc] = [descModel, descMatrix]

newTestNameDict = defaultdict(list)
newTestDescDict = defaultdict(list)

#bring in test set
with open("UC_testData2.csv", "r", encoding = "utf-8") as file:
    data = csv.reader(file)
    for idx,line in enumerate(data):
        adminid, usecase, formname, formdesc = line
        if idx == 0:
            continue
        else:
            #strip html from description text
            if len(formdesc) > 0:
                formdesc = str(html.fromstring(formdesc).text_content()).lower()
            formname = formname.lower()
            #split name and description into list of words and punctuation
            if adminid in newTestNameDict:
                adminid = adminid +'b'
            newTestNameDict[adminid].extend([usecase, formname])
            newTestDescDict[adminid].extend([usecase, formdesc])    

#calculate log likelihood that each form name and description come from each usecase, then select the max as the predicted label
for form in newTestNameDict:
    model_fit = {}
    form_text = newTestNameDict[form][1]
    for uc in newNameModelDict:
        model =  newNameModelDict[uc][0]
        model_fit[uc]= model.compute_log_likelihood(form_text,lamda=0.01,unknown_probability=1e-10)
    newTestNameDict[form].append(model_fit)
    v=list(model_fit.values())
    k=list(model_fit.keys())
    best = k[v.index(max(v))]
    newTestNameDict[form].append(best)

for form in newTestDescDict:
    model_fit = {}
    form_text = newTestDescDict[form][1]
    for uc in newDescModelDict:
        model =  newDescModelDict[uc][0]
        model_fit[uc]= model.compute_log_likelihood(form_text,lamda=0.01,unknown_probability=1e-10)
    newTestDescDict[form].append(model_fit)
    v=list(model_fit.values())
    k=list(model_fit.keys())
    best = k[v.index(max(v))]
    newTestDescDict[form].append(best)

newUsecases = sorted([uc for uc in newUCNameDict])    
newNameActual = [newTestNameDict[i][0] for i in newTestNameDict]
newNamePredict = [newTestNameDict[i][3] for i in newTestNameDict]
newNameConfusion = confusion_matrix(newNameActual, newNamePredict)
plot_confusion_matrix(newNameConfusion, newUsecases , "newNameConfusion.png")

newNameAcc = sum([1 for i in newTestNameDict if newTestNameDict[i][0] == newTestNameDict[i][3] ])/len(newTestNameDict)
  
newDescActual = [newTestDescDict[i][0] for i in newTestDescDict]
newDescPredict = [newTestDescDict[i][3] for i in newTestDescDict]
newDescConfusion = confusion_matrix(newDescActual, newDescPredict)
plot_confusion_matrix(newDescConfusion, newUsecases , "newDescConfusion.png")

newDescAcc = sum([1 for i in newTestDescDict if newTestDescDict[i][0] == newTestDescDict[i][3] ])/len(newTestDescDict)