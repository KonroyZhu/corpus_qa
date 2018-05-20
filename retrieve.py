import os
import pickle
from collections import Counter
from math import log, pow, sqrt
import gensim
import numpy as np
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity


def get_segment(mode="document"):
    """

    :param mode: toggle to switch from "document" to "question"
    :return: segmented questions list or documents list
    """
    path = "data/" + mode + "_seg.pkl"
    print("get segment", mode)
    if not os.path.exists(path):
        documents = pickle.load(open("data/" + mode + "_pair.pkl", "rb"))
        doc = []
        for line in documents:
            d = [tuple(t)[0] for t in line]
            doc.append(d)
        pickle.dump(doc, open(path, "wb"))
    else:
        doc = pickle.load(open(path, "rb"))
    return doc


def prepro():
    """

    :return: return the documents as well as the dictionary
    """
    # documents
    documents = get_segment("document")

    vocab_path = "data/vocab.pkl"
    if not os.path.exists(vocab_path):
        dictionary = corpora.Dictionary(documents)
        pickle.dump(dictionary, open(vocab_path, "wb"))
    else:
        print("load dictionary...")
        dictionary = pickle.load(open(vocab_path, "rb"))
    print(dictionary)
    return documents, dictionary


def word_idf(word):
    """
    calculate the idf for word
    :param word: a word in a sentence
    :return: the word's idf
    """
    # calculate word idf
    string = word + "#"  # for task 1
    sum = 0
    pos = 0
    for i in range(len(doc)):

        d = doc[i]
        if word in d:
            sum += 1
            string += str(i + 1) if pos == 0 else ("~" + str(i + 1))
            pos += 1
    result = log(len(doc) / sum)
    if sum == 0:
        result = 0
    string = string.split("#")
    string = string[0] + "#" + str(sum) + "#" + string[1]
    # print(string)
    return result


def get_idf(dic,option=""):
    """
    calculate and save all the idf from the dictionary
    :return:
    """
    path = "data/idf"+str(option)+".pkl"
    print("get idf...")
    if not os.path.exists(path):
        idf_list = []
        count = 0
        for idx in range(len(dic)):
            print(count / float(len(list(dictionary))))
            count += 1
            word = dictionary[idx]
            idf_list.append(word_idf(word))
        pickle.dump(idf_list, open(path, "wb"))
    else:
        idf_list = pickle.load(open(path, "rb"))
    return idf_list


def weight(t, idf):
    """

    :param t: term frequency
    :param idf: inverse document frequency
    :return: weight concerning both tf and idf
    """
    # print("terms",t+1)
    # print("idf:",idf)
    # print("log(terms+1):",np.log(1 + t))
    return  np.log(1 + t)*idf # too small


def doc2vec(Doc,idf,dictionary):
    """
    convert a document to tfidf weight vector
    :param Doc:
    :return:
    """
    # print("convert document to vector...")
    idf_vec = np.array(idf)  # length 73000
    freq_vec = [0] * len(idf)  # length 73000 TODO: change to len(idf)
    bow = dictionary.doc2bow(Doc)
    idx_list = [x[0] for x in bow]
    fre_list = [x[1] for x in bow]
    for i in range(len(idx_list)):
        # print(idx_list[i])
        freq_vec[idx_list[i]] = fre_list[i]
    freq_vec = np.array(freq_vec)
    weight_vec = weight(freq_vec, idf_vec)
    return weight_vec


def cos_sim(a, b):
    """
    calculate the cosine similarity between vector a and b
    :param a:
    :param b:
    :return:
    """
    # calculate cosine similarity between vector a b
    numerator = 0
    denominator = 0
    _a = 0
    _b = 0
    for i in range(len(a)):
        numerator += a[i] * b[i]
        _a += a[i] * a[i]
        _b += b[i] * b[i]
    sqr_a = sqrt(_a)
    sqr_b = sqrt(_b)
    denominator = sqr_a * sqr_b
    return numerator / denominator


def conver_part_doc(documents, part):
    path = "data/doc_vec" + part + ".pkl"
    if not os.path.exists(path):
        vec_list = []
        print(len(documents))
        for doc in documents:
            print(doc)
            vec = doc2vec(doc,idf,dictionary)
            print(len(vec))
            vec_list.append(vec)
        pickle.dump(vec_list, open(path, "wb"))
        print("part " + part + "  done !")
    else:
        vec_list = pickle.load(open(path, "rb"))
    return vec_list


def get_all_sim(query):
    conver_part_doc(doc[:1000], "0")
    conver_part_doc(doc[1000:2000], "1")
    conver_part_doc(doc[2000:3000], "2")
    conver_part_doc(doc[3000:4000], "3")
    conver_part_doc(doc[4000:5000], "4")
    conver_part_doc(doc[5000:], "5")

    query_vec = doc2vec(query,idf,dictionary)
    sim_list = []
    for i in range(6):
        vec = pickle.load(open("data/doc_vec" + str(i) + ".pkl", "rb"))

        for v in vec:
            # sim=cos_sim(query_vec,v)
            sim = cosine_similarity([query_vec], [v])[0][0]
            print(sim)
            sim_list.append(sim)

    return sim_list


def all_query_sim():
    path = "data/all_query_sim.pkl"
    if not os.path.exists(path):
        sim_list = []
        for query in questions:
            l = get_all_sim(query)
            print(l)
            sim_list.append(l)
            print(len(l))
        pickle.dump(sim_list, open(path, "wb"))
        print("similarity saved !")
    else:
        sim_list = pickle.load(open(path, "rb"))
    return sim_list


def get_top_5(question_id):
    sim_list = all_query_sim()
    question_sim = sim_list[question_id]

    sim_dict = {}
    for i in range(len(question_sim)):
        sim_dict[i] = question_sim[i]

    top_5_sort = sorted(sim_dict.items(), key=lambda d: d[1], reverse=True)[:5]# top_5_sort contain a li

    return top_5_sort


if __name__ == '__main__':
    doc, dictionary = prepro()
    idf = get_idf(dictionary,2)
    questions = get_segment("question")

    sim_list = all_query_sim()
    print(len(sim_list))

    for idx in range(len(sim_list)):
        print("question:", questions[idx])
        print(sim_list[idx])
        max=np.argmax(sim_list[idx])
        print(max,sim_list[idx][max])
        top_5 = get_top_5(idx)
        for t in top_5:
            id=t[0]
            print("document:",doc[id])
