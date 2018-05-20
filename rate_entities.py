import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance

from question_classify import all_interrogative, interrogative
from retrieve import prepro, doc2vec, get_idf
from segment import get_content


def jarccard(a, b):
    # if type(a)==list:
    #     a="".join(a)
    # if type(b) == list:
    #     b = "".join(b)
    distance = 0
    inter=0
    union=len(set(a))
    for item in a:
        if item in b:
            inter+=1
    for item in a:
        if not item in b:
            union+=1
    distance=float(inter)/union
    return distance

def is_duplicate(entity,entity_list):
    flag=False
    for e in entity_list:
        # print(e,entity)
        if jarccard(entity,e) > 0.5:
            flag=True
            break
    return flag

def replace_interrogative(entity, doc, interrogative):
    new_doc = [""] * len(doc)
    for i in range(len(doc)):
        if doc[i] in interrogative:
            for e in entity:
                new_doc.append(e)
        else:
            new_doc.append(doc[i])  # TODO:change to append as the entity is a list
    return new_doc


def ratin_entities(question, question_entities, entity_list, idf, interrogative, dictionary):
    doc = [c for c in jieba.cut(question)]
    question_vec = doc2vec(doc, idf, dictionary)
    # print(doc)
    # for v in question_vec:
    #     if not v == 0.0:
    #         print(v)
    mark = ""
    max = 0
    print("question entities:", question_entities)
    for entity in entity_list:
        is_dup=is_duplicate(entity,question_entities)
        print("is duplicate",entity,is_dup)
        if not is_dup:
            print("selecting")
            new_doc = replace_interrogative(entity, doc, interrogative)
            new_vec = doc2vec(new_doc, idf, dictionary)

            sim = cosine_similarity([new_vec], [question_vec])
            # print("vec", question_vec)
            # print("new vec", new_vec)
            # print(sim,entity)
            if sim > max:
                max = sim
                mark = entity
    return mark


def rate_sentence(question,questions_type, sentence_list, idf, dictionary,st):
    sim_list = []

    question_seg = [c for c in jieba.cut(question)]
    question_vec = doc2vec(question_seg, idf, dictionary)
    for sentence in sentence_list:
        sentence_seg = [c for c in jieba.cut(sentence)]
        sentence_vec = doc2vec(sentence_seg, idf, dictionary)
        sim = cosine_similarity([sentence_vec], [question_vec])
        sim_list.append(sim)
    return np.argmax(sim_list)


if __name__ == '__main__':
    """
    doc, dictionary = prepro()
    idf = get_idf(dictionary,2)
    questions = get_content("raw/quest.xml", "text")
    all=all_interrogative(interrogative)
    for q in questions[:1]:
        best_entity=ratin_entities(q, [['第一位'], ['墨西哥', '墨西哥'], ['第一位', '墨西哥']], idf, all, dictionary)
        print(best_entity)
    #"""


    # print(jarccard(['中国科学院', '化学', '研究所'], ['海尔', '科化', '工程塑料', '研究', '中心']))
    # print(jarccard(['海尔集团'] ,['海尔', '科化', '工程塑料', '研究', '中心']))
    # print(edit_distance(['a','b','d'],['a','b','d']))
    # print(edit_distance("add","adc"))