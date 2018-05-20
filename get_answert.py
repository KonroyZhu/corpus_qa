import re

import jieba

from question_classify import classify, interrogative, all_interrogative
from rate_entities import ratin_entities, rate_sentence
from retrieve import prepro, get_idf, get_segment, all_query_sim, get_top_5
from nltk.tag import StanfordNERTagger
import nltk

from segment import get_content

def get_entities(tagged_list,oriented):
    entity=[]
    entities=[]
    for i in range(len(tagged_list)):
        tag=tagged_list[i][1]
        ent=tagged_list[i][0]
        if tag == oriented:
            entity.append(ent)
            try:
                next_tag=tagged_list[i+1][1]
            except:
                next_tag="O"
            if not next_tag==oriented:
                entities.append(entity)
                entity=[]
    return entities

if __name__ == '__main__':
    doc, dictionary = prepro()
    #questions = get_segment("question")

    full_document=get_content("raw/doc.xml","TEXT")
    questions = get_content("raw/quest.xml", "text")
    idf = get_idf(dictionary,2)
    sim_list = all_query_sim()
    print(len(sim_list))
    st = StanfordNERTagger(model_filename='/home/konroy/standford/chinese/ner/chinese.misc.distsim.crf.ser.gz',
                           path_to_jar="/home/konroy/standford/core_full/stanford-corenlp-full/stanford-corenlp-3.9.1.jar")

    all_inter=all_interrogative(interrogative)
    for idx in range(len(sim_list)):
        print("question:", questions[idx])
        questions_type=classify(questions[idx],interrogative)
        print("question type:",questions_type)

        question_tagged = st.tag([c for c in jieba.cut(questions[idx])])
        # print(tagged_list)
        question_entities = get_entities(question_tagged, questions_type)

        top_5 = get_top_5(idx)
        for t in top_5[:1]:
            id = t[0]
            print("document:", full_document[id])
            sentence_list=re.split("[。!:?]",str(full_document[id]))
            sentence_idx=rate_sentence(questions[idx],questions_type, sentence_list, idf, dictionary,st)
            best_sentence=sentence_list[sentence_idx]
            print("best_sentence:",best_sentence)

            part_list=re.split("[,]",best_sentence)
            part_idx=rate_sentence(questions[idx],questions_type,part_list,idf,dictionary,st)
            best_part=part_list[part_idx]
            print("best part:",best_part)
            # print(full_document[id])
            tagged_list = st.tag([c for c in jieba.cut(best_part)])
            print("NER:",tagged_list)
            entities = get_entities(tagged_list, questions_type)
            print("candidate entities:",entities)
            best_entity = ratin_entities(questions[idx],question_entities, entities, idf, all_inter, dictionary)
            print("######### answer ##########")
            print("best entity:","".join(best_entity))
            print("######### answer ##########")
        print()
