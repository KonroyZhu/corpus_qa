import pickle
import re
from jieba import posseg


def get_content(path="quest.xml", tag="text"):
    content = "".join(open(path, "r").readlines()).replace("\n", "")
    content = re.findall("<" + tag + ">(.*?)</" + tag + ">", content)
    return content


if __name__ == '__main__':
    quesions = get_content("raw/quest.xml", "text")
    q_id = get_content("raw/quest.xml", "id")
    documents = get_content("raw/doc.xml", "TEXT")
    question_list = []
    document_list = []
    stop_words = [c.replace("\n", "") for c in open("raw/stop.txt").readlines()]
    print(stop_words)
    for q in quesions:

        question_list.append([c for c in posseg.cut(q) if not tuple(c)[0] in stop_words ])

    print("question ok")
    for d in documents:
        document_list.append([c for c in posseg.cut(d) if not tuple(c)[0] in stop_words ])
    print("document ok")

    with open("data/question_pair.pkl", "wb") as f:
        pickle.dump(question_list, f)
        print("questions saved")

    with open("data/document_pair.pkl", "wb") as f:
        pickle.dump(document_list, f)
        print("documents saved")
