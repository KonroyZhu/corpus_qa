from segment import get_content

def check_question(question,inter_list):
    res=False
    for w in inter_list:
        if w in question:
            res=True
            break
    return res

def classify(question,interrogative):
    flag=list(interrogative.keys())
    for i in range(len(flag)):
        li=interrogative[flag[i]]
        if check_question(question,li):
            return flag[i]

def all_interrogative(interrogative):
    all=[]
    for key in interrogative.keys():
        for w in interrogative[key]:
            all.append(w)
    return all

interrogative={
    'MISC':['哪一年','何时','什么时候'],
    'GPE':['哪里','何地'],
    'ORGANIZATION':['哪家'],
    'PERSON':['谁'],
    'NUMBER':['多少','多']
}


if __name__ == '__main__':
    questions=get_content("raw/quest.xml",'text')
    for q in questions:
        print(classify(q,interrogative))