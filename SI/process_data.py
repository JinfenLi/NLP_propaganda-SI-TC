import json
import collections
import nltk
import load_data

train_folder = "../datasets/train-articles"
dev_folder = "../datasets/dev-articles"
train_si_labels = "../datasets/train-task1-SI.labels"
dev_template_labels_file = "../datasets/dev-task-TC-template.out"
output_file = '../output/spanbert-output-SI.txt'

def mergespan(span):
    newspan=[]
    span.sort()
    for s,e in span:
        if not newspan:
            newspan.append((s,e))
        else:
            if s<=newspan[-1][1]+1:
                pas=newspan.pop()
                newspan.append((pas[0],max(e,pas[1])))
            else:
                newspan.append((s,e))
    newspan.sort()
    return newspan

def generate_train_data():
    articles, sentence = load_data.read_articles_from_folder(train_folder)

    ref_articles_id, ref_span_starts, ref_span_ends = load_data.read_predictions_from_si_label(train_si_labels)
    data = []
    concats = []
    article_span = collections.defaultdict(list)
    for id, s, e in zip(ref_articles_id, ref_span_starts, ref_span_ends):
        article_span[id].append((int(s), int(e)))

    for articleid, span in article_span.items():
        span=mergespan(span)
        article_d = {}
        article_d['article_id'] = articleid
        sentences = sentence[articleid]
        qid = 0
        span.sort()
        sent_id=0
        span_id=0
        cur_s=0
        prev_span_e=0
        qas=[]

        while sent_id<len(sentences):
            # start = 0
            qa={}
            cur_e = cur_s + len(sentences[sent_id])
            if span_id<len(span):

                if span[span_id][0]>=cur_e:
                    concat=sentences[sent_id]
                    is_impossible = True
                    answer=''
                    start=-1
                    sent_id += 1
                    cur_s = cur_e
                    prev_span_e = cur_e
                    qa['context'] = concat
                    qa['qid'] = articleid + '_' + str(qid)
                    qa['answer'] = answer
                    qa['start'] = start
                    qa['is_impossible'] = is_impossible
                    if concat not in ['\n', '\ufeff\n', '\u200f\n']:
                        qas.append(qa)
                        concats.append(concat)
                else:
                    if span_id+1<len(span):
                        if span[span_id][1]<=cur_e:
                            next_span_s = min(span[span_id+1][0],cur_e)
                        else:

                            next_span_s = span[span_id][1]
                            while span[span_id][1]>cur_e:
                                sent_id+=1
                                cur_e+=len(sentences[sent_id])
                    else:
                        next_span_s = max(cur_e+len(sentences[sent_id]),span[span_id][1])
                        if sent_id+1<len(sentences):
                            sent_id+=1
                            next_span_s = max(cur_e + len(sentences[sent_id]), span[span_id][1])
                    concat=articles[articleid][prev_span_e:next_span_s]
                    start=span[span_id][0]
                    answer = articles[articleid][start:span[span_id][1]]
                    is_impossible = False
                    qa['context'] = concat
                    qa['qid'] = articleid + '_' + str(qid)
                    qa['answer'] = answer
                    qa['start'] = start-prev_span_e
                    qa['is_impossible'] = is_impossible
                    if concat not in ['\n', '\ufeff\n', '\u200f\n']:
                        qas.append(qa)
                        concats.append(concat)
                    if span_id+1<len(span):
                        if span[span_id+1][0]>=cur_e:
                            sent_id+=1
                            cur_s = cur_e
                            prev_span_e = max(cur_s,span[span_id][1])
                        else:
                            prev_span_e = span[span_id][1]
                    else:
                        break
                    span_id += 1
            else:
                break
            qid += 1
        sent_id+=1
        qid += 1
        while sent_id < len(sentences):
            qa={}
            concat = sentences[sent_id]
            is_impossible = True
            answer = ''
            start = -1
            qa['context'] = concat
            qa['qid'] = articleid + '_' + str(qid)
            qa['answer'] = answer
            qa['start'] = start
            qa['is_impossible'] = is_impossible
            if concat not in ['\n', '\ufeff\n', '\u200f\n']:
                qas.append(qa)
                concats.append(concat)
            sent_id += 1
            qid+=1
        article_d['qas'] = qas
        data.append(article_d)
    # 160
    print(max(len(nltk.word_tokenize(c)) for c in concats))
    # 16719
    print(len(concats))
    with open('data/train.json', 'w') as file:
        json.dump({'data': data}, file)


def generate_dev_data():
    articles, sentence = load_data.read_articles_from_folder(dev_folder)
    # ref_articles_id, ref_span_starts, ref_span_ends = read_predictions_from_file(task_SI_output_file)
    data = []
    for id in articles.keys():
        id=str(id)
        article_d = {}
        article_d['article_id'] = id
        sentences = sentence[id]
        qas = []

        qid = 0
        for i in range(0, len(sentences), 2):

            concat = ''.join(sentences[i:i + 2])
            qa = {}
            qa['context'] = concat
            qa['qid'] = str(id) + '_' + str(qid)
            qa['answer'] = ''
            qa['start'] = -1
            qa['is_impossible'] = True
            if concat not in ['\n', '\ufeff\n', '\u200f\n','\n\n']:
                qas.append(qa)

            qid += 1

        article_d['qas'] = qas
        data.append(article_d)

    with open('data/dev.json', 'w') as file:
        json.dump({'data': data}, file)

def generate_goldlabel():
    articles, sentence = load_data.read_articles_from_folder(dev_folder)
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(dev_template_labels_file)
    gl = collections.defaultdict(list)
    for id, s, e in zip(ref_articles_id, ref_span_starts, ref_span_ends):
        gl[id].append((int(s), int(e)))

    total_span = 0
    newspa = []
    for k in gl.keys():
        m = mergespan(gl[k])
        total_span += len(m)
        # print(k)
        # print(m)
        for mm in m:
            newspa.append((k, mm[0], mm[1],articles[k][mm[0]:mm[1]]))
    with open('data/goldlable.txt', "w") as fout:
        for i, s, e,t in newspa:
            fout.write("%s\t%s\t%s\t%s\n" % (i, s, e,t))
    print(total_span)


def generate_prediction():
    articles, sentence = load_data.read_articles_from_folder(dev_folder)
    spa = []
    with open('data/predictions.json','r') as file:
        data=json.load(file)
    nd=collections.defaultdict(list)
    for k in data.keys():
        nd[k.split('_')[0]].append(k)
    for id in articles.keys():
        id=str(id)
        qid=0
        sentences = sentence[id]
        start=0
        for i in range(0, len(sentences), 2):
            concat = ''.join(sentences[i:i + 2])
            idd=id+'_'+str(qid)
            if (idd in nd[id]) and data[idd]!='':
                s=concat.find(data[idd])+start
                e=s+len(data[idd])

                spa.append((id,s,e))
            qid+=1
            start+=len(concat)
    with open(output_file, "w") as fout:
        for i,s,e in spa:
            fout.write("%s\t%s\t%s\n" % (i, s, e))

if __name__ == '__main__':
    generate_train_data()

