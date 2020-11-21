import numpy as np
import load_data
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import collections

train_folder = "../../datasets/train-articles"
dev_folder = "../../datasets/dev-articles"
train_tc_labels = "../../datasets/train-task2-TC.labels"
dev_template_labels_file = "../../datasets/dev-task-TC-template.out"
# output_file = '../../output/lr-output-TC.txt'


def read_from_file():
    train_articles, _ = load_data.read_articles_from_folder(train_folder)
    train_articles_id, train_span_starts, train_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(
        train_tc_labels)
    dev_articles, _ = load_data.read_articles_from_folder(dev_folder)
    dev_articles_id, dev_span_starts, dev_span_ends, dev_gold_labels = load_data.read_predictions_from_tc_label(
        dev_template_labels_file)
    return train_articles, train_articles_id, train_span_starts, train_span_ends, train_gold_labels,dev_articles, dev_articles_id, dev_span_starts, dev_span_ends, dev_gold_labels




def generate_raw_train_data():
    articles,_ = load_data.read_articles_from_folder(train_folder)
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(
        train_tc_labels)
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    final_t = []
    final_l = []
    for id, s, e, l in zip(ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):

        text = articles[id][int(s):int(e)]
        final_t.append(text.strip())
        if l == 'Appeal_to_Authority':
            final_l.append(0)
        elif l == 'Appeal_to_fear-prejudice':
            final_l.append(1)
        elif l == 'Bandwagon,Reductio_ad_hitlerum':
            final_l.append(2)
        elif l == 'Black-and-White_Fallacy':
            final_l.append(3)
        elif l == 'Causal_Oversimplification':
            final_l.append(4)
        elif l == 'Doubt':
            final_l.append(5)
        elif l == 'Exaggeration,Minimisation':
            final_l.append(6)
        elif l == 'Flag-Waving':
            final_l.append(7)
        elif l == 'Loaded_Language':
            final_l.append(8)
        elif l == 'Name_Calling,Labeling':
            final_l.append(9)
        elif l == 'Repetition':
            final_l.append(10)
        elif l == 'Slogans':
            final_l.append(11)
        elif l == 'Thought-terminating_Cliches':
            final_l.append(12)
        elif l == 'Whataboutism,Straw_Men,Red_Herring':
            final_l.append(13)
    pd.DataFrame(list(zip(final_l,final_t)),columns=['label','text']).to_csv('data/train.tsv',index=False,sep='\t')
    import collections
    print(collections.Counter(final_l))


def generate_tech_folders():
    articles, _ = load_data.read_articles_from_folder('../datasets/train-articles')
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(
        '../datasets/train-task2-TC.labels')
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    with open('data/fulltrain.txt','w') as file:
        for id, s, e, l in zip(ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):
            text = articles[id][int(s):int(e)]
                # file.write(id+'\t'+articles[id][int(s)-10:int(e)+10]+'\t'+s+'\t'+e+'\t'+text+'\n')
            file.write(id + '\t'+l+'\t' + s + '\t' + e + '\t' + text + '\n')

    # for id, s, e, l in zip(ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):
    #
    #     text = articles[id][int(s):int(e)]
    #     final_t.append(text.strip())
    #     with open('data/tech/'+l+'.txt','a') as file:
    #         # file.write(id+'\t'+articles[id][int(s)-10:int(e)+10]+'\t'+s+'\t'+e+'\t'+text+'\n')
    #         file.write(id + '\t' + s + '\t' + e + '\t' + text + '\n')




def generate_raw_dev_data():
    articles,_ = load_data.read_articles_from_folder(dev_folder)
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(
        dev_template_labels_file)
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    final_t = []
    final_l = []
    for id, s, e, l in zip(ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):

        text = articles[id][int(s):int(e)]
        final_t.append(text.strip())
        final_l.append(0)

    pd.DataFrame(list(zip(final_l,final_t)),columns=['label','text']).to_csv('bert/data/test.tsv', index=False, sep='\t')


def transform_data(type):
    lb = LabelBinarizer()
    labels = pd.read_csv("../data/"+type+".tsv",header=None,sep='\t')[0]
    print(collections.Counter(labels))
    lb.fit(labels)
    Y = lb.transform(labels)
    X = np.array(np.load("../semantic/"+type+".npy"))
    return X,Y



def generate_prediction(predictions,outputfile):

    article,_ = load_data.read_articles_from_folder(dev_folder)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels =load_data.read_predictions_from_tc_label(dev_template_labels_file)
    # testresult = pd.read_csv('datasets/tc/tc_emo_weight.tsv',sep='\t',header=None).values
    # predictions = []
    # for t in testresult:
    #     predictions.append(np.argmax(t))
    # print(predictions)
    final_prediction=[]
    for l in predictions:
        if l==0:
            final_prediction.append('Appeal_to_Authority')
        elif l == 1:
            final_prediction.append('Appeal_to_fear-prejudice')
        elif l==2:
            final_prediction.append('Bandwagon,Reductio_ad_hitlerum')
        elif l==3:
            final_prediction.append('Black-and-White_Fallacy')
        elif l==4:
            final_prediction.append('Causal_Oversimplification')
        elif l==5:
            final_prediction.append('Doubt')
        elif l==6:
            final_prediction.append('Exaggeration,Minimisation')
        elif l==7:
            final_prediction.append('Flag-Waving')
        elif l==8:
            final_prediction.append('Loaded_Language')
        elif l==9:
            final_prediction.append('Name_Calling,Labeling')
        elif l==10:
            final_prediction.append('Repetition')
        elif l==11:
            final_prediction.append('Slogans')
        elif l==12:
            final_prediction.append('Thought-terminating_Cliches')
        elif l==13:
            final_prediction.append('Whataboutism,Straw_Men,Red_Herring')


    with open(outputfile, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, final_prediction, dev_span_starts, dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
    print("Predictions written to file " + outputfile)

    with open('../../output/fullprediction.txt', "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, final_prediction, dev_span_starts, dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end, article[article_id][int(span_start):int(span_end)]))
    print("Predictions written to file " + '../../output/fullprediction.txt')




def missing_voc():
    import nltk
    articles, _ = load_data.read_articles_from_folder('../datasets/train-articles')
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(
        '../datasets/train-task2-TC.labels')
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    train_t = []
    for id, s, e, l in zip(ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):
        text = articles[id][int(s):int(e)]
        train_t.extend(nltk.word_tokenize(text.lower()))

    articles, _ = load_data.read_articles_from_folder('../datasets/dev-articles')
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = load_data.read_predictions_from_tc_label(
        '../datasets/dev-task-TC-template.out')
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    dev_t = []
    for id, s, e, l in zip(ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels):
        text = articles[id][int(s):int(e)]
        dev_t.extend(nltk.word_tokenize(text.lower()))
    print(set(dev_t)-set(train_t))




if __name__ == '__main__':

    # generate_tech_folders()
    missing_voc()


