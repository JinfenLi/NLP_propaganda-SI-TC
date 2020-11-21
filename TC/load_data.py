train_folder = "../datasets/train-articles" # check that the path to the datasets folder is correct,
dev_folder = "../datasets/dev-articles"     # if not adjust these variables accordingly
train_tc_labels = "../datasets/train-task2-TC.labels"
train_si_labels = "../datasets/train-task1-SI.labels"
dev_template_labels_file = "../datasets/dev-task-TC-template.out"
task_TC_output_file = "bert-train_sem-TC.txt"
task_SI_output_file = "bert-train_sem-SI.txt"

import glob
import os.path
import codecs

# train_folder/dev_folder
def read_articles_from_folder(folder_name, file_pattern="*.txt"):

    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    sentences = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
        with open(filename, "r",encoding="utf8") as f:
            sentences[article_id] = f.readlines()
    return articles, sentences

# dev_template_labels_file
def read_predictions_from_tc_label(filename):

    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels

def read_predictions_from_si_label(filename):

    articles_id, span_starts, span_ends = ([], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends