# propaganda-SI-TC

*SEMEVAL 2020 TASK 11
"DETECTION OF PROPAGANDA TECHNIQUES IN NEWS ARTICLES"*
> task link: https://propaganda.qcri.org/semeval2020-task11/index.html

> check our paper: https://arxiv.org/pdf/2008.10163.pdf

## Background
We refer to propaganda whenever information is purposefully shaped to foster a predetermined agenda. Propaganda uses psychological and rhetorical techniques to reach its purpose. Such techniques include the use of logical fallacies and appealing to the emotions of the audience. Logical fallacies are usually hard to spot since the argumentation, at first sight, might seem correct and objective. However, a careful analysis shows that the conclusion cannot be drawn from the premise without the misuse of logical rules. Another set of techniques makes use of emotional language to induce the audience to agree with the speaker only on the basis of the emotional bond that is being created, provoking the suspension of any rational analysis of the argumentation. All of these techniques are intended to go unnoticed to achieve maximum effect.

## Our Approach
This paper describes the BERT-based models proposed for two subtasks in SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles. We first build the model for Span
Identification (SI) based on SpanBERT, and facilitate the detection by a deeper model and a sentence-level representation. We then develop a hybrid model for the Technique Classification (TC). The hybrid model is composed of three submodels including two BERT models with different training methods, and a feature-based Logistic Regression model. We endeavor to deal with imbalanced dataset by adjusting cost function. We are in the seventh place in SI subtask (0.4711 of F1-measure), and in the third place in TC subtask (0.6783 of F1-measure) on the development set.

## How to Run
1. add datasets folder from SEMEVAL 2020 TASK 11 "DETECTION OF PROPAGANDA TECHNIQUES IN NEWS ARTICLES"
2. add pytorch_pretrained_bert from https://github.com/facebookresearch/SpanBERT/tree/master/code
3. add spanbert_hf_base from https://github.com/facebookresearch/SpanBERT
4. run process_data.py to generate train and dev dataset for SI and TC separetely
5. create a folder 'pro_output' to store the fine-tuned model
6. run SI+SpanBERT, SI+BERT, TC+SpanBERT, TC+BERT 
--model
spanbert_hf_base
--output_dir
pro_output
--train_file
datasets/sI/train.json
--test_file
datasets/sI/dev.json
--do_test
--version_2_with_negative
--dev_file
datasets/sI/train.json
--do_train

