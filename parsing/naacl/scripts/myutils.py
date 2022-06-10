import os

targets = ['UD_Ancient_Greek-PROIEL', 'UD_Arabic-PADT', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_Chinese-GSD', 'UD_Hebrew-HTB', 'UD_Korean-GSD', 'UD_Russian-GSD', 'UD_Swedish-Talbanken']
treebank_mlms = [['pranaydeeps/Ancient-Greek-BERT', 'nlpaueb/bert-base-greek-uncased-v1'], ['aubmindlab/bert-base-arabertv02', 'asafaya/bert-base-arabic'], ['bert-base-uncased', 'roberta-large', 'distilbert-base-uncased'], ['TurkuNLP/bert-base-finnish-uncased-v1', 'TurkuNLP/bert-base-finnish-cased-v1'], ['bert-base-chinese', 'hfl/chinese-bert-wwm-ext', 'hfl/chinese-roberta-wwm-ext'], ['onlplab/alephbert-base'], ['klue/bert-base','klue/roberta-large', 'kykim/bert-kor-base'], ['cointegrated/rubert-tiny', 'sberbank-ai/ruRoberta-large', 'blinoff/roberta-base-russian-v0'], ['KB/bert-base-swedish-cased']]
multi_mlms = ['bert-base-multilingual-cased', 'xlm-roberta-base', 'google/rembert']

seeds = ['692', '710', '932']

def getTrainDev(treebank):
    train = ''
    dev = ''
    path = 'data/' + treebank + '/'
    for conlFile in os.listdir(path):
        if conlFile.endswith('conllu'):
            if 'train' in conlFile:
                train = path + conlFile
            if 'dev' in conlFile:
                dev = path + conlFile
    return train, dev


def getModel(name):
    modelDir = 'mtp/logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.tar.gz'
            if os.path.isfile(modelPath):
                return modelPath
    return ''



