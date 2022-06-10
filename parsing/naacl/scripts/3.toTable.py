import conll18_ud_eval
import myutils
import os

def getScores(treebank, mlm, frozen):
    data = []
    _, pathGold = myutils.getTrainDev(treebank)
    for seed in myutils.seeds:
        name = '.'.join([treebank, mlm.replace('/', '-'), seed])
        if frozen:
            name += '.frozen'
        modelPath = myutils.getModel(name)
        if modelPath == '':
            print('Error: ' + name + ' not found' )
            return []
        pathPred = modelPath.replace('model.tar.gz', '') + treebank + '.dev.out'
        goldData = conll18_ud_eval.load_conllu(open(pathGold))
        predData = conll18_ud_eval.load_conllu(open(pathPred))
        scores = conll18_ud_eval.evaluate(goldData, predData)
        data.append(scores['LAS'].f1 *100)
        data.append(scores['UAS'].f1*100)
        data.append(0.0)
    return data

for treebank, lang_mlms in zip(myutils.targets, myutils.treebank_mlms):
    print(treebank)
    for mlm in myutils.multi_mlms + lang_mlms:
        scores = getScores(treebank, mlm, False)
        print('\t'.join([mlm] + [str(x) for x in scores]))
    print()

for treebank, lang_mlms in zip(myutils.targets, myutils.treebank_mlms):
    print(treebank)
    for mlm in myutils.multi_mlms:
        scores = getScores(treebank, mlm, True)
        print('\t'.join([mlm] + [str(x) for x in scores]))
    print()

