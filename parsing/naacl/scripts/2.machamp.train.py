from allennlp.common import Params
import os
import urllib.request, json 
import myutils

def getEmbedSize(name):
    with urllib.request.urlopen("https://huggingface.co/" + name + "/raw/main/config.json") as url:
        data = json.loads(url.read().decode())
        for key in ['hidden_size', 'dim']:
            if key in data:
                return data[key]
    return 0

def makeParams(mlm, frozen):
    paramsPath = 'configs/params.' + mlm.replace('/', '-') + '.json'
    if frozen:
        paramsPath = 'configs/params.' + mlm.replace('/', '-') + '.frozen.json'
    if not os.path.isfile(paramsPath):
        size = getEmbedSize(mlm)
        outFile = open(paramsPath, 'w')
        outFile.write('local transformer_model = "' + mlm + '";\n')
        outFile.write('local transformer_dim = ' + str(size) + ';\n')
        outFile.write(''.join(open('mtp/configs/params.json').readlines()[2:]))
        outFile.close()
    if frozen:
        os.system('sed -i \"s;\\\"train_parameters\\\": true;\\\"train_parameters\\\": false;g\" ' + paramsPath)
    return paramsPath

def dataConfig(treebank):
    dataconfPath = 'configs/' + treebank + '.json'
    if not os.path.isfile(dataconfPath):
        config = {}
        train, dev = myutils.getTrainDev(treebank)
        config['train_data_path'] = '../' + train
        config['validation_data_path'] = '../' + dev
        config['word_idx'] = 1
        config['tasks'] = {'dependency': {'task_type': 'dependency', 'column_idx': 6}}
        allenConfig = Params({treebank: config})
        allenConfig.to_file(dataconfPath)
    return dataconfPath

def train(treebank, mlm, frozen):
    paramsPath = makeParams(mlm, frozen)
    dataconfPath = dataConfig(treebank)
    for seed in myutils.seeds:
        name = '.'.join([treebank, mlm.replace('/', '-'), seed])
        if frozen:
            name += '.frozen'
        cmd = 'cd mtp && python3 train.py --dataset_config ../' + dataconfPath + ' --parameters_config ../' + paramsPath 
        cmd += ' --seed ' + seed + ' --name ' + name + ' && cd ..'
        if myutils.getModel(name) == '':
            print(cmd)

for treebank, lang_mlms in zip(myutils.targets, myutils.treebank_mlms):
    for mlm in lang_mlms + myutils.multi_mlms:
        train(treebank, mlm, False)
    for mlm in myutils.multi_mlms:
        train(treebank, mlm, True)

