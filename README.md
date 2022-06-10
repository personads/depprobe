# Dependency Probing

> Parsing UD with <3 Matrices

![Embeddings are multiplied with matrix B and L to transform them into a tree structural subspace and label subspace. Node with highest root probability sets directionality of all edges by pointing them away. Edges are labeled according to child embedding label from L.](header.png)

This archive contains implementations of the methods from the papers:

* "**Probing for Labeled Dependency Trees**" ([Müller-Eberstein, van der Goot and Plank, 2022a](https://personads.me/x/acl-2022-paper); ACL 2022);
* "**Sort by Structure: Language Model Ranking as Dependency Probing**" ([Müller-Eberstein, van der Goot and Plank, 2022b](https://personads.me/x/naacl-2022-paper); NAACL 2022).

After installing the required packages, and downloading the Universal Dependencies data, the experiments can be re-run using the `run-acl.sh` and `run-naacl.sh` scripts respectively. Please see the instructions below for details.

The current codebase is compatible with both the ACL 2022 and NAACL 2022 experiments. However, if you wish to use the original version of the code used in the ACL paper, the appropriate commit is tagged `acl-2022` and can be found [here](https://github.com/personads/depprobe/releases/tag/acl-2022).

## Installation

This repository uses Python 3.6+ and the associated packages listed in the `requirements.txt`:

```bash
(venv) $ pip install -r requirements.txt
```

## Data

The experiments are run on treebanks from Universal Dependencies version 2.8 (Zeman et al., 2021) which can be obtained separately at: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3687.

### Treebanks for ACL 2022

The 13 target treebanks used are: AR-PADT (Hajič et al., 2009), EN-EWT (Silveira et al., 2014), EU-BDT (Aranzabe et al., 2015), FI-TDT (Pyysalo et al., 2015), HE-HTB (McDonald et al., 2013), HI-HDTB (Palmer et al., 2009), IT-ISDT (Bosco et al., 2014), JA-GSD (Asahara et al., 2018), KO-GSD (Chun et al., 2018), RU-SynTagRus (Droganova et al., 2018), SV-Talbanken (McDonald et al., 2013), TR-IMST (Sulubacak et al., 2016), ZH-GSD (Shen et al., 2016).

### Treebanks for NAACL 2022

The nine target treebanks used are: AR-PADT (Hajič et al., 2009), EN-EWT (Silveira et al., 2014), FI-TDT (Pyysalo et al., 2015), GRC-PROIEL (Eckhoff et  al.,  2018),  HE-HTB  (McDonald et al., 2013a), KO-GSD (Chun et al., 2018), RU-GSD (McDonald et al., 2013b), SV-Talbanken (McDonald et al., 2013a), ZH-GSD (Shen et al., 2016).

## DepProbe

Given that the pre-requesites are fulfilled, the experiments can be re-run using the respective `run-acl.sh` or `run-naacl.sh` script. Please update the `data_path` and `exp_path` variables within the scripts to point to the appropriate data and experiment directories on your machine.

### ACL 2022 Experiments

```bash
(venv) $ ./run-acl.sh
```

will prepare the UD treebanks and train DepProbe as well as DirProbe models on all 13 target treebanks. Each experiment directory will contain predictions for the in-language test split and cross-lingual transfer results for the remaining 12 languages.

### NAACL 2022 Experiments

```bash
(venv) $ ./run-naacl.sh
```

will run all experiments using DepProbe from the main paper. This involves three random initializations across nine target languages with 4–6 language models each, totalling 138 setups.

## BAP (Biaffine Attention Parser)

We compare DepProbe and DirProbe with a full biaffine attention parser (BAP) as implemented in the MaChAmp framework version 0.3 (van der Goot et al., 2021). It is a freely available AllenNLP-based toolkit which is available at https://github.com/machamp-nlp/machamp.

For reproducing the results of BAP, we provide the parameter configurations and scripts for our dependency parsing experiments in `parsing/acl/` and `parsing/naacl/` respectively. Please use the same target treebanks in the probing experiments.

Parameters follow the default recommended configuration, but have been explicitly stated in order to ensure replicability. They can be found under `parameter-configs/params-rs*.json` with `rs-42` for example denoting the use of the random seed 42. All other parameters are identical.

## Further Analyses

### Lang2Vec Similarities

The `eval/langsim.py` script can be used to calculate the cosine similarities between all language pairs using Lang2Vec (Littell et al., 2017).

```bash
(venv) $ python eval/langsim.py "ara" "eng" "eus" "fin" "heb" "hin" "ita" "jpn" "kor" "rus" "swe" "tur" "zho"
```

### Subspace Angles

To perform an analysis of probe subspace angles across experiments/languages, use `eval/ssa.py`:

```bash
(venv) $ python eval/ssa.py /path/to/exp1 /path/to/exp2 ...
```

