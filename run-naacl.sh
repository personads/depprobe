#!/bin/bash

# experiment variables
# please update these on your machine
UD_PATH=/path/to/ud/treebanks
EXP_PATH=/path/to/experiments
# 'UD_Ancient_Greek-PROIEL', 'UD_Arabic-PADT', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_Chinese-GSD', 'UD_Hebrew-HTB', 'UD_Korean-GSD', 'UD_Russian-GSD', 'UD_Swedish-Talbanken']
TARGETS=( ar-padt en-ewt fi-tdt grc-proiel he-htb ko-gsd ru-gsd sv-talbanken zh-gsd )
LANGUAGES=( Arabic English Finnish Ancient_Greek Hebrew Korean Russian Swedish Chinese )
TREEBANKS=( PADT EWT TDT PROIEL HTB GSD GSD Talbanken GSD )
EVAL_SPLIT="dev"
SEEDS=( 692 710 932 )

# iterate over seeds
num_exp=0
num_train=0
for rsd_idx in "${!SEEDS[@]}"; do
	# iterate over source treebanks
	for src_idx in "${!TARGETS[@]}"; do
		source_tb=${TARGETS[$src_idx]}
		tb_dir=${UD_PATH}/UD_${LANGUAGES[$src_idx]}-${TREEBANKS[$src_idx]}

		# select language models for current treebank
		case $source_tb in
			ar-padt)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert aubmindlab/bert-base-arabertv02 asafaya/bert-base-arabic )
			layers=( 6 6 16 6 6 )
			;;

			en-ewt)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert bert-base-uncased roberta-large distilbert-base-uncased )
			layers=( 6 6 16 6 12 3 )
			;;

			fi-tdt)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert TurkuNLP/bert-base-finnish-uncased-v1 TurkuNLP/bert-base-finnish-cased-v1 )
			layers=( 6 6 16 6 6 )
			;;

			grc-proiel)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert pranaydeeps/Ancient-Greek-BERT nlpaueb/bert-base-greek-uncased-v1 )
			layers=( 6 6 16 6 6 )
			;;

			he-htb)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert onlplab/alephbert-base )
			layers=( 6 6 16 6 )
			;;

			ko-gsd)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert klue/bert-base klue/roberta-large kykim/bert-kor-base )
			layers=( 6 6 16 6 12 6 )
			;;

			ru-gsd)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert cointegrated/rubert-tiny sberbank-ai/ruRoberta-large blinoff/roberta-base-russian-v0 )
			layers=( 6 6 16 2 12 6 )
			;;

			sv-talbanken)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert KB/bert-base-swedish-cased )
			layers=( 6 6 16 6 )
			;;

			zh-gsd)
			models=( bert-base-multilingual-cased xlm-roberta-base google/rembert bert-base-chinese hfl/chinese-bert-wwm-ext hfl/chinese-roberta-wwm-ext )
			layers=( 6 6 16 6 6 6 )
			;;

			*)
			echo "[Error] Unknown target '${source_tb}'."
			exit
			;;
		esac

		# iterate over relevant models
		for mdl_idx in "${!models[@]}"; do
			# set up experiment directory
			exp_dir=${EXP_PATH}/depprobe/${source_tb}/model${mdl_idx}

			# set up model directory
			if [ ! -d "$exp_dir" ]; then
        mkdir -p $exp_dir
			fi
			# check if experiment already exists
      exp_dir=${exp_dir}/rs${SEEDS[$rsd_idx]}
			if [ -d "$exp_dir" ]; then
				echo "[Warning] Experiment '$exp_dir' already exists. Skipped."
				continue
			fi

			# train DepProbe
			echo "Training DepProbe on ${source_tb} using ${models[$mdl_idx]}..."
			python train.py ${tb_dir} ${exp_dir} \
				-lm ${models[$mdl_idx]} \
				-pt depprobe -el ${layers[$mdl_idx]} ${layers[$mdl_idx]} \
				-e 30 -es 3 -rs ${SEEDS[$rsd_idx]} \
				-td
			echo "DepProbe trained on ${source_tb} using ${models[$mdl_idx]} (RS=${SEEDS[$rsd_idx]})." > ${exp_dir}/experiment-info.txt
			((num_train++))

			# evaluate trained DepProbe on current target treebanks
			ud_target=${TARGETS[$src_idx]//-/_}
			target_file=$(ls ${UD_PATH}/*/${ud_target}-ud-${EVAL_SPLIT}.conllu)
			pred_file=${exp_dir}/${TARGETS[$src_idx]}-${EVAL_SPLIT}-predict.conllu
			eval_file=${exp_dir}/${TARGETS[$src_idx]}-${EVAL_SPLIT}-results.txt

			# predict target using model trained on source
			python predict.py ${exp_dir} ${target_file} ${pred_file} \
				-pt depprobe -lm ${models[$mdl_idx]} -el ${layers[$mdl_idx]} ${layers[$mdl_idx]}

			# evaluate LAS, UAS, UUAS
			python eval/dependencies.py ${pred_file} ${target_file} >> ${eval_file}
			echo "DepProbe on ${source_tb} (LM: ${models[$mdl_idx]}, RS: ${SEEDS[$rsd_idx]}) -> ${TARGETS[$src_idx]} ($EVAL_SPLIT):"
			cat ${eval_file}
			echo "--------------------"

			# increment experiment counter
			((num_exp++))
		done
	done
done

echo "Completed running ${num_exp} experiments (${num_train} trained models)."

echo "All experiments completed."
