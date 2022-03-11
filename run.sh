#!/bin/bash

# This script consolidates all experiments from the main paper.
# Please download Universal Dependencies separately and point the UD_PATH variable to the appropriate location on your machine.
#
# Executing this script will then:
# * Set up consistent train/dev/test splits across UD.
# * Create UD subset definition for each of the 13 target treebanks.
# * Train three randomly initialized DirProbe and DepProbe models per target.
# * Evaluate in-language and zero-shot transfer performance.
#
# For more details, please refer to the README.md.

# experiment variables
# please update these on your machine
UD_PATH=/path/to/ud-treebanks
TARGETS=( ar-padt en-ewt eu-bdt fi-tdt he-htb hi-hdtb it-isdt ja-gsd ko-gsd ru-syntagrus sv-talbanken tr-imst zh-gsd )
LANGUAGES=( Arabic English Basque Finnish Hebrew Hindi Italian Japanese Korean Russian Swedish Turkish Chinese )
TREEBANKS=( PADT EWT BDT TDT HTB HDTB ISDT GSD GSD SynTagRus Talbanken IMST GSD )
SPLIT="test"
MODELS=( directed depprobe )
SEEDS=( 41 42 43 )

# set up data
mkdir -p exp/data
echo "Creating Universal Dependencies split definition..."
python data/split.py $UD_PATH exp/data/ --keep_all
for idx in "${!TARGETS[@]}"; do
	echo "Creating split definition for treebank ${TARGETS[$idx]}..."
	mkdir exp/data/"${TARGETS[$idx]}"
	python data/filter.py $UD_PATH exp/data/split.pkl exp/data/"${TARGETS[$idx]}" -il "${LANGUAGES[$idx]}" -it "${TREEBANKS[$idx]}"
done

# run experiments
mkdir -p exp/run
exp_path=exp/run
# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
	# iterate over source treebanks
	for src_idx in "${!TARGETS[@]}"; do
		source_tb=${TARGETS[$src_idx]}

		# iterate over models
		for mdl_idx in "${!MODELS[@]}"; do
			model=${MODELS[$mdl_idx]}
			exp_path=${exp_dir}/${model}/${source_tb}-rs${SEEDS[$rsd_idx]}
			echo "[$src_idx/$mdl_idx] $source_tb ($model)..."

			# train model
			case $model in
			
			directed)
			python train.py ${UD_PATH} ${exp_path} \
				-s exp/data/${source_tb}/filtered.pkl \
				-pt directed -el 7 \
				-e 30 -es 3 -rs ${SEEDS[$rsd_idx]}
			;;

			depprobe)
			python train.py ${UD_PATH} ${exp_path} \
				-s exp/data/${source_tb}/filtered.pkl \
				-pt rooted -el 6 7 \
				-e 30 -es 3 -rs ${SEEDS[$rsd_idx]}
			;;

			*)
			echo "Unknown model '${model}'."
			;;
			esac

			# iterate over target treebanks
			for tgt_idx in "${!TARGETS[@]}"; do
				ud_target=${TARGETS[$tgt_idx]//-/_}
				target_file=$(ls ${UD_PATH}*/${ud_target}-ud-${SPLIT}.conllu)
				pred_file=${exp_path}/${TARGETS[$tgt_idx]}-${SPLIT}-predict.conllu
				eval_file=${exp_path}/${TARGETS[$tgt_idx]}-${SPLIT}-results.txt

				# predict target using model trained on source
				case $model in

				directed)
				python predict.py ${exp_path} ${target_file} ${pred_file} \
					-pt directed -el 7 \
				;;

				depprobe)
				python predict.py ${exp_path} ${target_file} ${pred_file} \
					-pt rooted -el 6 7 \
				;;

				*)
				echo "Unknown model '${model}'."
				;;
				esac

				# evaluate LAS, UAS, UUAS
				python eval/dependencies.py ${pred_file} ${target_file} >> ${eval_file}
				echo "${model}-${source_tb}-rs${SEEDS[$rsd_idx]} -> ${TARGETS[$tgt_idx]} ($SPLIT):"
				cat ${eval_file}
				echo "--------------------"
			done
		done
	done
done

echo "All experiments completed."
