#!/usr/bin/python3

import argparse

import lang2vec.lang2vec as l2v
import numpy as np

from scipy.spatial.distance import pdist, squareform


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Language Similarities')
    arg_parser.add_argument('languages', nargs='+', help='sequence of ISO-639-3 language codes')
    arg_parser.add_argument(
        '-f', '--features',
        default='syntax_knn+phonology_knn+inventory_knn',
        help='lang2vec features to query')
    return arg_parser.parse_args()


def main():
    args = parse_arguments()

    # get lang2vec embeddings
    embeddings = []
    for language in args.languages:
        if language not in l2v.LANGUAGES:
            print(f"[Error] Language '{language}' is not supported by lang2vec.")
            exit(1)
        embeddings.append(l2v.get_features(language, args.features)[language])
    embeddings = np.array(embeddings)
    print(f"Gathered L2V embeddings for {len(embeddings)} languages with features '{args.features}'.")

    distances = 1 - squareform(pdist(embeddings, 'cosine'))
    # print results
    print("Language Distances:")
    print('Language\t' + '\t'.join(args.languages))
    for idx, language in enumerate(args.languages):
        print(language + '\t' + '\t'.join([f'{d:.4f}' for d in distances[idx, :]]))


if __name__ == '__main__':
    main()
