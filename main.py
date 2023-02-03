from argparse import ArgumentParser
import json
import torch
import random
import numpy as np
from data import EACDataset
from util import generate_vocabs, score_graphs_gold_AI
import os
from model import Model
from pprint import pprint
from joblib import Parallel, delayed
import pickle as pkl
from tqdm import tqdm


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def main():
	parser = ArgumentParser()

	parser.add_argument('--input_dir', type=str, required=True,
		help='path to the input directory')
	parser.add_argument('--output_dir', type=str, required=True,
		help='path to the output directory')
	parser.add_argument('--dataset_name', type=str, required=True,
		help='name of the dataset')
	parser.add_argument('--split', type=str, required=True,
		help='split of the dataset')

	parser.add_argument('--lm_name', type=str, required=True,
		help='name of language model')

	parser.add_argument('--model_name', type=str, required=True,
		help='name of model')

	parser.add_argument('--mode', type=str, required=True,
		help='mode of program (prompting or predict)', choices=['prompting', 'predict'])


	args = parser.parse_args()

	

	#build model

	model = Model(arg_role_map_path = 'src/arg_role_map.json',
		arg_type_map_path= 'src/arg_type_map.json', 
		arg_role_entity_type_constraint_map_path = \
		'src/arg_role_entity_type_constraint_map.json',
		time_expression_lexicon_path = 'src/time_expression_lexicon.pickle')
	


	
	input_file = os.path.join(args.input_dir, 
		'{}.event.json'.format(args.split))

	input_dataset = EACDataset(input_file)

	vocabs = generate_vocabs([input_dataset])

	input_dataset.numberize(vocabs)

	if(args.mode == 'prompting'):

		arg_score_list_dataset = []

		

		for instance in tqdm(input_dataset):

			arg_score_list_instance = []
			
			for event in instance.events:
				arg_score_list_event = model.prompting(instance.sentence, event)
				arg_score_list_instance.append(arg_score_list_event)

			arg_score_list_dataset.append(arg_score_list_instance)

		pkl.dump(arg_score_list_dataset, open(os.path.join(args.output_dir, '{}_{}_{}_arg_score_list_dataset.pickle'.format(args.lm_name, args.dataset_name, args.split)), 'wb'))
		

	elif(args.mode == 'predict'):
		output_file = os.path.join(args.output_dir, 
			"{}_{}_output.event.json".format(args.model_name, 
				args.dataset_name))

		arg_score_list_dataset = pkl.load(open(os.path.join(args.output_dir, '{}_{}_{}_arg_score_list_dataset.pickle'.format(args.lm_name, args.dataset_name, args.split)), 'rb'))
		
		with open(output_file, 'w') as fw:
			
			for instance, arg_score_list_instance in \
			tqdm(zip(input_dataset, arg_score_list_dataset)):
				output = model.predict(instance, arg_score_list_instance)
			
				fw.write(json.dumps(output) + '\n')
				fw.flush()


		
		## Evaluate

		gold_dataset = EACDataset(input_file)
		pred_dataset = EACDataset(output_file)

		vocabs = generate_vocabs([gold_dataset, pred_dataset])

		gold_dataset.numberize(vocabs)
		pred_dataset.numberize(vocabs)

		gold_graphs, pred_graphs = [], []

		i = 0
		for inst1, inst2 in zip(gold_dataset, pred_dataset):
			i += 1
			gold_graphs.append(inst1.graph)
			pred_graphs.append(inst2.graph)

		score_graphs_gold_AI(gold_graphs, pred_graphs)


if __name__ == "__main__":
    main()