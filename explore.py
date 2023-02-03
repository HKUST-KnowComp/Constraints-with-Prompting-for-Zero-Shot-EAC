import json
from util import *
from data import EACDataset

input_file = 'data/ACE05-E+_converted/test.event.json'
output_file = 'output/gpt2-xl_with_constraint_ACE05-E+_output.event.json'

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
