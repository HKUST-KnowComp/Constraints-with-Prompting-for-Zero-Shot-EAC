from argparse import ArgumentParser
import random
import numpy as np
import json
from data import EACDataset
from util import generate_vocabs, score_graphs_gold_AI
import copy
import os
from pprint import pprint

random.seed(0)
np.random.seed(0)

def main():

    parser = ArgumentParser()

    parser.add_argument('--input_f', type=str, required=True,
        help='path to the input file')
    parser.add_argument('--dataset_name', type=str, required=True,
        help='name of the dataset')
    parser.add_argument('--output_dir', type=str, required=True,
        help='path to the output directory')
    
    args = parser.parse_args()

    arg_role_map = json.load(open('src/arg_role_map.json'))
    event_role_count = {}

    for event, role_list in arg_role_map.items():
        event_role_count[event] = [0]*len(role_list)

    with open(args.input_f, 'r', encoding='utf-8') as r:
        for line in r:
            inst = json.loads(line)
            for event in inst['event_mentions']:
                event_type = event['event_type'].split(':')[1].upper() \
                if ':' in event['event_type'] else event['event_type']

                for arg in event['arguments']:
                    event_role_count[event_type][arg_role_map[event_type].index(arg['role'].lower())]+=1


    mvb = {}
    for event, role_count in event_role_count.items():
        max_count = max(role_count)
        max_role = arg_role_map[event][role_count.index(max_count)]
        mvb[event] = max_role

    input_dataset = EACDataset(args.input_f)

    vocabs = generate_vocabs([input_dataset])

    input_dataset.numberize(vocabs)

    output_file = os.path.join(args.output_dir, 
        "{}_{}_output.event.json".format('majority_voting_baseline_merge', 
            args.dataset_name))



    with open(output_file, 'w') as fw:
        
        for i, instance in enumerate(input_dataset):

            print(i, instance.sentence)
            pred_events = []

            for event in instance.events:
                event_type = event['event_type']
                pred_event = copy.deepcopy(event)
                for i in range(len(event['arguments'])):
                    pred_event['arguments'][i]['role'] = mvb[event_type].capitalize()

                pred_events.append(pred_event)


            # Gold events and model predictions will also be printed.
            print('Gold events:')
            pprint(instance.events)
            print('Pred events:')
            pprint(pred_events)
            print('\n')

            output = {'doc_id': instance.doc_id,
                      'sent_id': instance.sent_id,
                      'sentence': instance.sentence,
                      'event_mentions': pred_events
                      }

            fw.write(json.dumps(output) + '\n')
            fw.flush()


    ## Evaluate

    gold_dataset = EACDataset(args.input_f)
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
