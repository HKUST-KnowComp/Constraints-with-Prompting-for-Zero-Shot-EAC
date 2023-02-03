import itertools
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from graph import Graph

instance_fields = [ 'doc_id', 'sent_id', 'sentence', 'graph', 'events']
Instance = namedtuple('Instance', field_names=instance_fields)



def get_arg_list(events, role_vocab, type_vocab):
    arg_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            arg_list.append((i, arg['start'], arg['end'], 
                role_vocab[arg['role']], 
                type_vocab[arg['entity_type']]))
    return arg_list

class EACDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.data = []
        self.load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['entity_type'])
        return type_set



    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as r:

            for line in r:
                inst = json.loads(line)

                for event in inst['event_mentions']:
                    event_type = event['event_type']
                    if ':' in event_type:
                        event['event_type'] = event_type.split(':')[1].upper()
                self.data.append(inst)


        print('Loaded {} instances from {}'.format(len(self), self.path))



    def numberize(self, vocabs):
       
        event_type_stoi = vocabs['event_type']
        role_type_stoi = vocabs['role_type']
        entity_type_stoi = vocabs['entity_type']

        trigger_label_stoi = vocabs['trigger_label']

        data = []
        for inst in self.data:

            sent_id = inst['sent_id']
            events = inst['event_mentions']
            sent = inst['sentence']
            doc_id = inst['doc_id']
            events.sort(key=lambda x: x['trigger']['start'])
            

            trigger_list = [(e['trigger']['start'], e['trigger']['end'],
                             event_type_stoi[e['event_type']])
                            for e in events]

            arg_list = get_arg_list(events, role_type_stoi, entity_type_stoi)

            graph = Graph(
                triggers=trigger_list,
                args=arg_list,
                vocabs=vocabs
            )

            instance = Instance(
                doc_id = doc_id,
                sent_id=sent_id,
                sentence = sent,
                graph = graph,
                events = events
            )
            
            data.append(instance)

        self.data = data