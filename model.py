import torch
import torch.nn as nn
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPTJForCausalLM, \
T5ForConditionalGeneration, T5TokenizerFast, BartTokenizer, BartForConditionalGeneration, \
RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertForMaskedLM
import copy
from itertools import product
import pickle as pkl
import re
import sys
#from drqa.pipeline.drqa import DrQA

entity_type_map = {'PER': 'person', 'ORG': 'organization', 
'LOC': 'location', 'FAC': 'facility', 'WEA': 'weapon', 'VEH': 'vehicle',
'GPE': 'geo-political entity', 'CRIME': 'crime', 'NUM': 'number',
'TIME': 'time', 'JOB': 'job title', 'SEN': 'sentence'}

class Model():

    def __init__(self, arg_role_map_path, arg_type_map_path, 
        arg_role_entity_type_constraint_map_path, time_expression_lexicon_path):
        super(Model, self).__init__()
        self.arg_role_map = json.load(open(arg_role_map_path))
        self.arg_type_map = json.load(open(arg_type_map_path))
        self.arg_role_entity_type_constraint_map = json.load(open(\
            arg_role_entity_type_constraint_map_path))
        self.time_expression_lexicon = pkl.load(open(\
            time_expression_lexicon_path, 'rb'))
        #self.retriever = DrQA(cuda=False)
        #self.tokenizer = T5TokenizerFast.from_pretrained('t5-11b')
        #self.LM = T5ForConditionalGeneration.from_pretrained('t5-11b', 
        #   return_dict=True)
        #self.tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-j-6B')
        #self.LM = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', 
        #    low_cpu_mem_usage=True, return_dict=True)

        #self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        #self.LM = BartForConditionalGeneration.from_pretrained('facebook/bart-large',
        #   return_dict=True, forced_bos_token_id=0)

        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        #self.LM = RobertaForMaskedLM.from_pretrained('roberta-large',
        #   return_dict=True)

        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        self.LM = BertForMaskedLM.from_pretrained("bert-large-uncased",
           return_dict=True)


    def score_labels(self, label_list, prompt, is_et):
        '''
        Use the negative value of the language modeling loss when assigning a label to 
        the mask in the prompt as the score of that label. => see the example 
        at https://huggingface.co/transformers/v3.5.1/model_doc/t5.html#transformers.T5ForConditionalGeneration
        '''

        #input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids

        score_list = []

        for label in label_list:
            

  
            if(is_et):
                label = entity_type_map[label]
        


            #labels = self.tokenizer('<extra_id_0> {} <extra_id_1>'.format(label), 
            #   return_tensors='pt').input_ids
            
            
            '''
            inputs = self.tokenizer(prompt.format(et_label, role), 
                return_tensors='pt')
            '''

    
            #print('prompt: ', prompt)

            labels = self.tokenizer(prompt.format(label), 
               return_tensors='pt').input_ids

            inputs = self.tokenizer(prompt.format('[MASK]'), return_tensors='pt').input_ids

            
            #print('decoded label tokens: ', [self.tokenizer.decode(tok) for tok in labels.tolist()[0]])
            
            #print('decoded input tokens: ', [self.tokenizer.decode(tok) for tok in inputs.input_ids.tolist()[0]])
            

            diff = labels.shape[1] - inputs.shape[1]
            #print('labels.shape: ', labels.shape)
            #print('inputs.input_ids.shape: ', inputs.input_ids.shape)
            #print('diff: ', diff)

            


            inputs = self.tokenizer(prompt.format(' '.join(['[MASK]' for _ in \
                        range(diff+1)])), return_tensors='pt')
            
            #print('inputs.input_ids.shape: ', inputs.input_ids.shape)
            
            '''
            print('input_ids: ', self.tokenizer.batch_decode(inputs.input_ids))
            print('input_ids.shape: ', inputs.input_ids.shape)
            print('labels: ', self.tokenizer.batch_decode(labels))
            print('labels.shape: ', labels.shape)
            '''
            labels = torch.where(inputs.input_ids == self.tokenizer.mask_token_id, labels, -100)

            '''
            print('prompt: ', prompt)
            print('label: ', label)
            print('input_ids: ', self.tokenizer.batch_decode(inputs.input_ids))
            print('input_ids.shape: ', inputs.input_ids.shape)
            print('labels: ', labels)
            print('labels.shape: ', labels.shape)
            '''
            outputs = self.LM(**inputs, labels=labels)

            score = - outputs.loss.item()

            score_list.append(score)

        return score_list


            

    def prompting(self, sentence, event):

        event_type = event['event_type']
        event_trigger = event['trigger']['text']
        eac_label_list = self.arg_role_map[event_type]
        et_label_list = self.arg_type_map[event_type]
        arg_score_list = []
        
        

        event_type_prefix_str = event_type
        if('-' in event_type):
            event_type_prefix_str = " ".join(event_type.split('-'))

        event_type_prefix_str = event_type_prefix_str.lower()

        prefix = 'This is a {} event whose occurrence is most clearly expressed by "{}". '.format(event_type_prefix_str, event_trigger)
        
        
        
        for arg in event['arguments']:

            start = arg['start']
            end = arg['end']
            '''
            arg_text = arg['text']
            doc_text, null_flag = self.retriever.process(query=arg_text, candidates=None, top_n=1, n_docs=1,
                return_context=False)
            if(not null_flag):

                doc_text = doc_text.split('\n\n')[1].split('.')[0] + '. '

            prefix = doc_text + prefix
            '''

            prompt = None

            '''
            #t5 prompt
            #prompt = prefix + sentence[0:start] + "<extra_id_0> especially " + sentence[start:]
            
            if(end >= len(sentence)):
                prompt = prefix + sentence + " and any other <extra_id_0> ."
            elif(sentence[end] == ' '):
                prompt = prefix + sentence[0:end+1] + "and any other <extra_id_0> " + sentence[end+1:]
            else:#sentence[end] == punctuation
                prompt = prefix + sentence[0:end] + " and any other <extra_id_0>" + sentence[end:]
            
            
            #t5 prompt
            
            
            if(sentence[end] == ' '):
                prompt = sentence[0:end+1] + "and any other <extra_id_0> " + sentence[end+1:]
            else:#sentence[end] == punctuation
                prompt = sentence[0:end] + " and any other <extra_id_0>" + sentence[end:]
            '''
            
            
            
            #gpt2 prompt
            #prompt = prefix + sentence[0:start] + "{} especially " + sentence[start:]
            
            
            if(end >= len(sentence)):
                prompt = prefix + '[SEP] ' + sentence + " and any other {}."
            
            elif(sentence[end] == ' '):
                prompt = prefix + '[SEP] ' + sentence[0:end+1] + "and any other {} " + sentence[end+1:]

            else:#sentence[end] == punctuation
                prompt = prefix + '[SEP] ' + sentence[0:end] + " and any other {} " + sentence[end:]
            
            '''
            
            
            if(end >= len(sentence)):
                prompt = sentence + " and any other {} ."
            
            elif(sentence[end] == ' '):
                prompt = sentence[0:end+1] + "and any other {} " + sentence[end+1:]

            else:#sentence[end] == punctuation
                prompt = sentence[0:end] + " and any other {} " + sentence[end:]
            '''
            
            
            eac_score_list = self.score_labels(eac_label_list, prompt, False)
            et_score_list = self.score_labels(et_label_list, prompt, True)

            arg_score_list.append({'eac_score_list': eac_score_list, 
                'et_score_list': et_score_list})
            

            
        return arg_score_list



    def predict(self, instance, arg_score_list_instance):


        sentence = instance.sentence

        events = instance.events

        pred_events = []

        event_type_list = []

        for event, arg_score_list_event in zip(events, 
            arg_score_list_instance):
            
            event_type = event['event_type']

            event_type_list.append(event_type)

            role_entity_type_map = self.arg_role_entity_type_constraint_map[event_type]

            pred_event = copy.deepcopy(event)

            for j, arg_score in enumerate(arg_score_list_event):
            
                role_candidate_labels = copy.deepcopy(self.arg_role_map\
                    [event_type])
                role_candidate_scores = copy.deepcopy(arg_score\
                    ["eac_score_list"])
                
                entity_type_candidate_labels = copy.deepcopy(self.arg_type_map\
                    [event_type])

                entity_type_candidate_scores = copy.deepcopy(arg_score\
                    ["et_score_list"])
                
                

                
                max_role_score = max(role_candidate_scores)
                max_entity_type_score = max(entity_type_candidate_scores)
                
                max_role_index = role_candidate_scores.index(max_role_score)
                max_entity_type_index = entity_type_candidate_scores.index(\
                    max_entity_type_score)

                pred_role = role_candidate_labels[max_role_index]
                

                pred_entity_type = entity_type_candidate_labels\
                [max_entity_type_index]
                
                
                
                tokens =  re.split(', | |-', pred_event['arguments'][j]['text']) 

                is_time_flag = False

                for token in tokens:
                    if(token in self.time_expression_lexicon):
                        is_time_flag = True
                        break

                    if(token.isdigit()):
                        int_val = int(token)
                        if(int_val>=1900 and int_val<=9999):
                            is_time_flag = True
                            break
                    
                    re_res = re.search('\d+:\d{2}', token)

                    if(re_res is not None):
                        is_time_flag = True
                        break

                    re_res = re.search('\d+\ds', token)

                    if(re_res is not None):
                        is_time_flag = True
                        break



                
                time_role_idx = self.arg_role_map[event_type].index('time')
                time_entity_type_idx = self.arg_type_map[event_type].index('TIME')

                if(is_time_flag):
                    pred_role = 'time'
                    role_candidate_scores[time_role_idx] = float('inf')
                    max_role_score = float('inf')
                    pred_entity_type = 'TIME'
                    entity_type_candidate_scores[time_entity_type_idx] = float('inf')
                    max_entity_type_score = float('inf')
                else:

                    role_candidate_scores[time_role_idx] = float('-inf')
                    max_role_score = max(role_candidate_scores)
                    max_role_index = role_candidate_scores.index(max_role_score)
                    pred_role = role_candidate_labels[max_role_index]

        
                    entity_type_candidate_scores[time_entity_type_idx] = float('-inf')
                    max_entity_type_score = max(entity_type_candidate_scores)
                    max_entity_type_index = entity_type_candidate_scores.index(max_entity_type_score)
                    pred_entity_type = entity_type_candidate_labels[max_entity_type_index]
                
                

                
                temp_role_candidate_scores = copy.deepcopy(role_candidate_scores)
                temp_entity_type_candidate_scores = copy.deepcopy(entity_type_candidate_scores)
                temp_role_candidate_labels = copy.deepcopy(role_candidate_labels)
                temp_entity_type_candidate_labels = copy.deepcopy(entity_type_candidate_labels)
                
                while(pred_entity_type not in role_entity_type_map\
                    [pred_role]):


                    if(max_role_score >= max_entity_type_score):
                        

                        temp_entity_type_candidate_labels = [l for i, l in \
                        enumerate(temp_entity_type_candidate_labels) \
                        if i != max_entity_type_index]

                        temp_entity_type_candidate_scores = [l for i, l in \
                        enumerate(temp_entity_type_candidate_scores) \
                        if i != max_entity_type_index]

                        max_entity_type_score = max(temp_entity_type_candidate_scores)
                        max_entity_type_index = temp_entity_type_candidate_scores.index(\
                            max_entity_type_score)

                        pred_entity_type = temp_entity_type_candidate_labels\
                            [max_entity_type_index]

                    else:

                        temp_role_candidate_labels = [l for i, l in \
                        enumerate(temp_role_candidate_labels) \
                        if i != max_role_index]

                        temp_role_candidate_scores = [l for i, l in \
                        enumerate(temp_role_candidate_scores) \
                        if i != max_role_index]

                        max_role_score = max(temp_role_candidate_scores)
                        max_role_index = temp_role_candidate_scores.index(\
                            max_role_score)

                        pred_role = temp_role_candidate_labels[max_role_index]
                
                
            
            
                
            
                


                pred_event['arguments'][j]['role'] = pred_role.capitalize()
                pred_event['arguments'][j]['entity_type'] = pred_entity_type
                pred_event['arguments'][j]['role_score'] = max_role_score
                pred_event['arguments'][j]['role_score_list'] =  role_candidate_scores
                pred_event['arguments'][j]['et_score'] = max_entity_type_score
                pred_event['arguments'][j]['et_score_list'] =  entity_type_candidate_scores
                


            '''
            if(event_type == "TRANSPORT"):
                
                origin_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Origin']

                if(len(origin_role_tup)>1):
                    
                    origin_role_tup = sorted(origin_role_tup, key=lambda item:item[0])
                    origin_role_idx = self.arg_role_map[event_type].index('origin')

                    for _, j in origin_role_tup[:-1]:
                        score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                        score_list[origin_role_idx] = min(score_list) -1
                        pred_event['arguments'][j]['role_score'] = max(score_list)
                        max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                        pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()

                
                
            
                
                dest_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Destination']

                if(len(dest_role_tup)>1):
                    
                    dest_role_tup = sorted(dest_role_tup, key=lambda item:item[0])
                    dest_role_idx = self.arg_role_map[event_type].index('destination')

                    for _, j in dest_role_tup[:-1]:
                        score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                        score_list[dest_role_idx] = min(score_list) -1
                        pred_event['arguments'][j]['role_score'] = max(score_list)
                        max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                        pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()
            
            '''


            
            
            if(event_type == "END-POSITION"):
                '''
                person_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Person']

                if(len(person_role_tup)>1):
                    
                    person_role_tup = sorted(person_role_tup, key=lambda item:item[0])
                    person_role_idx = self.arg_role_map[event_type].index('person')

                    for _, j in person_role_tup[:-1]:
                        score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                        score_list[person_role_idx] = min(score_list) -1
                        pred_event['arguments'][j]['role_score'] = max(score_list)
                        max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                        pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()

                
                
                
                
                
                entity_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Entity']

                if(len(entity_role_tup)>1):
                    
                    entity_role_tup = sorted(entity_role_tup, key=lambda item:item[0])
                    entity_role_idx = self.arg_role_map[event_type].index('entity')

                    for _, j in entity_role_tup[:-1]:
                        score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                        score_list[entity_role_idx] = min(score_list) -1
                        pred_event['arguments'][j]['role_score'] = max(score_list)
                        max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                        pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()

                '''
                
                
                
                position_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Position']

                if(len(position_role_tup)>1):
                    
                    position_role_tup = sorted(position_role_tup, key=lambda item:item[0])
                    position_role_idx = self.arg_role_map[event_type].index('position')

                    for _, j in position_role_tup[:-1]:
                        score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                        
                        score_list[position_role_idx] = min(score_list) -1
                        pred_event['arguments'][j]['role_score'] = max(score_list)
                        max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                        pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()
                  
              


            '''
            if(event_type not in ["PHONE-WRITE", "TRANSPORT", "EXTRADITE"]):
                place_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Place']

                if(len(place_role_tup)>1):
                    
                    place_role_tup = sorted(place_role_tup, key=lambda item:item[0])
                    place_role_idx = self.arg_role_map[event_type].index('place')

                    for _, j in place_role_tup[:-1]:
                        score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                        score_list[place_role_idx] = min(score_list) -1
                        pred_event['arguments'][j]['role_score'] = max(score_list)
                        max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                        pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()
        
            '''

            
            
            
            
            '''
            time_role_tup = [(arg['role_score'], j) for j, arg in enumerate(pred_event['arguments']) if arg['role'] == 'Time']

            if(len(time_role_tup)>1):
                
                time_role_tup = sorted(time_role_tup, key=lambda item:item[0])
                time_role_idx = self.arg_role_map[event_type].index('time')

                for _, j in time_role_tup[:-1]:
                    score_list = copy.deepcopy(pred_event['arguments'][j]['role_score_list'])
                    score_list[time_role_idx] = min(score_list) -1
                    pred_event['arguments'][j]['role_score'] = max(score_list)
                    max_idx = score_list.index(pred_event['arguments'][j]['role_score'])
                    pred_event['arguments'][j]['role'] = self.arg_role_map[event_type][max_idx].capitalize()
            
            '''
            
            
            pred_events.append(pred_event)


            
                
        

        '''
    
        if('START-POSITION' in event_type_list and 'END-POSITION' in event_type_list):
            start_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'START-POSITION']
            end_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'END-POSITION']

            for start_event, end_event in product(start_list, end_list):
                
                start_args = pred_events[start_event]['arguments']
                end_args = pred_events[end_event]['arguments']
                start_gold_args = events[start_event]['arguments']
                end_gold_args = events[end_event]['arguments']

                start_arg_len_idx_list = list(range(len(start_args)))
                end_arg_len_idx_list = list(range(len(end_args)))

                for s_idx, e_idx in product(start_arg_len_idx_list, 
                    end_arg_len_idx_list):

                    s_arg = start_args[s_idx]
                    e_arg = end_args[e_idx]
                    s_arg_gold = start_gold_args[s_idx]
                    e_arg_gold = end_gold_args[e_idx]

                    if(s_arg['start'] != e_arg['start'] or s_arg['end']!= e_arg['end']):
                        continue

                    arg1 = None
                    arg2 = None
                    arg1_idx = None
                    arg2_idx = None

                    if(s_arg['role_score'] >= e_arg['role_score']):

                        arg1 = s_arg
                        arg1_gold = s_arg_gold
                        arg1_idx = s_idx
                        arg2 = e_arg
                        arg2_gold = e_arg_gold
                        arg2_idx = e_idx

                    else:

                        arg1 = e_arg
                        arg1_gold = e_arg_gold
                        arg1_idx = e_idx
                        arg2 = s_arg
                        arg2_gold = s_arg_gold
                        arg2_idx = s_idx

                    if(arg1['role'] == 'Person'):
                        if(arg2['role'] != 'Person'):
                            print('sentence: ', sentence)
                            print('arg1 gold: ', arg1_gold)
                            print('arg1_pred role: Person')
                            print('arg2 gold: ', arg2_gold)
                            print('arg2_pred role: ', arg2['role'])
                            count+=1
                        arg2['role'] = 'Person'
                    elif(arg1['role'] == 'Entity'):
                        if(arg2['role'] != 'Entity'):
                            print('sentence: ', sentence)
                            print('arg1 gold: ', arg1_gold)
                            print('arg1_pred role: Entity')
                            print('arg2 gold: ', arg2_gold)
                            print('arg2_pred role: ', arg2['role'])
                            count+=1
                        arg2['role'] = 'Entity'
                    elif(arg1['role'] == 'Position'):
                        if(arg2['role'] != 'Position'):
                            print('sentence: ', sentence)
                            print('arg1 gold: ', arg1_gold)
                            print('arg1_pred role: Position')
                            print('arg2 gold: ', arg2_gold)
                            print('arg2_pred role: ', arg2['role'])
                            count+=1
                        arg2['role'] = 'Position'
        
        '''


        '''

        if('ARREST-JAIL' in event_type_list and 'CHARGE-INDICT' in event_type_list):
            arrest_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'ARREST-JAIL']
            charge_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'CHARGE-INDICT']

            for arrest_event, charge_event in product(arrest_list, charge_list):
                
                arrest_args = pred_events[arrest_event]['arguments']
                charge_args = pred_events[charge_event]['arguments']
                

                arrest_arg_len_idx_list = list(range(len(arrest_args)))
                charge_arg_len_idx_list = list(range(len(charge_args)))


                for a_idx, c_idx in product(arrest_arg_len_idx_list, 
                    charge_arg_len_idx_list):
                    
                    a_arg = arrest_args[a_idx]
                    c_arg = charge_args[c_idx]
                    

                    if(a_arg['start'] != c_arg['start'] or a_arg['end']!= c_arg['end']):
                        continue

                    arg1 = None
                    arg2 = None
                    arg1_idx = None
                    arg2_idx = None

                    if(a_arg['role_score'] >= c_arg['role_score']):
                        arg1 = a_arg
                        arg1_idx = a_idx
                        arg2 = c_arg
                        arg2_idx = c_idx
                    else:
                        arg1 = c_arg
                        arg1_idx = c_idx
                        arg2 = a_arg
                        arg2_idx = a_idx

                    if(arg1['role'] == 'Person'):

                        arg2['role'] = 'Defendant'
                    elif(arg1['role'] == 'Defendant'):

                        arg2['role'] = 'Person'
                    elif(arg1['role'] == 'Crime'):

                        arg2['role'] = 'Crime'
        '''
        

        '''
        if('ATTACK' in event_type_list and 'DIE' in event_type_list):
            attack_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'ATTACK']
            die_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'DIE']

            for attack_event, die_event in product(attack_list, die_list):
                
                attack_args = pred_events[attack_event]['arguments']
                die_args = pred_events[die_event]['arguments']
                

                attack_arg_len_idx_list = list(range(len(attack_args)))
                die_arg_len_idx_list = list(range(len(die_args)))

                for a_idx, d_idx in product(attack_arg_len_idx_list, 
                    die_arg_len_idx_list):
                    
                    a_arg = attack_args[a_idx]
                    d_arg = die_args[d_idx]
                    

                    if(a_arg['start'] != d_arg['start'] or a_arg['end']!= d_arg['end']):
                        continue

                    arg1 = None
                    arg2 = None
                    arg1_idx = None
                    arg2_idx = None

                    if(a_arg['role_score'] >= d_arg['role_score']):
                        arg1 = a_arg
                        arg1_idx = a_idx
                        arg2 = d_arg
                        arg2_idx = d_idx
                    else:
                        arg1 = d_arg
                        arg1_idx = d_idx
                        arg2 = a_arg
                        arg2_idx = a_idx

                    if(arg1['role'] == 'Place'):
                        arg2['role'] = 'Place'
                    elif(arg1['role'] == 'Victim'):
                        arg2['role'] = 'Target'
                    elif(arg1['role'] == 'Target'):
                        arg2['role'] = 'Victim'
                    elif(arg1['role'] == 'Instrument'):
                        arg2['role'] = 'Instrument'
                    elif(arg1['role'] == 'Time'):
                        arg2['role'] = 'Time'
                    elif(arg1['role'] == 'Attacker'):
                        arg2['role'] = 'Agent'
                    elif(arg1['role'] == 'Agent'):
                        arg2['role'] = 'Attacker'
        '''
        
        
        '''
        if('ATTACK' in event_type_list and 'INJURE' in event_type_list):
            attack_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'ATTACK']
            injure_list = [i for i, event_type in enumerate(event_type_list) if event_type == 'INJURE']

            for attack_event, injure_event in product(attack_list, injure_list):
                
                attack_args = pred_events[attack_event]['arguments']
                injure_args = pred_events[injure_event]['arguments']

                for a_arg, i_arg in product(attack_args, injure_args):
                    
                    if(a_arg['start'] != i_arg['start'] or a_arg['end']!= i_arg['end']):
                        continue

                    arg1 = None
                    arg2 = None

                    if(a_arg['role_score'] >= i_arg['role_score']):
                        arg1 = a_arg
                        arg2 = i_arg
                    else:
                        arg1 = i_arg
                        arg2 = a_arg

                    if(arg1['role'] == 'Place'):
                        arg2['role'] = 'Place'
                    elif(arg1['role'] == 'Victim'):
                        arg2['role'] = 'Target'
                    elif(arg1['role'] == 'Target'):
                        arg2['role'] = 'Victim'
                    elif(arg1['role'] == 'Instrument'):
                        arg2['role'] = 'Instrument'
                    elif(arg1['role'] == 'Time'):
                        arg2['role'] = 'Time'
                    elif(arg1['role'] == 'Attacker'):
                        arg2['role'] = 'Agent'
                    elif(arg1['role'] == 'Agent'):
                        arg2['role'] = 'Attacker'
        '''
        

        output = {'doc_id': instance.doc_id,
                  'sent_id': instance.sent_id,
                  'sentence': instance.sentence,
                  'event_mentions': pred_events
                  }

        return output


                



















