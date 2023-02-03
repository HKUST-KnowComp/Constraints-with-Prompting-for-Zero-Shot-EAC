from data import EACDataset


def generate_vocabs(datasets):
	"""Generates vocabularies from a list of data sets
	:param datasets (list): A list of data sets
	:return (dict): A dictionary of vocabs
	"""
	event_type_set = set()
	role_type_set = set()
	entity_type_set = set()

	for dataset in datasets:
		event_type_set.update(dataset.event_type_set)
		role_type_set.update(dataset.role_type_set)
		entity_type_set.update(dataset.entity_type_set)
	# entity and trigger labels
	prefix = ['B', 'I']
	trigger_label_stoi = {'O': 0}

	for t in event_type_set:
		for p in prefix:
			trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)


	event_type_stoi = {k: i for i, k in enumerate(event_type_set, 1)}
	event_type_stoi['O'] = 0


	role_type_stoi = {k: i for i, k in enumerate(role_type_set, 1)}
	role_type_stoi['O'] = 0

	entity_type_stoi = {k: i for i, k in enumerate(entity_type_set, 1)}
	entity_type_stoi['O'] = 0


	return {
		'event_type': event_type_stoi,
		'role_type': role_type_stoi,
		'entity_type': entity_type_stoi,
		'trigger_label': trigger_label_stoi
	}


def safe_div(num, denom):
	if denom > 0:
		if num / denom <= 1:
			return num / denom
		else:
			return 1
	else:
		return 0


def compute_f1(predicted, gold, matched):
	precision = safe_div(matched, predicted)
	recall = safe_div(matched, gold)
	f1 = safe_div(2 * precision * recall, precision + recall)
	return precision, recall, f1


def convert_arguments(triggers, args):
	new_args = set()
	for arg in args:
		trigger_idx = arg[0]
		trigger_label = triggers[trigger_idx][-1]
		new_args.add((trigger_label, arg[1], arg[2], arg[3], 
			arg[4]))
	return new_args



def score_graphs_gold_AI(gold_graphs, pred_graphs):

	eac_correct = 0
	et_correct = 0
	total = 0

	for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
		
		for pred_arg, gold_arg in zip(pred_graph.args, gold_graph.args):
			total+=1
			if(pred_arg[-2] == gold_arg[-2]):
				eac_correct+=1
			if(pred_arg[-1] == gold_arg[-1]):
				et_correct+=1

	eac_correct = float(eac_correct)
	et_correct = float(et_correct)
	total = float(total)

	print('EAC score: ', (eac_correct/total)*100.0)
	print('ET score: ', (et_correct/total)*100.0)


def score_graphs(gold_graphs, pred_graphs):
	gold_arg_num = pred_arg_num = arg_idn_num = arg_role_num = arg_type_num = 0
	gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
	gold_men_num = pred_men_num = men_match_num = 0

	for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):

		# Trigger
		gold_triggers = gold_graph.triggers
		pred_triggers = pred_graph.triggers

		'''
		gold_trigger_num += len(gold_triggers)
		pred_trigger_num += len(pred_triggers)
		for trg_start, trg_end, event_type in pred_triggers:
			matched = [item for item in gold_triggers
			           if item[0] == trg_start and item[1] == trg_end]
			if matched:
				trigger_idn_num += 1
				if matched[0][-1] == event_type:
					trigger_class_num += 1
		'''

		# Argument
		gold_args = convert_arguments(gold_triggers, gold_graph.args)
		pred_args = convert_arguments(pred_triggers, pred_graph.args)
		gold_arg_num += len(gold_args)
		pred_arg_num += len(pred_args)

		for pred_arg in pred_args:
			event_type, arg_start, arg_end, role, \
			entity_type = pred_arg
			gold_idn = {item for item in gold_args
			            if item[1] == arg_start and item[2] == arg_end
			            and item[0] == event_type}
			if gold_idn:

				arg_idn_num += 1
				gold_role = {item for item in gold_idn if item[-2] == role}
				gold_entity_type = {item for item in gold_idn if item[-1] == entity_type}
				
				if gold_role:
					arg_role_num += 1
				if gold_entity_type:
					arg_type_num += 1

	

	'''
	trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
		pred_trigger_num, gold_trigger_num, trigger_idn_num)
	trigger_prec, trigger_rec, trigger_f = compute_f1(
		pred_trigger_num, gold_trigger_num, trigger_class_num)
	role_id_prec, role_id_rec, role_id_f = compute_f1(
		pred_arg_num, gold_arg_num, arg_idn_num)
	'''
	role_prec, role_rec, role_f = compute_f1(
		float(pred_arg_num), float(gold_arg_num), float(arg_role_num))
	type_prec, type_rec, type_f = compute_f1(
		float(pred_arg_num), float(gold_arg_num), float(arg_type_num))
	'''
	print('Trigger Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		trigger_id_prec * 100.0, trigger_id_rec * 100.0, trigger_id_f * 100.0))
	print('Trigger Classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		trigger_prec * 100.0, trigger_rec * 100.0, trigger_f * 100.0))
	print('Argument Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
	'''
	print('Argument Classification Scores: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
	print('Argument Entity Typing Scores: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		type_prec * 100.0, type_rec * 100.0, type_f * 100.0))
	
	'''
	scores = {
		'TC': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
		'TI': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
		       'f': trigger_id_f},
		'AC': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
		'AI': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
	}
	
	scores = {'prec': role_prec, 'rec': role_rec, 'f': role_f}

	return scores

	'''









