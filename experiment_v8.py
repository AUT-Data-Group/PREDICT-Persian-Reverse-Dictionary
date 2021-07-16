from datetime import datetime
from time import time
from sklearn.model_selection import KFold
from random import shuffle, sample, seed
import pickle
import json
from collections import defaultdict, Counter
from numpy import zeros, array, nan_to_num, flip, median, var, linalg as LA
from scipy.spatial import distance as dist
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Input, Embedding, Dense, Lambda, LSTM, Dropout, GRU, Bidirectional, Flatten, Reshape, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from tensorflow.keras.backend import mean, expand_dims, squeeze
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import re
import pandas as pd
from bs4 import BeautifulSoup as bs

def colorize(words, attention_scores):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    template = '<span style="text-shadow: 0px 0px 5px #000000; white-space:nowrap;color: white; background-color:hsl(0,100%,{}%)">{}</span>'
    colored_string = '<div style="text-align:right;direction:rtl">'
    for word, score in zip(words, attention_scores):
        if word != 'PAD':
            x = 50+50*(1-score)
            colored_string += template.format(x, '&nbsp' + word + '&nbsp')
    colored_string += '</div>'
    return colored_string
	
def html(text):
    text = '<div style="text-align:right; direction:rtl;font-family:tahoma">'+text+'</div>'
    return display(HTML(text))
	
def now_date_time():
    return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def remove_html_tags(string):
    # return bs(string, 'lxml').text
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', string)
    return cleantext
    
def has_any_persian(string):
    for ch in string:
        if re.match('[\u0600-\u06FF\s]',ch):
            return True
    return False
    
def normalize_characters(string, normalizer_dict):
    new_string = ''
    keys = set(normalizer_dict.keys())
    for ch in string:
        if ch in keys:
            new_string += normalizer_dict[ch]
        else:
            new_string += ' '
    return new_string
    
def correct_whitespaces(string):
    return re.sub(' +',' ', string).strip()
    
def tokenize(string):
    return string.split()
    
def remove_self_definition(tokens, word):
    tokens = [token for token in tokens if not token == word]
    return tokens
    
def remove_stopwords(tokens, stopwords):
    stopwords = set(stopwords)
    tokens = [token for token in tokens if token not in stopwords]
    return tokens
    
def has_vector(word, vector_model):
    try:
        vector_model[word]
        return True
    except:
        return False
        
def most_frequent(lst, k):
    freq = Counter()
    for item in lst:
        freq[item] += 1
    topk = freq.most_common(k)
    topk = [item for (item, count) in topk]
    topk = set(topk)
    return topk

def cosine_sim(y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, -1)
    y_pred = tf.nn.l2_normalize(y_pred, -1)    
    return tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)
    
def cosine_loss(y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, -1)
    y_pred = tf.nn.l2_normalize(y_pred, -1)    
    return 1 - tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)

def rank_loss(m):
    def rl(y_true, y_pred):
        y_true = tf.nn.l2_normalize(y_true, -1)
        y_pred = tf.nn.l2_normalize(y_pred, -1)
        #y_rnd = tf.random.shuffle(y_true)
        y_rnd = tf.gather(y_true, tf.random.shuffle(tf.range(tf.shape(y_true)[0])))
        return tf.keras.backend.mean(   tf.maximum(  tf.constant(0.0),  tf.constant(m) - cosine_sim(y_true, y_pred) + cosine_sim(y_true,y_rnd)  )   )
    return rl
    
    
    
def cosine_similarity(y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, -1)
    y_pred = tf.nn.l2_normalize(y_pred, -1)  
    return math_ops.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)


def e_end(metric, epoch, logs, file_dir):
    log = 'Epoch: '+str(epoch)+' - Loss: '+str(logs['loss'])+' - Val_Loss: '+str(logs['val_loss'])+' - Cosine Similarity: '+str(logs[metric])+' - Val Cosine Similarity: '+str(logs['val_'+metric])+'\n' 
    f = open(file_dir, 'a+', encoding='utf-8')
    f.write(log)
    f.close()
    
def intent_e_end(epoch, logs, file_dir):
    log =  '\n'+'Epoch #'+str(epoch)+'\n'
    log += 'Loss: '+str(logs['loss'])+' - Top1: '+str(logs['top1'])+' - Top2: '+str(logs['top2']) + ' - Top3: '+str(logs['top3'])
    log += '\n'+'val_Loss: '+str(logs['val_loss'])+' - val_Top1: '+str(logs['val_top1'])+' - val_Top2: '+str(logs['val_top2']) + ' - val_Top3: '+str(logs['val_top3'])+'\n'
    f = open(file_dir, 'a+', encoding='utf-8')
    f.write(log)
    f.close()
  
  
def update_ranking_based_on_intent(ranking, intent, id2h, word_tags):
    temp_ranking_first_part = []
    temp_ranking_second_part = []
    for word_idx in ranking:
        flag = False
        curr_word = id2h[word_idx]
        for tag in intent:
            if tag in word_tags[curr_word]:
                flag = True
                break
        if flag == True:
            temp_ranking_first_part.append(word_idx)
        else:
            temp_ranking_second_part.append(word_idx)
    new_ranking = temp_ranking_first_part
    new_ranking.extend(temp_ranking_second_part)
    return new_ranking
    
def preprocess_definition(definition, max_seq_len, tools):
    globals().update(tools)
    active = True
    preprocessed_definition = definition[:]
    # remove html tags
    definition = remove_html_tags(definition)
    preprocessed_definition = remove_html_tags(preprocessed_definition)
    # check if has Persian info
    if not has_any_persian(preprocessed_definition):
        active = False
    # normalizing the characters
    if active == True:
        preprocessed_definition = normalize_characters(preprocessed_definition, normalizer)
    # whitespace correction
    if active == True:
        preprocessed_definition = correct_whitespaces(preprocessed_definition)
    # tokenization
    if active == True:
        preprocessed_definition = tokenize(preprocessed_definition)
    # removing stopwords
    if active == True:
        preprocessed_definition = remove_stopwords(preprocessed_definition, stopwords)
    # normalizing tokens by frequency
    if active == True:
        preprocessed_definition = [token if token in t2id else 'UNK' for token in preprocessed_definition]
        curr_tokens = set(preprocessed_definition)
        if len(curr_tokens) == 1 and curr_tokens == {'UNK'}:
            active = False
    # check if is a short definition
    if active == True:
        if len(preprocessed_definition) == 1 and len(preprocessed_definition[0]) < 3:
            active = False
    # check if the definition is empty
    if active == True:
        if len(preprocessed_definition) == 0:
            active = False
    # fix sequence length
    if active == True:
        tokens = preprocessed_definition
        preprocessed_definition = tokens[:max_seq_len] if len(tokens)>=max_seq_len else tokens+['PAD' for j in range(max_seq_len-len(tokens))]
    # removing the definitions consisted of only 'PAD' and 'UNK'
    if active == True:
        tokens = set(preprocessed_definition)
        if tokens.issubset({'PAD','UNK'}):
            active = False
    # generate the sample in numpy
    if active == True:
        nparray = [t2id[token] for token in preprocessed_definition]
        nparray = array(nparray)
    # returning the result
    if active == False:
        return definition, [''], np.array([1 for k in range(max_seq_len)])
    else:
        return definition, preprocessed_definition, nparray

def get_rank(lst, items):
    items = set(items)
    for i, el in enumerate(lst):
        if el in items:
            return el, i+1
        
def query_intent_classifier(queries, tools, method='names'):
    globals().update(tools)
    queries = [normalize_characters(query, normalizer) for query in queries]
    tokenized_queries = [query.split() for query in queries]
    tokenized_queries = [[token if token in intent_g2id else 'UNK' for token in tokens_lst] for tokens_lst in tokenized_queries]
    samples = [[intent_g2id[token] for token in tokens_lst] for tokens_lst in tokenized_queries]
    samples = [sample[:intent_maxlen] if len(sample)>=intent_maxlen else sample+[0 for i in range(intent_maxlen-len(sample))] for sample in samples]
    samples = np.array(samples)
    preds = intent_model(samples).numpy()
    preds_masks = [[1 if prob>=intent_threshold else 0 for prob in p] for p in preds]
    if method == 'names':
        classes = [[id2c[idx] for idx, mask in enumerate(pm) if mask == 1] for pm in preds_masks]
        return classes
    elif method == 'one-hot':
        one_hot = preds_masks
        one_hot = np.array(one_hot)
        return one_hot
    elif method == 'probs':
        return preds
        
def query(inputs, topn, max_seq_len, models, tools):
    globals().update(models)
    words = inputs['words'] if 'words' in inputs else ['UNK' for definition in inputs['definitions']]
    definitions = inputs['definitions']
    general_tags= inputs['general_tags'] if 'general_tags' in inputs else [set() for definition in inputs['definitions']]
    sense_tags= inputs['sense_tags'] if 'sense_tags' in inputs else [set() for definition in inputs['definitions']]
    sources = inputs['sources'] if 'sources' in inputs else ['UNK' for definition in inputs['definitions']]
    preprocessed_definitions = []
    globals().update(tools)
    
    if 'intent_model' in globals():
        intent_tools = {'intent_g2id':intent_g2id, 'id2c':intent_id2c, 'normalizer':normalizer,'intent_maxlen':intent_maxlen
        ,'intent_model':intent_model, 'intent_threshold':intent_threshold}
        intent_classes = query_intent_classifier(definitions, intent_tools, method='names')
        intent_probs = query_intent_classifier(definitions, intent_tools, method='probs')
    else:
        intent_classes = [set() for i in range(len(definitions))]
    
    
    samples = []
    items = []
    for (word, definition) in zip(words, definitions):
        # making a sample out of the definition
        original_definition, preprocessed_definition, nparray = preprocess_definition(definition, max_seq_len, tools)
        preprocessed_definitions.append(preprocessed_definition)
        sample = nparray.reshape((-1, max_seq_len))
        samples.append(sample)
    samples = array(samples)
    samples = samples.reshape((-1, max_seq_len))
    # running the models on the sample
    preds = model.predict(samples).reshape((-1,output_emb_size))
    if 'attention_model' in models:
        weights = attention_model(samples).numpy().reshape((-1, max_seq_len)) # batch_size*1*
    else:
        weights = array([array([1/max_seq_len for kprim in range(max_seq_len)]) for k in range(len(samples))])
    # calculating the cosine distance between the output and each head word
    dists = dist.cdist(preds, comparison_matrix, metric="cosine")
    sims = 1 - dists
    del dists
    sims = nan_to_num(sims)

    # combine intent score -begin
    if 'intent_model' in globals():
        intent_c2id = {}
        for (intent_i, intent_c) in intent_id2c.items():
            intent_c2id[intent_c] = intent_i
        for sample_idx, (s, p) in enumerate(zip(sims, intent_probs)):
            for head_idx in range(len(s)):
                head_word = id2h[head_idx]
                head_word_pos_tags = word_tags[head_word] # a set
                head_word_pos_ids = [intent_c2id[tag] for tag in head_word_pos_tags if tag in intent_c2id]
                curr_intent_score = 0
                for hwp_id in head_word_pos_ids:
                    if p[hwp_id] >= intent_threshold:
                        curr_intent_score += p[hwp_id] 
                sims[sample_idx][head_idx] = sims[sample_idx][head_idx] + min(intent_coefficient, intent_coefficient*curr_intent_score)
    # combine intent score -end

    candidate_ids = flip(sims.argsort(), 1)
    del sims
    
    
    
    for (cid, weight, word, definition, preprocessed_definition, nparray, source, stags, gtags, intent_class) in zip(candidate_ids, weights, words, definitions, preprocessed_definitions, samples, sources, sense_tags, general_tags, intent_classes):
        

        # curr_def_is_a_word = False
        # if definition.strip() in h2id:
        #     curr_def_is_a_word = True
        # if curr_def_is_a_word == True:
        #     curr_def_id = h2id[definition.strip()]
        #     cid = [curr_id for curr_id in cid if curr_def_id!=curr_id]
        
        # remove all definition words from cid -begin
        definition_tokens = set(definition.split())
        definition_tokens = set([dt for dt in definition_tokens if dt in h2id])
        definition_tokens_ids = set([h2id[dt] for dt in definition_tokens])
        cid = [curr_id for curr_id in cid if curr_id not in definition_tokens_ids]
        # remove all definition words from cid -end
        
        
        cid = list(cid)
        # getting the top words from the ranking
        topwords = [id2h[idx] for idx in cid[:topn]]
        # calculating the ranks
        num_words = len(h2id)
        accepted_output_words = synonyms[word] if word in synonyms else set()
        accepted_output_words.add(word)
        try:
            main_word_rank = cid.index(h2id[word])+1
        except:
            main_word_rank = num_words
        accepted_output_ids = [h2id[w] for w in accepted_output_words if w in h2id and w!=original_definition.strip()]
        try:
            output_word_id, given_word_rank = get_rank(cid, accepted_output_ids)
        except:
            given_word_rank = num_words
            output_word_id = False
        # making the report item
        item = {
            'main_word':word,
            'original_definition':definition,
            'preprocessed_definition':preprocessed_definition,
            'array':nparray,
            'attention':weight,
            'intent':set(intent_class),
            'main_word_rank':main_word_rank,
            'synset_word_rank':given_word_rank,
            'synset_word':'UNK' if not output_word_id else id2h[output_word_id],
            'topwords':topwords,
            'source':source,
            'sense_tags':stags,
            'general_tags':gtags
        }
        items.append(item)
    return items



def evaluate(items):
    main_ranks = [item['main_word_rank'] for item in items]
    synset_ranks = [item['synset_word_rank'] for item in items]
    main_med = median(main_ranks)
    main_count10 = sum(i<=10 for i in main_ranks)
    main_count100 = sum(i<=100 for i in main_ranks)
    main_accuracy10 = main_count10/len(main_ranks)
    main_accuracy100 = main_count100/len(main_ranks)
    main_variance = var(main_ranks, ddof=1)
    
    synset_med = median(synset_ranks)
    synset_count10 = sum(i<=10 for i in synset_ranks)
    synset_count100 = sum(i<=100 for i in synset_ranks)
    synset_accuracy10 = synset_count10/len(synset_ranks)
    synset_accuracy100 = synset_count100/len(synset_ranks)
    synset_variance = var(synset_ranks, ddof=1)
    
    main_eval = {'median':main_med, 'variance':main_variance, 'acc@10':main_accuracy10, 'acc@100':main_accuracy100}
    synset_eval = {'median':synset_med, 'variance':synset_variance, 'acc@10':synset_accuracy10, 'acc@100':synset_accuracy100}
    
    all_sources = set([item['source'] for item in items])
    source_samples = {source:len([1 for item in items if item['source'] == source]) for source in all_sources}
    bad_samples = {source:len([1 for item in items if item['source'] == source and 10<item['synset_word_rank']<=100])/source_samples[source] for source in all_sources}
    
    return {'main_eval':main_eval, 'synset_eval':synset_eval,'bad_results':bad_samples}
    
def evaluate_intent_classifier(items, word_tags):
    true = 0
    for item in items:
        curr_word = item['main_word']
        curr_intent = item['intent']
        curr_word_tags = item['sense_tags'] if len(item['sense_tags'])>0 else item['general_tags']
        if len(curr_intent.intersection(curr_word_tags))>0:
            true += 1
    result = {'accuracy':true/len(items)}
    return result

def group_print_report(items):
    for item in items:
        html('کلمه اصلی: '+ item['main_word'])
        html('تعریف واقعی: ' + item['original_definition'])
        html('ورودی مدل: '+colorize(item['preprocessed_definition'], item['attention']))
        intent_str = "نقش دستوری: "+"-".join(item['intent'])
        html(intent_str)
        topwords_str = 'خروجی مدل: '+" -- ".join(item['topwords'])
        html(topwords_str)
        given_rank_str = 'مدل کلمه '+item['synset_word']+' را در رتبه '+str(item['synset_word_rank'])+' می دهد'
        html(given_rank_str)
        main_rank_str = 'مدل کلمه اصلی ('+item['main_word']+') را در رتبه '+str(item['main_word_rank'])+' می دهد'
        html(main_rank_str)
        html('---------------------------------------------------------')

def save_group_reports(items, report_dir):
    report_str = '<html><body>'
    div_begin = '<div style="text-align:right; direction:rtl;font-family:tahoma">'
    div_end = '</div>'
    new_line = '</br>'
    for item in items:
        report_str += div_begin+'کلمه اصلی: '+ item['main_word']+div_end+new_line
        report_str += div_begin+ 'تعریف واقعی: ' + item['original_definition']+div_end+new_line
        report_str += div_begin+ 'ورودی مدل: '+colorize(item['preprocessed_definition'], item['attention']) +div_end+new_line
        intent_str = "نقش دستوری: "+"-".join(item['intent'])
        report_str += div_begin+ intent_str + div_end+new_line
        topwords_str = 'خروجی مدل: '+" -- ".join(item['topwords'])
        report_str += div_begin + topwords_str + div_end+new_line
        synset_rank_str = 'مدل کلمه '+item['synset_word']+' را در رتبه '+str(item['synset_word_rank'])+' می دهد'
        report_str += div_begin + synset_rank_str+ div_end+new_line
        main_rank_str = 'مدل کلمه اصلی ('+item['main_word']+') را در رتبه '+str(item['main_word_rank'])+' می دهد'
        report_str += div_begin+ main_rank_str +div_end+new_line
        report_str += div_begin + '---------------------------------------------------------' + div_end
    report_str += '</body></html>'
    with open(report_dir, 'w',encoding='utf-8') as file:
        file.write(report_str)
    
def get_training_information(word, train_data):
    curr_data = train_data[word]
    html_txt = '<select style="font-family:tahoma"><option>'
    html_txt += 'نمونه‌های آموزشی مدل برای این کلمه: '+ str(len(curr_data)) + ' مورد'
    html_txt += '</option>'
    for d in curr_data:
        html_txt += "<option>" + d + "</option>"
    html_txt += "</select>"
    html(html_txt)
    
def get_training_data(word, train_original_data):
    curr_data = train_original_data[word]
    html_txt = '<select style="font-family:tahoma"><option>'
    html_txt += 'نمونه‌های آموزشی مدل برای این کلمه: '+ str(len(curr_data)) + ' مورد'
    html_txt += '</option>'
    for d in curr_data:
        html_txt += "<option>" + d + "</option>"
    html_txt += "</select>"
    html(html_txt)
    
def get_testing_information(word, test_data):
    curr_data = test_data[word]
    html_txt = '<select style="font-family:tahoma"><option>'
    html_txt += 'نمونه‌های آزمایشی مدل برای این کلمه: '+ str(len(curr_data)) + ' مورد'
    html_txt += '</option>'
    for d in curr_data:
        html_txt += "<option>" + d + "</option>"
    html_txt += "</select>"
    html(html_txt)
    
def get_testing_data(word, test_original_data):
    curr_data = test_original_data[word]
    html_txt = '<select style="font-family:tahoma"><option>'
    html_txt += 'نمونه‌های آزمایشی مدل برای این کلمه: '+ str(len(curr_data)) + ' مورد'
    html_txt += '</option>'
    for d in curr_data:
        html_txt += "<option>" + d + "</option>"
    html_txt += "</select>"
    html(html_txt)
    
def prepare_survey_csv(words, definitions, topn, max_seq_len, model, attention_model, tools):
    globals().update(tools)
    h2id = {value:key for (key, value) in id2h.items()}
    num_words = len(h2id)
    items = []
    for (word, definition) in zip(words, definitions):
        try:
            original_definition, preprocessed_definition, nparray = preprocess_definition(definition, max_seq_len, tools)
            item = {
            'word':word,
            'original_definition':original_definition,
            'preprocessed_definition':preprocessed_definition,
            'array':nparray
            }
            items.append(item)
        except:
            continue
    samples = np.array([item['array'] for item in items])
    samples = samples.reshape((-1, max_seq_len))
    pred = model.predict(samples)
    attention_pred = attention_model.predict(samples)
    dists = dist.cdist(pred, comparison_matrix, metric="cosine")
    sims = 1 - dists
    del dists
    sims = nan_to_num(sims)
    candidate_ids = flip(sims.argsort(), 1)
    del sims
    given_ranks = []
    main_ranks = []
    report_items = []
    for (cid, item, pred) in zip(candidate_ids, items, attention_pred):
        
        
        report_item = {}
        report_item['attention'] = pred
        cid = list(cid)
        
        curr_def_is_a_word = False
        if item['original_definition'].strip() in h2id:
            curr_def_is_a_word = True
        if curr_def_is_a_word == True:
            curr_def_id = h2id[item['original_definition'].strip()]
            cid = [curr_id for curr_id in cid if curr_id != curr_def_id]
        
        intent = intent_classifier(item['original_definition'], word_tags)
        intent = set([tag for tag in intent.keys() if intent[tag]>0])
        cid = update_ranking_based_on_intent(cid, intent, id2h, word_tags)
        true_output_word = item['word']
        accepted_output_words = synonyms[true_output_word] if true_output_word in synonyms else set()
        accepted_output_words.add(true_output_word)
        try:
            main_word_rank = cid.index(h2id[true_output_word])+1
        except:
            main_word_rank = num_words
        accepted_output_ids = [h2id[w] for w in accepted_output_words if w in h2id and w!=item['original_definition'].strip()]
        report_item['main_word'] = true_output_word
        try:
            output_word_id, given_word_rank = get_rank(cid, accepted_output_ids)
        except:
            given_word_rank = num_words
            output_word_id = 1
            
        report_item['given_word_rank'] = given_word_rank
        report_item['main_word_rank'] = main_word_rank
        report_item['given_word'] = id2h[output_word_id]
        report_item['intent'] = intent
        main_ranks.append(main_word_rank)
        given_ranks.append(given_word_rank)
        input_definition = item['preprocessed_definition']
        report_item['preprocessed_definition'] = input_definition
        original_definition = item['original_definition']
        report_item['original_definition'] = original_definition
        report_item['topwords'] = [id2h[idx] for idx in cid[:topn]]
        report_items.append(report_item)
    result = [{'input':item['original_definition'],'words':item['topwords']} for item in report_items]
    lists = [[] for i in range(topn)]
    for item in result:
        curr_input = item['input']
        for i, word in enumerate(item['words']):
            curr_str = word+','+curr_input
            lists[i].append(curr_str)
    strings = []
    for lst in lists:
        lst_str = "\n".join(lst)
        strings.append(lst_str)
    return strings
    
def stratified_sample(items, sources, num_head_words, sample_size):
    items = [item for item in items if item['source'] in sources]
    num_items = num_head_words
    section_size = int(num_items/sample_size)
    sections = [[] for i in range(sample_size)]
    for i in range(sample_size):
        sections[i] = [item for item in items if section_size*i<=item['rank']<=section_size*(i+1)]
    samples = [sample(sections[i], 1)[0] if len(sections[i])>0 else sample(sections[i-1]+sections[i+1],1)[0] for i in range(sample_size)]
    return samples

def save_optimizer_state(model, path):
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
    with open(path, 'wb') as f:
        pickle.dump(weight_values, f)