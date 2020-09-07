import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from nltk import pos_tag, word_tokenize
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def second_longest_sentence(df):
    longest_sentence = 0
    second_longest = 0
    for sents in df['headline'].values:
        if len(sents.split()) > longest_sentence:
            second_longest = longest_sentence
            longest_sentence = len(sents.split())
    return second_longest

def word2vec(word, word_embedding_model):
    try:
        return word_embedding_model[word]
    except:
        return word_embedding_model['unk']

def sentence2vecs(sentence, word_embedding_model):
    return [word2vec(x, word_embedding_model) for x in sentence.split()]

def tag_to_num(tag):
    pos_tags_list = ['CC', 'CD', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS','NNP', \
                  'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', \
                  'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'] 
    len_ = len(pos_tags_list)
    mean_ = np.mean(np.arange(len_))
    std_ =  np.std(np.arange(len_))
    
    if tag in pos_tags_list:
        tag_emb =  (pos_tags_list.index(tag)+1)
        return (tag_emb - mean_)/(2*std_)  # zero mean and std=1
    else:
        return 0

def gen_data(df, word_embedding_model, gen_pos_data=False, keep_len_info=False):
    
    corpus_size = df['headline'].size
    split_factor = 8

    word2vec_model_size = len(word_embedding_model['unk'])
    model_sentence_length = second_longest_sentence(df)

    if gen_pos_data:
        data_padded = np.zeros((corpus_size, model_sentence_length , word2vec_model_size+1))
    else:
        data_padded = np.zeros((corpus_size, model_sentence_length , word2vec_model_size))
        
    if keep_len_info:  # keep sizes for later analysis of model performance 
        labels = np.zeros((corpus_size, 2), dtype=np.int)
    else:
        labels = np.zeros(corpus_size, dtype=np.int)
    
    indices = np.arange(corpus_size)
    
    for ii in range(df.shape[0]):
        sents = df['headline'].iloc[ii]
        tok_sents = word_tokenize(sents)
        if keep_len_info:  
            labels[ii] = [df['is_hate'].iloc[ii], len(tok_sents)]
        else:
            labels[ii]= df['is_hate'].iloc[ii]

        vectors = sentence2vecs(sents, word_embedding_model) 
        for jj, vector in enumerate(vectors):
            if jj == model_sentence_length:
                break
            vec = np.asarray(vector)
            if gen_pos_data:
                vec = np.hstack((vec, tag_to_num(pos_tag([tok_sents[jj]])[0][1]))) 
            data_padded[ii, jj,:] = vec
    
    training_samples  = int(corpus_size*split_factor/10)
    validation_samples = int((corpus_size - training_samples)/2)
    test_samples = int((corpus_size - training_samples)/2)
    
    np.random.seed(42)
    np.random.shuffle(indices)
    data_padded = data_padded[indices]
    
    labels = labels[indices]
    X_train = data_padded[:training_samples]
    y_train = labels[:training_samples]

    X_val = data_padded[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    X_test = data_padded[training_samples+validation_samples: training_samples + validation_samples+test_samples]
    y_test = labels[training_samples+validation_samples: training_samples + validation_samples+test_samples]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, model_sentence_length

def gen_bert_data(df, tokenizer):
	
	sentences = df.headline.values
	labels = df.is_hate.values

	max_len = 0

	for sent in sentences:

	    input_ids = tokenizer.encode(sent, add_special_tokens=True)
	    max_len = max(max_len, len(input_ids))

	input_ids = []
	attention_masks = []

	for sent in sentences:
	    encoded_dict = tokenizer.encode_plus(
	                        sent,                     
	                        add_special_tokens = True, 
	                        max_length = 256, 
	                        pad_to_max_length = True,
	                        return_attention_mask = True,  
	                        return_tensors = 'pt',   
	                        truncation=True
	                   )
	    
	    input_ids.append(encoded_dict['input_ids'])
	    attention_masks.append(encoded_dict['attention_mask'])

	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)

	dataset = TensorDataset(input_ids, attention_masks, labels)

	train_size = int(0.8 * len(dataset))
	val_size = int((len(dataset) - train_size) / 2)
	test_size = val_size
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size+1])

	print('{:>5,} training samples'.format(train_size))
	print('{:>5,} validation samples'.format(val_size))
	print('Original: ', sentences[0])
	print('Token IDs:', input_ids[0])

	print('Max sentence length: ', max_len)
	batch_size = 16

	train_dataloader = DataLoader(
	            train_dataset, 
	            sampler = RandomSampler(train_dataset), 
	            batch_size = batch_size 
	        )

	validation_dataloader = DataLoader(
	            val_dataset, 
	            sampler = SequentialSampler(val_dataset),
	            batch_size = batch_size
	        )
	test_dataloader =  DataLoader(
	            test_dataset,
	            sampler = SequentialSampler(test_dataset),
	            batch_size = batch_size 
	        )

	return train_dataloader, validation_dataloader, test_dataloader

def plot_graphs(best_model):
    plt.figure(figsize=(20, 12))
    plt.subplot(311)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,best_model.epochs_used+1),best_model.train_acc,label="Train")
    plt.plot(range(1,best_model.epochs_used+1),best_model.val_acc,label="Validate")
    plt.xticks(np.arange(1, best_model.epochs_used+1, 1.0))
    plt.legend()
    plt.subplot(312)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Epoch average)")
    plt.plot(range(1,best_model.epochs_used+1),best_model.train_loss,label="Train")
    plt.plot(range(1,best_model.epochs_used+1),best_model.val_loss,label="Validate")
    plt.xticks(np.arange(1, best_model.epochs_used+1, 1.0))
    plt.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)



