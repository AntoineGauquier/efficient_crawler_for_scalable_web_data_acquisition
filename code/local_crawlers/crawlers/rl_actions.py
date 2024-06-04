import math
import numpy as np
import hnswlib

class Action:
    def __init__(self, id, label, a, m, w, hash_vector):
        self.id = id
        self.label = label
        self.nb_times_selected = 0
        self.value = 0
        self.nb_associated_links = 1
        self.epsilon = 10e-7
        self.centroid_vector = hash_vector
        self.number_of_associated_vectors = 1
        
    def __str__(self):
        return "[label=" + self.label + ", nb_times_selected=" + str(self.nb_times_selected) + ", value=" + str(self.value) + ", nb_associated_links=" + str(self.nb_associated_links) + "]"

    def __eq__(self, other):
        return self.label == other.label

    def add_link_to_np(self):
        self.nb_associated_links += 1

    def remove_link_from_np(self):
        #print("Removed link from NP.")
        self.nb_associated_links -= 1

    def np_is_selected(self):
        self.nb_times_selected += 1

    def get_score(self, alpha, t):
        if self.nb_associated_links == 0 or t == 0: # t == 0 being the case where we start the crawling so the score can be arbitrary
            return 0
        return self.value + alpha * math.sqrt(math.log(t)/(self.nb_times_selected + self.epsilon))

    def update_value_mean(self, reward):
        self.value += (1/(self.nb_times_selected)) * (reward - self.value)

    def update_vectors(self, new_hashed_vector, index_structure):
        index_structure.mark_deleted(self.id)
        self.centroid_vector = (self.number_of_associated_vectors * self.centroid_vector + new_hashed_vector) / (self.number_of_associated_vectors + 1)
        self.number_of_associated_vectors += 1
        return add_entry_to_index_structure(self.id, self.centroid_vector, index_structure)
         
def initialize_index_structure(m):
    index_structure = hnswlib.Index(space='cosine', dim=2**m)
    index_structure.init_index(max_elements=100000, allow_replace_deleted = True)
    return index_structure

def add_entry_to_index_structure(idx, vector, index_structure):
    index_structure.add_items(data=vector, ids=[idx], replace_deleted = True)
    return index_structure

def multiplicative_hash(x, a, m, w):
    return math.floor(((x*a)%(2**w))/(2**(w-m)))

def tf_vector_to_hash_vector(tf_vector, a, m, w):
    hash_table = dict()
    for dim in range(len(tf_vector)):
        hashed_dim = multiplicative_hash(dim, a, m, w)
        if hashed_dim in hash_table:
            hash_table[hashed_dim].append(tf_vector[dim])
        else:
            hash_table[hashed_dim] = [tf_vector[dim]]
    
    hash_vector = [0 for _ in range(2**m)]
    for key in hash_table:
        values = hash_table[key]
        if len(values) > 1:
            hash_vector[key] = np.mean(values)
        else:
            hash_vector[key] = values[0]

    return np.array(hash_vector)

def map_path_to_action(p, actions, n, a, m, w, threshold, vocab_dict, len_vocab, index_structure, is_offline_baseline, path_to_action_structure):
    vocab_dict, len_vocab, tf_vector_p = get_tf_vector(p.dom_path, n, vocab_dict, len_vocab)
    hash_vector_p = tf_vector_to_hash_vector(tf_vector_p, a, m, w)
    similarity = 0

    if len(actions) > 0:
        idx_nearest, distance_nearest = index_structure.knn_query(data=hash_vector_p, k = 1)
        idx_nearest = int(idx_nearest[0][0])
        similarity = 1 - distance_nearest[0][0] # Index structure is initialized with cosine distance, so its similarity is just 1 - distance. 
    if similarity >= threshold:
        index_structure = actions[idx_nearest].update_vectors(hash_vector_p, index_structure)
        p.set_associated_np(idx_nearest)
        actions[idx_nearest].add_link_to_np()
        if is_offline_baseline:
            path_to_action_structure[p.dom_path] = idx_nearest
    else:
        p.set_associated_np(len(actions))
        new_np = Action(len(actions), p.dom_path, a, m, w, hash_vector_p)
        index_structure = add_entry_to_index_structure(new_np.id, new_np.centroid_vector, index_structure)
        if is_offline_baseline:
            path_to_action_structure[p.dom_path] = new_np.id
        actions.append(new_np)
    if is_offline_baseline:
        return p, actions, vocab_dict, len_vocab, index_structure, path_to_action_structure
    return p, actions, vocab_dict, len_vocab, index_structure

def get_tf_vector(p, n, vocab_dict, len_vocab):
    tokenized_p = tokenize(p, n)
    len_p = len(tokenized_p)
    tf_vector = [0 for _ in range(len_vocab)]

    for token in tokenized_p:
        id_token_if_exists = vocab_dict.get(token)
        if id_token_if_exists != None:
            tf_vector[id_token_if_exists] += (1 / len_p)
        else:
            vocab_dict[token] = len_vocab
            len_vocab += 1
            tf_vector.append(1/len_p)

    return vocab_dict, len_vocab, np.array(tf_vector)

def tokenize(p, n): 
    list_1_gram = p.split("/")
    list_1_gram.insert(0, "[SOS]")
    list_1_gram.append("[EOS]") 

    if n == 1:
        return list_1_gram

    list_n_grams = []
    for idx in range(len(list_1_gram) - (n-1)):
        token = list_1_gram[idx]
        for shift in range(1, n): # {1, ..., (n-1)}
            token += "/" + list_1_gram[idx + shift]
        list_n_grams.append(token)

    return list_n_grams