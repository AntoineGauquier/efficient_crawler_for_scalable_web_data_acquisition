import math
import numpy as np
import hnswlib

class Action:
    def __init__(self, action_id, label, a, m, w, hash_vector):
        self.id = action_id
        self.label = label
        self.nb_times_selected = 0
        self.value = 0
        self.epsilon = 10e-7
        self.centroid_vector = hash_vector
        self.number_of_available_links = 1
        self.number_of_associated_vectors = 1
        self.is_deleted = False

    def __str__(self):
        return "[id=" + str(self.id) + ", nb_times_selected=" + str(self.nb_times_selected) + ", value=" + str(self.value) + ", number_of_available_links=" + str(self.number_of_available_links) + ", number_of_associated_vectors="+ str(self.number_of_associated_vectors) + ", centroid=" + str(self.centroid_vector) + "]"

    def __eq__(self, other):
        return self.id == other.id

    def add_link_to_np(self):
        self.number_of_available_links += 1

    def remove_link_from_np(self):
        self.number_of_available_links -= 1

    def np_is_selected(self):
        self.nb_times_selected += 1

    def get_score(self, alpha, t):
        if self.number_of_available_links == 0 or t == 0: # t == 0 being the case where we start the crawling so the score can be arbitrary
            return 0
        return self.value + alpha * math.sqrt(math.log(t)/(self.nb_times_selected + self.epsilon))

    def update_value_mean(self, reward):
        self.value += (1/(self.nb_times_selected)) * (reward - self.value)

    def update_vectors(self, new_hashed_vector, actions, index_structure):
        if self.is_deleted:
            self.centroid_vector = new_hashed_vector
            self.is_deleted = False
        else:
            self.centroid_vector = (self.number_of_associated_vectors * self.centroid_vector + new_hashed_vector) / (self.number_of_associated_vectors + 1)
        
        self.number_of_associated_vectors += 1

        nb_elements_in_index_before_update = index_structure.get_current_count()

        return add_entry_to_index_structure(self.id, self.centroid_vector, index_structure, nb_elements_in_index_before_update, True)
         
def initialize_index_structure(m):
    index_structure = hnswlib.Index(space='cosine', dim=2**m)
    index_structure.init_index(max_elements=1000000, allow_replace_deleted = True)
    index_structure.set_num_threads(1)
    return index_structure

def add_entry_to_index_structure(idx, vector, index_structure, nb_elements_index, from_update_vectors):
    
    index_structure.add_items(data=vector, ids=[idx], replace_deleted = True)
    if from_update_vectors: # Updating existing action
        nb_elements_in_index_after_update = index_structure.get_current_count()
        assert(nb_elements_index == nb_elements_in_index_after_update)
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

def get_elements_from_index_not_marked_deleted(actions, index_structure):
    not_marked_deleted = []
    similarities = []
    for action_id in actions:
        labels, distances = index_structure.knn_query(data=actions[action_id].centroid_vector, k=1)
        if labels[0] == action_id:
            not_marked_deleted.append(action_id)
            similarities.append(1 - distances[0][0])

    return not_marked_deleted, similarities

def unmap_path_to_action(p, actions, n, a, m, w, vocab_dict, len_vocab, index_structure, frontier):
    action_to_be_modified = actions[p.associated_np_idx]
    action_to_be_modified.number_of_associated_vectors -= 1

    k = action_to_be_modified.number_of_associated_vectors 
    action_to_be_modified.remove_link_from_np()

    if k >= 0: # If we deleted the only vector that was associated with the action, then it means that this action should not have been created in the first place, and we simply delete it.
        vocab_dict, len_vocab, tf_vector_p = get_tf_vector(p.dom_path, n, vocab_dict, len_vocab)
        hash_vector_p = tf_vector_to_hash_vector(tf_vector_p, a, m, w)
        if k == 0:
            action_to_be_modified.centroid_vector = None
            action_to_be_modified.is_deleted = True

        else:
            new_centroid_vector = ((k + 1) / k) * action_to_be_modified.centroid_vector - (hash_vector_p / k)  
            action_to_be_modified.centroid_vector = new_centroid_vector

            nb_elements_in_index_before_update = index_structure.get_current_count()
            index_structure.add_items(data=action_to_be_modified.centroid_vector, ids=[action_to_be_modified.id], replace_deleted = True)
            nb_elements_in_index_after_update = index_structure.get_current_count()
        
            assert(nb_elements_in_index_before_update == nb_elements_in_index_after_update)

    else:
        assert(False)
        nb_elements_in_index_before_marked_deleted = index_structure.get_current_count()
        index_structure.mark_deleted(action_to_be_modified.id)
        nb_elements_in_index_after_marked_deleted = index_structure.get_current_count()
        assert(nb_elements_in_index_before_marked_deleted == nb_elements_in_index_after_marked_deleted)
        elements_in_actions_before_deletion = [key for key in actions]
        del actions[p.associated_np_idx]
        del frontier[p.associated_np_idx]
        elements_in_actions_after_deletion = [key for key in actions]
        assert((len(elements_in_actions_before_deletion) - 1) == len(elements_in_actions_after_deletion))
        assert(p.associated_np_idx not in elements_in_actions_after_deletion)
    return p, actions, vocab_dict, len_vocab, index_structure, frontier

def map_path_to_action(p, actions, new_action_id, n, a, m, w, threshold, vocab_dict, len_vocab, index_structure, is_offline_baseline, path_to_action_structure):
    vocab_dict, len_vocab, tf_vector_p = get_tf_vector(p.dom_path, n, vocab_dict, len_vocab)
    hash_vector_p = tf_vector_to_hash_vector(tf_vector_p, a, m, w)
    similarity = 0

    if len(actions) > 0:
        idx_nearest, distance_nearest = index_structure.knn_query(data=hash_vector_p, k = 1)
        idx_nearest = int(idx_nearest[0][0])
        similarity = 1 - distance_nearest[0][0] # Index structure is initialized with cosine distance, so its similarity is just 1 - distance. 
    if similarity >= threshold:
        index_structure = actions[idx_nearest].update_vectors(hash_vector_p, actions, index_structure)
        p.set_associated_np(idx_nearest)
        actions[idx_nearest].add_link_to_np()
        if is_offline_baseline:
            path_to_action_structure[p.dom_path] = idx_nearest
    else:
        elements_in_action_before = [key for key in actions]
        assert(new_action_id not in elements_in_action_before)
        p.set_associated_np(new_action_id)
        new_np = Action(new_action_id, p.dom_path, a, m, w, hash_vector_p)

        nb_elements_in_index_before_adding_action = index_structure.get_current_count()
        index_structure = add_entry_to_index_structure(new_np.id, new_np.centroid_vector, index_structure, len(index_structure.get_ids_list()), False)
        nb_elements_in_index_after_adding_action = index_structure.get_current_count()
        if is_offline_baseline:
            path_to_action_structure[p.dom_path] = new_np.id
        actions[new_action_id] = new_np
        elements_in_action_after = [key for key in actions]
        assert((len(elements_in_action_before) + 1) == len(elements_in_action_after))
        assert((nb_elements_in_index_before_adding_action + 1) == nb_elements_in_index_after_adding_action)
        new_action_id += 1
    if is_offline_baseline:
        return p, actions, new_action_id, vocab_dict, len_vocab, index_structure, path_to_action_structure
    
    return p, actions, new_action_id, vocab_dict, len_vocab, index_structure

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
