import sys
import os
sys.path.append(os.path.join(os.getcwd(),'crawlers'))

from generic_local_crawler import *
from rl_actions import *
import math
import numpy as np
from random import choice
from tqdm import tqdm
import multiprocessing
import cProfile
import sqlite3
import os

class AUERCrawler(GenericLocalCrawler):
    def __init__(self, db_path, table_name, log_path, starting_url, budget, alpha, use_url_classifier, classifier_params, threshold, m, w, n_grams_path_dom_path_representation):
        super().__init__(db_path=db_path, 
            table_name=table_name, 
            log_path=log_path, 
            starting_url=starting_url, 
            use_url_classifier=use_url_classifier, 
            classifier_params=classifier_params, 
            is_standard_baseline=False, 
            is_offline_baseline=False, 
            budget=budget)

        logging.info("-\talpha (exploration-exploitation coefficient trade-off):" + str(alpha))
        logging.info("-\tsimilarity threshold:" + str(threshold))
        logging.info("-\tm:" + str(m))
        logging.info("-\tw:" + str(w))
        logging.info("-\tn in n_grams used in DOM path vector representation:" + str(n_grams_path_dom_path_representation))


        self.frontier = dict()
        self.frontier_length = 0
        self.actions = []
        self.id_last_chosen_np = None
        self.alpha = alpha
        self.n_grams_path_dom_path_representation = n_grams_path_dom_path_representation
		
        self.vocabulary_dict = dict()
        self.len_vocabulary_dict = 0

        self.m = m
        self.w = w
        self.threshold = threshold
        self.a = 766245317

        self.n_grams_path_dom_path_representation = n_grams_path_dom_path_representation
        self.index_structure = initialize_index_structure(self.m)

    def get_next_link(self):
        chosen_link = None
        id_chosen_np = np.argmax([np.get_score(self.alpha, self.nb_episodes) for np in self.actions])
        if isinstance(id_chosen_np, list):
            id_chosen_np = choice(id_chosen_np)

        candidates = self.frontier[id_chosen_np]
        if len(candidates) > 0:
            chosen_link = choice(tuple(candidates))
        
        self.frontier[id_chosen_np].remove(chosen_link)
        self.frontier_length -= 1       
 
        self.id_last_chosen_np = id_chosen_np
        return chosen_link

    def is_crawling_terminated(self):
        return self.frontier_length == 0

    def add_link(self, link):
        link, self.actions, self.vocabulary_dict, self.len_vocabulary_dict, self.index_structure = map_path_to_action(
            p = link, 
            actions = self.actions, 
            n = self.n_grams_path_dom_path_representation, 
            a = self.a, 
            m = self.m, 
            w = self.w, 
            threshold = self.threshold, 
            vocab_dict = self.vocabulary_dict, 
            len_vocab = self.len_vocabulary_dict, 
            index_structure = self.index_structure, 
            is_offline_baseline = False, 
            path_to_action_structure = None)
        
        if link.associated_np_idx not in self.frontier:
            self.frontier[link.associated_np_idx] = set()
        self.frontier[link.associated_np_idx].add(link)
        self.frontier_length += 1

    def modify_action_when_forcing_crawling(self, link):
        link, self.actions, self.vocabulary_dict, self.len_vocabulary_dict, self.index_structure = map_path_to_action(p = link, actions = self.actions, n = self.n_grams_path_dom_path_representation, a = self.a, m = self.m, w = self.w, threshold = self.threshold, vocab_dict = self.vocabulary_dict, len_vocab = self.len_vocabulary_dict, index_structure = self.index_structure, is_offline_baseline = False, path_to_action_structure = None)

        idx_new_last_chosen_id = -1
        for idx in range(len(self.actions)):
            if  idx == link.associated_np_idx:
                idx_new_last_chosen_id = idx
                break

        self.id_last_chosen_np = idx_new_last_chosen_id

    def is_link_in_frontier(self, link):
        for key in self.frontier:
            if link in self.frontier[key]:        
                return True
        return False	

    def update_score(self, reward, idx_np=None):
        self.actions[self.id_last_chosen_np].remove_link_from_np()
        self.actions[self.id_last_chosen_np].np_is_selected()
        self.actions[self.id_last_chosen_np].update_value_mean(reward)

    def update_score_resource_not_html(self):
        if len(self.actions) > 0: # Filters the scenario where we have a redirection on starting URL: in such a case, there is no action
            self.actions[self.id_last_chosen_np].remove_link_from_np()

    def save_actions(self, log_path):
        np.save(os.path.join(log_path, "actions.npy"), self.actions)

    def get_current_action(self):
        return self.id_last_chosen_np 

    def override_current_action(self, new_action):
        self.id_last_chosen_np = new_action

    def update_data_structure_offline_learning(self):
        pass

if __name__ == "__main__":
    threshold = float(sys.argv[1])
    m = int(sys.argv[2])
    w = int(sys.argv[3])
    budget = int(sys.argv[4])
    n_grams_path_dom_path_representation = int(sys.argv[5])
    batch_size_url = int(sys.argv[6])
    use_url_classifier = int(sys.argv[7])
    alpha = sys.argv[8]
    site_name = sys.argv[9]
    log_path_all_sites = sys.argv[10]

    db_path = os.path.join(os.getcwd(), "data", str(site_name) + ".db")
    table_name = site_name
    log_path = os.path.join(os.getcwd(), "crawlers/auer/logs", str(log_path_all_sites), str(site_name))
   
    if not os.path.exists(os.path.join(os.getcwd(), "crawlers","auer","logs")):
        os.mkdir(os.path.join(os.getcwd(), "crawlers","auer","logs"))

    if not os.path.exists(os.path.join(os.getcwd(), "crawlers","auer","logs", str(log_path_all_sites))):
        os.mkdir(os.path.join(os.getcwd(), "crawlers","auer","logs", str(log_path_all_sites)))    

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    co = sqlite3.connect(os.path.join(os.getcwd(),"data", "site_names_to_urls.db"))
    cu = co.cursor()

    cu.execute("SELECT start_url FROM site_names_to_urls WHERE site_name LIKE \"" + site_name + "\";")
    res = cu.fetchall()[0]

    starting_url = res[0]

    if alpha == "2s2":
        alpha = 2 * math.sqrt(2)
    else:
        alpha = float(alpha)

    if budget == -1:
        budget = 1e10

    if use_url_classifier == 1:
        use_url_classifier = True
    else:
        use_url_classifier = False

    classifier_params = {'max_iter_optimizer':100, 'random_state_optimizer':42, 'n_in_ngrams':2, 'batch_size':batch_size_url}

    crawler = AUERCrawler(db_path=db_path, 
        table_name=table_name, 
        log_path=log_path, 
        starting_url=starting_url, 
        budget=budget, 
        alpha=alpha, 
        use_url_classifier=use_url_classifier, 
        classifier_params=classifier_params, 
        threshold=threshold, 
        m=m, 
        w=w, 
        n_grams_path_dom_path_representation=n_grams_path_dom_path_representation)
    crawler.crawl()
