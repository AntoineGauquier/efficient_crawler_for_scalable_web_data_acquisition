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
from collections import deque
from urllib.parse import urljoin, urldefrag, quote, unquote, urlparse, urlunparse
import heapq

class OfflineDOMCrawler(GenericLocalCrawler):
    def __init__(self, db_path, table_name, log_path, starting_url, budget, use_url_classifier, classifier_params, threshold, m, w, n_grams_path_dom_path_representation):
        super().__init__(db_path=db_path, 
            table_name=table_name, 
            log_path=log_path, 
            starting_url=starting_url, 
            use_url_classifier=use_url_classifier, 
            classifier_params=classifier_params, 
            is_standard_baseline=False, 
            is_offline_baseline=True, 
            budget=budget)

        logging.info("-\tsimilarity threshold:" + str(threshold))
        logging.info("-\tm:" + str(m))
        logging.info("-\tw:" + str(w))
        logging.info("-\tn in n_grams used in DOM path vector representation:" + str(n_grams_path_dom_path_representation))


        self.frontier = deque()

        self.actions = []
        self.id_last_chosen_np = None
        self.n_grams_path_dom_path_representation = n_grams_path_dom_path_representation
        
        self.vocabulary_dict = dict()
        self.len_vocabulary_dict = 0

        self.m = m
        self.w = w
        self.threshold = threshold
        self.a = 766245317

        self.n_grams_path_dom_path_representation = n_grams_path_dom_path_representation
        self.index_structure = initialize_index_structure(self.m)

        self.path_to_action_structure = {}

    def get_next_link(self):
        chosen_link = None
        if self.nb_episodes <= 3000:
            chosen_link = self.frontier.popleft()
            self.learning_phase_iteration(chosen_link)
        else:
            chosen_link = heapq.heappop(self.frontier)

        return chosen_link

    def is_crawling_terminated(self):
        return len(self.frontier) == 0

    def add_link(self, link):
        link, self.actions, self.vocabulary_dict, self.len_vocabulary_dict, self.index_structure, self.path_to_action_structure = map_path_to_action(
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
            is_offline_baseline = True, 
            path_to_action_structure = self.path_to_action_structure)
        
        if self.nb_episodes <= 3000:
            self.frontier.append(link)
        else:
            link.score = self.actions[self.path_to_action_structure[link.dom_path]].value
            heapq.heappush(self.frontier, link)

    def modify_action_when_forcing_crawling(self, url):
        pass

    def is_link_in_frontier(self, link):
        return link in self.frontier

    def update_score(self, reward, idx_np):
        self.actions[idx_np].nb_times_selected += 1
        self.actions[idx_np].update_value_mean(reward)

    def update_score_resource_not_html(self):
        pass

    def save_actions(self, log_path):
        np.save(os.path.join(log_path, "actions.npy"), self.actions)

    def get_current_action(self):
        return self.id_last_chosen_np 

    def override_current_action(self, new_action):
        self.id_last_chosen_np = new_action

    def update_data_structure_offline_learning(self):
        if self.nb_episodes == 3000:
            new_data_structure = []
            for element in self.frontier:
                element.score = self.actions[self.path_to_action_structure[element.dom_path]].value 
                heapq.heappush(new_data_structure, element)
            self.frontier = new_data_structure

    def learning_phase_iteration(self, link):
            score = 0

            cursor = self.connection.cursor()
            query_str = "SELECT http_response, headers, body FROM {} WHERE url=?".format(self.table_name)
            query = cursor.execute(query_str, (link.get_url(),))
            result = query.fetchall()

            if len(result) > 0:

                result = result[0]
                http_response = result[0]
                headers = result[1]
                body = result[2]
           
                if http_response.startswith("2"):
                    if len(body) > 0 and not body.isspace():
                        new_elements, base_url = self.extract_urls_from_html(body)

                    for new_element in new_elements:
                        if base_url is None:
                            try:
                                full_new_url = urldefrag(urljoin(link.get_url(), new_element['url'])).url
                            except:
                                logging.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" . It probably contains unhandled characters under NKFC normalization. Probably due to an error in the HTML code itself.")
                                continue     
                        else:
                            try:
                                full_new_url = urldefrag(urljoin(base_url, new_element['url'])).url
                            except:
                                logging.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" . It probably contains unhandled characters under NKFC normalization. Probably due to an error in the HTML code itself.")
                                continue

                        if not self.is_url_on_same_or_sub_domain(self.starting_url, full_new_url):#not is_on_same_domain:
                            continue

                        new_link = Link(full_new_url, new_element['dom_path'])

                        cursor = self.connection.cursor()
                        query_str_new = "SELECT http_response, headers FROM {} WHERE url=?".format(self.table_name)
                        query_new = cursor.execute(query_str_new, (new_link.get_url(),))
                        result_new = query_new.fetchall()
                
                        if len(result_new) > 0:
                            result_new = result_new[0]
                            http_response_new = result_new[0]
                            headers_new = result_new[1]

                            headers_new = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers_new)).items()}

                            if 'Content-Type' in headers_new:
                                mime_type = headers_new['Content-Type'][0].decode('utf-8')
                                if ";" in mime_type:
                                    mime_type = mime_type.split(";")[0]

                                if mime_type in self.mime_types_data_resources:
                                    score += 1
            self.update_score(score, link.associated_np_idx)

if __name__ == "__main__":

    if len(sys.argv) < 9:
        print("Usage: python3 " + sys.argv[0] + " <similarity_threshold> <m> <w> <budget:-1 for unlimited> <n in n_grams used for DOM path vector representation> <batch_size_url> <log_path_all_sites> <num_executions> <site_name>")
    else:
        threshold = float(sys.argv[1])
        m = int(sys.argv[2])
        w = int(sys.argv[3])
        budget = int(sys.argv[4])
        n_grams_path_dom_path_representation = int(sys.argv[5])
        batch_size_url = int(sys.argv[6])
        log_path_all_sites = sys.argv[7]
        site_name = sys.argv[8]

        db_path = os.path.join(os.getcwd(), "data", str(site_name) + ".db")
        table_name = site_name
        log_path = os.path.join(os.getcwd(), "crawlers/offline_dom/logs", str(log_path_all_sites), str(site_name))
       
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        co = sqlite3.connect(os.path.join(os.getcwd(),"data", "site_names_to_urls.db"))
        cu = co.cursor()

        cu.execute("SELECT start_url FROM site_names_to_urls WHERE site_name LIKE \"" + site_name + "\";")
        res = cu.fetchall()[0]

        starting_url = res[0]


        if budget == -1:
            budget = 1e10

        use_url_classifier = True
        classifier_params = {'max_iter_optimizer':100, 'random_state_optimizer':42, 'n_in_ngrams':2, 'batch_size':batch_size_url}

        crawler = OfflineDOMCrawler(db_path=db_path, 
            table_name=table_name, 
            log_path=log_path, 
            starting_url=starting_url, 
            budget=budget, 
            use_url_classifier=use_url_classifier, 
            classifier_params=classifier_params, 
            threshold=threshold, 
            m=m, 
            w=w, 
            n_grams_path_dom_path_representation=n_grams_path_dom_path_representation)
        crawler.crawl()
