import sys
import os
sys.path.append(os.path.join(os.getcwd(),'local_crawlers'))

import os
import sqlite3
from generic_local_crawler import GenericLocalCrawler
from collections import deque

class DFSCrawler(GenericLocalCrawler):
    def __init__(self, db_path, table_name, log_path, starting_url, use_url_classifier, classifier_params, budget=10e7):
        super().__init__(db_path=db_path, 
            table_name=table_name, 
            log_path=log_path, 
            starting_url=starting_url, 
            use_url_classifier=use_url_classifier, 
            classifier_params=classifier_params, 
            is_standard_baseline=True, 
            is_offline_baseline=False, 
            budget=budget)

        self.frontier = deque()

    def get_next_link(self):
        return self.frontier.pop()

    def is_crawling_terminated(self):
        return not self.frontier

    def add_link(self, link):
        self.frontier.append(link)

    def is_link_in_frontier(self, link):
        return link in self.frontier

    def update_score(self, reward, idx_np=None):
        pass

    def update_score_resource_not_html(self):
        pass

    def modify_action_when_forcing_crawling(self, link):
        pass
    
    def get_current_action(self):
        pass

    def override_current_action(self, action):
        pass


    def save_actions(self):
        pass

    def update_data_structure_offline_learning(self):
        pass

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 " + sys.argv[0] + " <budget:-1 if unlimited> <log_path_all_sites> <site_name>")
    else:
        budget = int(sys.argv[1])
        log_path_all_sites = sys.argv[2]
        site_name = sys.argv[3]

        if budget == -1:
            budget = 1e10

        db_path = os.path.join(os.getcwd(), "data", str(site_name) + ".db")
        table_name = site_name
        log_path = os.path.join(os.getcwd(), "local_crawlers/dfs/logs", str(log_path_all_sites), str(site_name))
       
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        co = sqlite3.connect(os.path.join(os.getcwd(),"data", "site_names_to_urls.db"))
        cu = co.cursor()

        cu.execute("SELECT start_url FROM site_names_to_urls WHERE site_name LIKE \"" + site_name + "\";")
        res = cu.fetchall()[0]

        starting_url = res[0]

        use_url_classifier = False
        classifier_params = {'max_iter_optimizer':100, 'random_state_optimizer':42, 'n_in_ngrams':2, 'batch_size':10}

        dfs_crawler = DFSCrawler(db_path=db_path, 
            table_name=table_name, 
            log_path=log_path, 
            starting_url=starting_url, 
            use_url_classifier=use_url_classifier, 
            classifier_params=classifier_params, 
            budget=budget)
        dfs_crawler.crawl()
