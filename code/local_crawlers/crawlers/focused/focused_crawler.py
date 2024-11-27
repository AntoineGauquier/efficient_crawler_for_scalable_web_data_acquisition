import sys
import sqlite3
from abc import ABC, abstractmethod
import os
from datetime import datetime
import logging
import ast
from urllib.parse import urljoin, urldefrag, quote, unquote, urlparse, urlunparse
from time import perf_counter
from lxml import html
import numpy as np
import re
import copy

from lxml.etree import ParserError

import random
from itertools import product
from sklearn.linear_model import SGDClassifier

from collections import deque
import heapq

from sklearn.feature_extraction.text import CountVectorizer

sys.setrecursionlimit(10000)

class Link:
    def __init__(self, url, dom_path):
        self.url = url
        self.dom_path = dom_path
        self.associated_np_idx = -1
        self.depth = -1
        self.priority = -1 # The priority of a link, given by the link classifier. 1 means maximal probability of being a target, 0 means maximal probability of not being. 

    def __str__(self):
        return self.url

    def __eq__(self, other):
        return self.url == other.url

    def __hash__(self):
        return hash(self.url)

    def __lt__(self, other):
        return self.priority > other.priority # Because heapq orders by increasing value, and we want decreasing order

    def change_url(self, new_url):
        self.url = new_url

    def set_associated_np(self, idx):
        self.associated_np_idx = idx

    def get_url(self):
        return self.url

    def get_dom_path(self):
        return self.dom_path

    def set_ground_truth(self, ground_truth):
        self.ground_truth = ground_truth

class URL:
    def __init__(self, url, label, mime_type, ground_truth):
        self.url = url
        self.label = label
        self.mime_type = mime_type 
        self.ground_truth = ground_truth

class GenericLocalCrawlerAdapted(ABC):
    def __init__(self, db_path, table_name, log_path, starting_url,  budget=10e7):
        self.db_path = db_path
        self.table_name = table_name
        self.starting_url = starting_url
        self.nb_episodes = 0
        self.budget = budget
        self.prologue_pattern = re.compile(r'<\?xml\s.*?\?>')

        self.already_visited = set()
        self.resources_of_interest = set()
        self.not_exploitable_resources = set()

        self.data_volume = [] # A dictionnary of tuples (x, y) where x is the total volume of crawled data so far and y the total volum of crawled resources of interest so far
        self.elapsed_times = []
        self.nb_resources_of_interest = []

        self.mime_types_of_interest = {'application/octet-stream', 'application/pdf', 'text/csv', 'application/csv', 'text/x-csv', 'application/x-csv', 'text/x-comma-separated-values', 'text/comma-separated-values', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.oasis.opendocument.spreadsheet', 'application/pdf', 'application/x-pdf', 'application/zip', 'application/x-zip-compressed', 'application/zip-compressed', 'application/x-tar', 'application/x-gtar', 'application/x-gzip', 'application/xml', 'application/json', 'text/json', 'application/yaml', 'text/yaml', 'text/x-yaml', 'application/x-yaml', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.wordprocessingml.template', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'text/plain', 'application/vnd.oasis.opendocument.text', 'application/vnd.ms-excel.sheet.macroenabled.12', 'application/x-7z-compressed', 'application/vnd.oasis.opendocument.presentation', 'application/rdf+xml', 'application/rss+xml', 'application/vnd.ms-excel', 'application/vnd.rar', 'application/x-rar-compressed', 'application/x-gtar'}

        self.connection = sqlite3.connect(db_path)
        self.log_path = log_path + (datetime.now()).strftime("%d-%m-%Y_%H:%M:%S") + "_" + self.table_name + "/"
        os.mkdir(self.log_path) 
        logging.basicConfig(filename=self.log_path + 'crawl.log', level=logging.INFO)

        
        logging.info("PARAMETERS OF THE MODEL : ")
        logging.info("-\tdb_path: " + str(db_path))
        logging.info("-\ttable_name:" + str(table_name))
        logging.info("-\tlog_path:" + str(log_path))
        logging.info("-\tstarting_url:" + str(starting_url))
        logging.info("-\tbudget:" + str(budget))

        self.is_redirection = False

        self.model = SGDClassifier(loss = 'log_loss', max_iter = 100, random_state = 42)
        ascii_chars = [chr(i) for i in range(32, 127)] 
        self.all_n_grams = [''.join(chars) for chars in product(ascii_chars, repeat = 2)]

        self.vectorizer = CountVectorizer(vocabulary=self.all_n_grams, analyzer='char', ngram_range=(2, 2))

    def __del__(self):
        self.connection.close()

    def train_link_classifier(self):
        train_data = [[l.classifier_features[0]] + l.classifier_features[1].toarray().flatten().tolist() + l.classifier_features[2].toarray().flatten().tolist() for l in self.labeled_links]
        train_data = np.array(train_data, dtype=np.float32)

        ground_truths = [l.ground_truth for l in self.labeled_links]

        self.model.partial_fit(train_data, ground_truths, classes=['target', 'non_target'])

        self.labeled_links = []


    def get_classification_probabilities(self, link):
        data_to_predict = [link.classifier_features[0]] + link.classifier_features[1].toarray().flatten().tolist() + link.classifier_features[2].toarray().flatten().tolist()

        data_to_predict = np.array(data_to_predict, dtype=np.float32).reshape(1, -1)
        link.priority = self.model.predict_proba(data_to_predict)[0][1]

    def update_data_structure(self):
        self.train_link_classifier()

        new_data_structure = []

        for link in self.frontier:
            self.get_classification_probabilities(link)
            heapq.heappush(new_data_structure, link)

        self.frontier = new_data_structure

    def add_labeled_data(self, link, ground_truth):
        link.set_ground_truth(ground_truth)
        self.labeled_links.append(link)

    def set_classifier_features(self, link):
        features = [link.depth]

        url_2_grams = self.vectorizer.transform([link.url])
        features.append(url_2_grams)

        anchor_2_grams = self.vectorizer.transform([link.anchor])
        features.append(anchor_2_grams)

        link.classifier_features = features


    @abstractmethod
    def get_next_url(self):
        pass

    @abstractmethod
    def is_crawling_terminated(self):
        pass

    @abstractmethod
    def add_url(self, url): 
        pass

    @abstractmethod
    def is_url_in_frontier(self, url):
        pass

    def save_information(self):
        np.save(self.log_path + "already_visited.npy", self.already_visited)
        np.save(self.log_path + "resources_of_interest.npy", self.resources_of_interest)
        np.save(self.log_path + "elapsed_times.npy", self.elapsed_times)
        np.save(self.log_path + "nb_resources_of_interest.npy", self.nb_resources_of_interest)
        np.save(self.log_path + "data_volume.npy", self.data_volume)
 
        logging.info("Information about the crawling saved succesfully in " + self.log_path)

    def get_dom_path(self, element):
        path = []
        while element is not None:
            path.insert(0, element.tag)
            if 'class' in element.attrib:
                path[0] += '.' + ' '.join(element.attrib['class'].split())
            if 'id' in element.attrib:
                path[0] += '#' + element.attrib['id']
            element = element.getparent()
        return '/' + '/'.join(path)

    def extract_urls_from_html(self, html_content):
        results = []
        cleaned_html_content = re.sub(self.prologue_pattern, '', html_content)
        try:
            tree = html.fromstring(cleaned_html_content)
            base_element = tree.find(".//base")
        
            base_url = base_element.get("href") if base_element is not None else None

            xpath_expressions = [
                        '//a/@href',         # Select href attribute from <a> tags
                        '//area/@href',   # Select href attribute from <area> tags
                        '//frame/@src',   # Select src attribute from <frame> tags
                        '//iframe/@src',     # Select src attribute from <iframe> tags
            ]

            for xpath_expression in xpath_expressions:
                elements = tree.xpath(xpath_expression)
                for element in elements:
                    url = element.strip()
                    dom_path = ''
                    anchor_text = element.getparent().text_content().strip() if element.getparent() is not None else ''
                    results.append({'url':url, 'dom_path':dom_path, 'anchor_text':anchor_text})
        
            return results, base_url
        except:
            logging.info("Error in parsing HTML document. No URLs extracted.")
            return results, None

    def count_request(self, content_length, is_resource_of_interest):
        self.nb_episodes += 1
        self.nb_resources_of_interest.append(len(self.resources_of_interest))
         
        if is_resource_of_interest:
            self.data_volume.append((self.data_volume[-1][0], self.data_volume[-1][1] + content_length))
        else:
            self.data_volume.append((self.data_volume[-1][0] + content_length, self.data_volume[-1][1]))
        self.elapsed_times.append(self.elapsed_times[-1] + perf_counter() - self.beginning)
        self.beginning = perf_counter()

    def is_url_on_same_or_sub_domain(self, starting_url, url_to_check):
        if urlparse(url_to_check).hostname == None:
            return False

        starting_domain_parts = urlparse(starting_url).hostname.split('.')
        starting_domain_parts.reverse()
        if starting_domain_parts[-1] == "www":
            starting_domain_parts = starting_domain_parts[:-1]

        url_to_check_domain_parts = urlparse(url_to_check).hostname.split('.')
        url_to_check_domain_parts.reverse()
        if url_to_check_domain_parts[-1] == "www":
            url_to_check_domain_parts = url_to_check_domain_parts[:-1]

        if len(starting_domain_parts) > len(url_to_check_domain_parts):
            return False

        return starting_domain_parts == url_to_check_domain_parts[:len(starting_domain_parts)]


    def crawl(self):
        logging.info("Starting crawling on database \"" + self.db_path + "\", on table \"" + self.table_name + "\", with starting URL \"" + self.starting_url + ".")
        
        self.data_volume.append((0, 0))
        self.elapsed_times.append(0)
        self.nb_resources_of_interest.append(0)

        self.max_depth = 0

        self.beginning = perf_counter()
        start_link = Link(self.starting_url, "")
        start_link.depth = 0
        #start_link.path = "root"
        self.crawl_next_resource(start_link)

        while not self.is_crawling_terminated() and self.nb_episodes < self.budget:
            link = self.get_next_url()
            self.crawl_next_resource(link)

        self.elapsed_times.append(self.elapsed_times[-1] + perf_counter() - self.beginning)
        self.nb_resources_of_interest.append(len(self.resources_of_interest))

        self.save_information()
    
    def crawl_next_resource(self, link):
        self.already_visited.add(link)
        if link.depth > self.max_depth:
            self.max_depth = link.depth

        logging.info("\"" + link.get_url() + "\" crawled from database. Number of episodes : " + str(self.nb_episodes + 1) + ".") # +1 because self.nb_episodes not updated yet

        if self.nb_episodes > 100 and len(self.labeled_links) == 10:
            self.train_link_classifier()
        elif self.nb_episodes == 100:
            self.update_data_structure()

        cursor = self.connection.cursor()
        query_str = "SELECT http_response, headers, body, content_length FROM {} WHERE url=?".format(self.table_name)
        query = cursor.execute(query_str, (link.get_url(),))
        result = query.fetchall()

        if len(result) == 0:
            self.count_request(0, False)
            logging.info("HTTP Error (40X or 500) : Resource \"" + link.get_url() + "\" is not in database. Identified at episode " + str(self.nb_episodes) + ".")
            self.not_exploitable_resources.add(link.get_url())
            return

        result = result[0]
        http_response = result[0]
        headers = result[1]
                                    
        headers_dict = {key:value for key, value in (ast.literal_eval(headers)).items()}
        if type([hk for hk in headers_dict.keys()][0]) == bytes:
            headers_dict = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers)).items()}

        headers = headers_dict
        body = result[2]
        content_length = result[3]


        if http_response[0] in ['4', '5'] or (http_response[0] == '3' and headers.get("Location", None) is None): # Scenario where the resource is not in DB (equivalent to a status 40X or 500 in an "on-line" fashion)
            self.count_request(0, False)
            logging.info("HTTP Error (40X or 500) : Resource \"" + link.get_url() + "\" is not in database. Identified at episode " + str(self.nb_episodes) + ".")
            #print("Resource not available.")
            self.not_exploitable_resources.add(link.get_url())
            return

        if http_response[0] == "2":
            #print([key for key, _ in (ast.literal_eval(headers)).items()])
            #headers = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers)).items()}
            if 'Content-Type' in headers: 
                mime_type = headers['Content-Type'][0].decode('utf-8')
                if ";" in mime_type:
                    mime_type = mime_type.split(";")[0]
                if 'html' in mime_type: # We are crawling an HTML page.
                    if len(body) > 0 and not body.isspace():
                        #print(len(body))
                        try:
                             new_elements, base_url = self.extract_urls_from_html(body)
                        except ParserError as e:
                             logger.info("Error parsing HTML content of \"" + link.get_url() + "\": " + str(e))
                             new_elements = []
                    else:
                        new_elements = []
                    if self.nb_episodes > 1:
                        self.add_labeled_data(link, 'non_target')
                    self.count_request(content_length, False)
                else:# Scenario in which we predicted html but it was either interest or none_of_them, or for baslines where we do not use HEAD requests
                    is_of_interest = mime_type in self.mime_types_of_interest # If true, we are crawling a resource of interest : only observed when we use an URL classifier with a RL agent, an corresponds to a misclassifiction (interest instead of html).
                    if is_of_interest:
                        if self.nb_episodes > 1:
                            self.add_labeled_data(link, 'target')
                        self.resources_of_interest.add(link)
                    else:
                        if self.nb_episodes > 1:
                            self.add_labeled_data(link, 'non_target')
                        self.not_exploitable_resources.add(link.get_url())
                    self.count_request(content_length, is_of_interest)

                    return
            else: # To filter responses which head does not come up with a content type. We cannot do anything in that case. Observed on https://fonction-publique.gouv.fr/
                if self.nb_episodes > 1:
                    self.add_labeled_data(link, 'non_target')
                self.not_exploitable_resources.add(link.get_url()) # We keep track of not exploitable resources (40X/500, neither of interest nor html, no content-type) so that we do not crawl them twice. 
                self.count_request(content_length, False)
                return

        elif http_response[:2] == '30': # Redirection (30X)
            self.count_request(content_length, False)
            try:
                location = headers['Location'][0].decode('utf-8') # We use the Location to know where we should look for the page.
            except:
                print("Error:" + str(headers))
                location = "#"

            try:
                full_new_redirected_url = urldefrag(urljoin(link.get_url(), str(location))).url
            except:
                logging.info("Exception occured while trying to parse the URL \"" + str(link.get_url()) + "\" in a redirection. It probably contains unhandled characters under NKFC normalization. Probably due to an error in the HTML code itself.")
                return

            redirection_link = copy.copy(link)
            redirection_link.url = full_new_redirected_url

            logging.info("Status " + http_response + " encountered for URL " + link.get_url() + ", which was redirected to " + full_new_redirected_url + "!")
            if redirection_link not in self.already_visited and not self.is_url_in_frontier(redirection_link) and redirection_link not in self.resources_of_interest and redirection_link.get_url() not in self.not_exploitable_resources: # To cover the scenario where the location is an URL that is already seen. Added because of https://www.fonction-publique.gouv.fr/coronavirus-covid-19, which redirects to itself. 
                if link.get_url() == self.starting_url: # To cover the case where we have a redirection at starting URL.
                    self.add_url(redirection_link)
                    return
                else:
                    self.crawl_next_resource(redirection_link)
                    return

            else:
                if self.nb_episodes > 1:
                    self.add_labeled_data(link, 'non_target')
                return

        else:
            print("Problem of status unhandled.")

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
            if new_link not in self.already_visited and not self.is_url_in_frontier(new_link) and new_link not in self.resources_of_interest and new_link.get_url() not in self.not_exploitable_resources:
                new_link.depth = link.depth + 1
                new_link.anchor = new_element['anchor_text']
                self.set_classifier_features(new_link)
                self.add_url(new_link)

class FocusedCrawler(GenericLocalCrawlerAdapted):
    def __init__(self, db_path, table_name, log_path, starting_url, budget):
            super().__init__(db_path, table_name, log_path, starting_url, budget)
            self.frontier = deque()
            self.labeled_links = []

    def add_url(self, link):
        if self.nb_episodes <= 100:
            self.frontier.append(link)
        else:
            self.get_classification_probabilities(link)
            heapq.heappush(self.frontier, link)

    def get_next_url(self):
        chosen_link = None
        if self.nb_episodes <= 100:
            chosen_link = self.frontier.popleft()
        else:
            chosen_link = heapq.heappop(self.frontier)

        return chosen_link

    def is_crawling_terminated(self):
        return len(self.frontier) == 0

    def is_url_in_frontier(self, link):
        return link in self.frontier


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python3 " + str(sys.argv[0]) + " <site_name> <log_path_all_sites> <budget>")

    site_name = sys.argv[1]
    log_path_all_sites = sys.argv[2]
    budget = int(sys.argv[3])

    if budget == -1:
        budget = 1000000000

    db_path = os.path.join(os.getcwd(), "data", str(site_name) + ".db")
    table_name = site_name

    log_path = os.path.join(os.getcwd(), "crawlers/focused/logs", str(log_path_all_sites), str(site_name) + "/")
   
    if not os.path.exists(os.path.join(os.getcwd(), "crawlers/focused/logs")):
        os.mkdir(os.path.join(os.getcwd(), "crawlers/focused/logs"))
    if not os.path.exists(os.path.join(os.getcwd(), "crawlers/focused/logs", str(log_path_all_sites))):
        os.mkdir(os.path.join(os.getcwd(), "crawlers/focused/logs", str(log_path_all_sites)))
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    co = sqlite3.connect(os.path.join(os.getcwd(),"data", "site_names_to_urls.db"))
    cu = co.cursor()

    cu.execute("SELECT start_url FROM site_names_to_urls WHERE site_name LIKE \"" + site_name + "\";")
    res = cu.fetchall()[0]

    starting_url = res[0]

    crawler = FocusedCrawler(db_path, table_name, log_path, starting_url, budget)
    crawler.crawl()