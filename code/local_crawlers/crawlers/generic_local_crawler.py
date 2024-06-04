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

from lxml.etree import ParserError

import random
from itertools import product
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer

sys.setrecursionlimit(10000)

class Link:
    def __init__(self, url, dom_path):
        self.url = url
        self.dom_path = dom_path
        self.associated_np_idx = -1
        self.depth = -1
        self.score = -1 # Only used for offline DOM baseline

    def __str__(self):
        return self.url

    def __eq__(self, other):
        return self.url == other.url

    def __hash__(self):
        return hash(self.url)

    def __lt__(self, other):
        return self.score > other.score

    def change_url(self, new_url):
        self.url = new_url

    def set_associated_np(self, idx):
        self.associated_np_idx = idx

    def get_url(self):
        return self.url

    def get_dom_path(self):
        return self.dom_path

class URL:
    def __init__(self, url, label, mime_type, ground_truth):
        self.url = url
        self.label = label
        self.mime_type = mime_type 
        self.ground_truth = ground_truth

class GenericLocalCrawler(ABC):
    def __init__(self, db_path, table_name, log_path, starting_url, use_url_classifier, classifier_params, is_standard_baseline, is_offline_baseline, budget=10e7):
        self.db_path = db_path
        self.table_name = table_name
        self.starting_url = starting_url
        self.nb_episodes = 0
        self.budget = budget
        self.prologue_pattern = re.compile(r'<\?xml\s.*?\?>')

        self.already_visited = set()
        self.data_resources = set()
        self.not_exploitable_resources = set()
        self.is_standard_baseline = is_standard_baseline
        self.is_offline_baseline = is_offline_baseline

        self.data_volume = [] 
        self.elapsed_times = []
        self.nb_data_resources = []

        self.mime_types_data_resources = {'text/csv', 'application/csv', 'text/x-csv', 'application/x-csv', 'text/x-comma-separated-values', 'text/comma-separated-values', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.oasis.opendocument.spreadsheet', 'application/pdf', 'application/x-pdf', 'application/zip', 'application/x-zip-compressed', 'application/zip-compressed', 'application/x-tar', 'application/x-gtar', 'application/x-gzip', 'application/xml', 'application/json', 'text/json', 'application/yaml', 'text/yaml', 'text/x-yaml', 'application/x-yaml', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.wordprocessingml.template', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'text/plain', 'application/vnd.oasis.opendocument.text', 'application/vnd.ms-excel.sheet.macroenabled.12', 'application/x-7z-compressed', 'application/vnd.oasis.opendocument.presentation', 'application/rdf+xml', 'application/rss+xml', 'application/vnd.ms-excel', 'application/vnd.rar', 'application/x-rar-compressed', 'application/x-gtar'}

        self.connection = sqlite3.connect(db_path)

        self.log_path = os.path.join(log_path, (datetime.now()).strftime("%d-%m-%Y_%H:%M:%S") + "_" + self.table_name)
        os.mkdir(self.log_path) 
        logging.basicConfig(filename=os.path.join(self.log_path, 'crawl.log'), level=logging.INFO)

        self.labeled_urls = []
        
        logging.info("PARAMETERS OF THE CRAWL : ")
        logging.info("-\tdb_path: " + str(db_path))
        logging.info("-\ttable_name:" + str(table_name))
        logging.info("-\tlog_path:" + str(log_path))
        logging.info("-\tstarting_url:" + str(starting_url))
        logging.info("-\tuse_url_classifier:" + str(use_url_classifier))
        logging.info("-\tclassifier_params:" + str(classifier_params))
        logging.info("-\tbudget:" + str(budget))
        logging.info("-\tis_standard_baseline:" + str(is_standard_baseline))
        logging.info("-\tis_offline_baseline:" + str(is_offline_baseline))

        self.use_url_classifier = use_url_classifier
        self.list_of_rewards = []

        self.is_classifier_exploited_yet = False
        self.set_of_url_automatically_classified = {}
        self.classifier_params = classifier_params

        self.is_redirection = False

        if self.use_url_classifier:
            self.model = SGDClassifier(loss = 'log_loss', max_iter = self.classifier_params['max_iter_optimizer'], random_state = self.classifier_params['random_state_optimizer'])
            ascii_chars = [chr(i) for i in range(32, 127)] 
            self.all_n_grams = [''.join(chars) for chars in product(ascii_chars, repeat = self.classifier_params['n_in_ngrams'])]

    def __del__(self):
        self.connection.close()

    def get_type_of_url_head(self, url):
            cursor = self.connection.cursor()

            query_type_str = "SELECT http_response, headers FROM {} WHERE url=?".format(self.table_name)
            query_type = cursor.execute(query_type_str, (url,))
            result_query_type = query_type.fetchall()

            if len(result_query_type) == 0:
                if self.use_url_classifier:
                    self.count_request(0, False)
                logging.info("HTTP Error (40X or 500) : Resource \"" + url + "\" is not in database. Identified at iteration " + str(self.nb_episodes) + ".")
                
                label = 'none_of_them'
                self.not_exploitable_resources.add(url)
                return label
                
            result_query_type = result_query_type[0]

            headers_query_type = result_query_type[1]
            http_response = result_query_type[0]

            headers = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers_query_type)).items()}
            
            if self.use_url_classifier:
                self.count_request(len(headers_query_type.encode()), False)

            if http_response == '200':
                if 'Content-Type' in headers:
                    mime_type = headers['Content-Type'][0].decode('utf-8')
                    if ";" in mime_type:
                        mime_type = mime_type.split(";")[0]
                    if 'html' in mime_type:
                        label = 'html'
                    elif mime_type in self.mime_types_data_resources:
                        label = 'data_resource'
                    else:
                        label = 'none_of_them'
                else:
                    label = 'none_of_them'
                    mime_type = None
            else: 
                label = 'html'
                mime_type = None

            if label != 'none_of_them':
                self.labeled_urls.append(URL(url, label, mime_type, label))

            else:
                self.not_exploitable_resources.add(url)
            
            logging.info("Doing HEAD request over resource \"" + url + "\" at iteration " + str(self.nb_episodes) + ". Got class \"" + str(label) + "\" (Content-Type: \"" + str(mime_type) + "\").")
            return label

    def check_if_classifier_is_correct_when_trusted(self, url, classifier_prediction):
        cursor = self.connection.cursor()

        query_type_str = "SELECT http_response, headers FROM {} WHERE url=?".format(self.table_name)
        query_type = cursor.execute(query_type_str, (url,))
        result_query_type = query_type.fetchall()

        if len(result_query_type) == 0:
            return 'CLASS_FALSE_' + str(classifier_prediction) + "_INSTEAD_OF_none_of_them"

        result_query_type = result_query_type[0]
        headers = result_query_type[1]
        http_response = result_query_type[0]
        
        mime_type = None
        if http_response == '200':
            headers = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers)).items()}
            if 'Content-Type' in headers:
                mime_type = headers['Content-Type'][0].decode('utf-8')
                if ";" in mime_type:
                    mime_type = mime_type.split(";")[0]
                if 'html' in mime_type:
                    label_class = 'html'
                elif mime_type in self.mime_types_data_resources:
                    label_class = 'data_resource'
                else:
                    label_class = 'none_of_them'
            else:
                label_class = 'none_of_them'

        else:
            label_class = 'html'

        if label_class == classifier_prediction:
            return 'CLASS_TRUE'
        else:
            return 'CLASS_FALSE_' + str(classifier_prediction) + "_INSTEAD_OF_" + str(label_class)

    def transform_to_bow(self, data):
        vectorizer = CountVectorizer(vocabulary=self.all_n_grams, analyzer='char', ngram_range=(self.classifier_params['n_in_ngrams'], self.classifier_params['n_in_ngrams']))
        return vectorizer.transform(data)
    
    def get_type_of_url_classifier(self, url):
        bow_url = self.transform_to_bow([url])
        predicted_label = self.model.predict(bow_url)[0]
        correctness = self.check_if_classifier_is_correct_when_trusted(url, predicted_label)
        self.set_of_url_automatically_classified[url] = URL(url, predicted_label, 'unknown', 'unknown')
        logging.info("At iteration " + str(self.nb_episodes) + ", used URL classifier to get label of \"" + url + "\". Predicted class: " + str(predicted_label) + ". Code: " + str(correctness))
        return predicted_label

    def training_epoch_classifier(self):
        X = [url.url for url in self.labeled_urls]
        y = [url.ground_truth for url in self.labeled_urls]

        bow_X = self.transform_to_bow(X)
        self.model.partial_fit(bow_X, y, classes=['html', 'data_resource'])

        self.labeled_urls = []

    def get_type_of_url(self, url):
        if not self.use_url_classifier:
            tmp_result = self.get_type_of_url_head(url)
            return tmp_result
        
        if self.is_classifier_exploited_yet:
            tmp_result = self.get_type_of_url_classifier(url)
            if len(self.labeled_urls) >= self.classifier_params['batch_size']:
                self.training_epoch_classifier()
            return tmp_result


        if len(self.labeled_urls) >= self.classifier_params['batch_size'] and not self.is_classifier_exploited_yet: 
            self.training_epoch_classifier()
            self.is_classifier_exploited_yet = True

        tmp_result = self.get_type_of_url_head(url)
        return tmp_result

    @abstractmethod
    def get_next_link(self):
        pass

    @abstractmethod
    def is_crawling_terminated(self):
        pass

    @abstractmethod
    def add_link(self, link): 
        pass

    @abstractmethod
    def is_link_in_frontier(self, link):
        pass

    @abstractmethod
    def update_score(self, reward, idx_np=None):
        pass

    @abstractmethod
    def update_score_resource_not_html(self):
        pass

    @abstractmethod
    def modify_action_when_forcing_crawling(self, link):
        pass

    @abstractmethod
    def save_actions(self, log_path):
        pass

    @abstractmethod
    def get_current_action(self):
        pass

    @abstractmethod
    def override_current_action(self, new_action):
        pass

    @abstractmethod
    def update_data_structure_offline_learning(self):
        pass

    def save_information(self):
        np.save(os.path.join(self.log_path, "already_visited.npy"), self.already_visited)
        np.save(os.path.join(self.log_path, "data_resources.npy"), self.data_resources)
        np.save(os.path.join(self.log_path, "elapsed_times.npy"), self.elapsed_times)
        np.save(os.path.join(self.log_path, "nb_data_resources.npy"), self.nb_data_resources)
        np.save(os.path.join(self.log_path, "data_volumes.npy"), self.data_volume)
        np.save(os.path.join(self.log_path, "reward_distribution.npy"), self.list_of_rewards)

        if not self.is_standard_baseline:
            self.save_actions(self.log_path) 

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
        tree = html.fromstring(cleaned_html_content)
        base_element = tree.find(".//base")
        
        base_url = base_element.get("href") if base_element is not None else None

        xpath_expressions = [
                        '//a/@href',         # Select href attribute from <a> tags
                        '//area/@href',      # Select href attribute from <area> tags
                        '//frame/@src',      # Select src attribute from <frame> tags
                        '//iframe/@src',     # Select src attribute from <iframe> tags
        ]

        for xpath_expression in xpath_expressions:
            elements = tree.xpath(xpath_expression)
            for element in elements:
                url = element.strip()
                dom_path = self.get_dom_path(element.getparent())
                results.append({'url':url, 'dom_path':dom_path})
        
        return results, base_url

    def count_request(self, content_length, is_data_resource):
        if self.is_offline_baseline:
            self.update_data_structure_offline_learning()

        self.nb_episodes += 1
        self.nb_data_resources.append(len(self.data_resources))
         
        if is_data_resource:
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
        self.nb_data_resources.append(0)

        self.max_depth = 0
        self.is_data_resource = False

        self.beginning = perf_counter()
        start_link = Link(self.starting_url, "")
        start_link.depth = 0
        self.crawl_next_resource(start_link)

        while not self.is_crawling_terminated() and self.nb_episodes < self.budget:
            link = self.get_next_link()
            self.crawl_next_resource(link)

        self.elapsed_times.append(self.elapsed_times[-1] + perf_counter() - self.beginning)
        self.nb_data_resources.append(len(self.data_resources))

        self.save_information()

    def crawl_next_resource(self, link):
        self.already_visited.add(link)
        if link.depth > self.max_depth:
            self.max_depth = link.depth

        logging.info("\"" + link.get_url() + "\" crawled from database. Number of episodes : " + str(self.nb_episodes + 1) + ".")

        cursor = self.connection.cursor()
        query_str = "SELECT http_response, headers, body, content_length FROM {} WHERE url=?".format(self.table_name)
        query = cursor.execute(query_str, (link.get_url(),))
        result = query.fetchall()

        if len(result) == 0:
            self.count_request(0, False)
            logging.info("HTTP Error (4XX or 5XX) : Resource \"" + link.get_url() + "\" is not in database. Identified at episode " + str(self.nb_episodes) + ".")
            self.not_exploitable_resources.add(link.get_url())
            if not self.is_data_resource:
                self.update_score_resource_not_html() 
            return

        result = result[0]
        http_response = result[0]
        headers = result[1]
        body = result[2]
        content_length = result[3]

        associated_predicted_url = None
        if link.get_url() in self.set_of_url_automatically_classified:
            associated_predicted_url = self.set_of_url_automatically_classified[link.get_url()]

        if http_response == "200":
            headers = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers)).items()}
            if 'Content-Type' in headers: 
                mime_type = headers['Content-Type'][0].decode('utf-8')
                if ";" in mime_type:
                    mime_type = mime_type.split(";")[0]
                if 'html' in mime_type: 
                    if associated_predicted_url is not None:
                        associated_predicted_url.mime_type = mime_type
                        associated_predicted_url.ground_truth = 'html'
                        self.labeled_urls.append(associated_predicted_url)
                        del self.set_of_url_automatically_classified[link.get_url()]
                    if self.is_redirection:
                         self.modify_action_when_forcing_crawling(link)
                    if len(body) > 0 and not body.isspace():
                        new_elements, base_url = self.extract_urls_from_html(body)
                    else:
                        new_elements = []
                    self.count_request(content_length, False)
                else:
                    is_data_resource = mime_type in self.mime_types_data_resources
                    if is_data_resource: 
                        if associated_predicted_url is not None:
                            associated_predicted_url.mime_type = mime_type
                            associated_predicted_url.ground_truth = 'data_resource'
                            self.labeled_urls.append(associated_predicted_url)
                            del self.set_of_url_automatically_classified[link.get_url()]
                        self.data_resources.add(URL(link.url, 'data_resource', mime_type, 'date_resource'))
                    else:
                        self.not_exploitable_resources.add(link.get_url())

                    if not self.is_data_resource:
                        self.update_score_resource_not_html()
                    self.count_request(content_length, is_data_resource)

                    return
            else:
                self.not_exploitable_resources.add(link.get_url())
                if not self.is_data_resource:
                    self.update_score_resource_not_html()
                    self.count_request(content_length, False)
                return

        elif http_response[:2] == '30':
            self.count_request(content_length, False)
            headers = {key.decode('utf-8'):value for key, value in (ast.literal_eval(headers)).items()}
            location = headers['Location'][0].decode('utf-8')
            try:
                full_new_redirected_url = urldefrag(urljoin(link.get_url(), str(location))).url
            except:
                logging.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" in a redirection. It probably contains unhandled characters under NKFC normalization.")
                return

            redirection_link = Link(full_new_redirected_url, link.get_dom_path())

            logging.info("Status " + http_response + " encountered for URL " + link.get_url() + ", which was redirected to " + full_new_redirected_url + "!")
            if redirection_link not in self.already_visited and not self.is_link_in_frontier(redirection_link) and redirection_link not in self.data_resources and redirection_link.get_url() not in self.not_exploitable_resources: 
                if link.get_url() == self.starting_url:
                    self.add_link(redirection_link)
                    return
                else:
                    self.crawl_next_resource(redirection_link)
                    return

            else:
                self.update_score_resource_not_html()
                return

        else:
            print("Problem of status unhandled.")

        reward = 0
        for new_element in new_elements:
           
            if base_url is None:
                try:
                    full_new_url = urldefrag(urljoin(link.get_url(), new_element['url'])).url
                except:
                    logging.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" . It probably contains unhandled characters under NKFC normalization.")
                    continue     
            else:
                try:
                    full_new_url = urldefrag(urljoin(base_url, new_element['url'])).url
                except:
                    logging.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" . It probably contains unhandled characters under NKFC normalization.")
                    continue

            if not self.is_url_on_same_or_sub_domain(self.starting_url, full_new_url):
                continue

            new_link = Link(full_new_url, new_element['dom_path'])

            if new_link not in self.already_visited and not self.is_link_in_frontier(new_link) and new_link not in self.data_resources and new_link.get_url() not in self.not_exploitable_resources:
                new_link.depth = link.depth + 1
                
                if not self.is_standard_baseline and not self.is_offline_baseline:
                    url_type = self.get_type_of_url(new_link.get_url())

                    if url_type == 'html':
                        logging.info("URL " + new_link.get_url() + " was added to frontier, from URL " + link.get_url() + ".")
                        self.add_link(new_link)

                    elif url_type == 'data_resource':
                        prediction_code = self.check_if_classifier_is_correct_when_trusted(new_link.get_url(), 'data_resource')
                        if "CLASS_FALSE_data_resource_INSTEAD_OF_html" in prediction_code:
                            logging.info("Misclassified the HTML resource \"" + str(new_link.get_url()) + "\"(was classified as data resource). We thus crawl it now.")
                            current_action = self.get_current_action()
                            self.modify_action_when_forcing_crawling(new_link)
                            self.crawl_next_resource(new_link)
                            self.override_current_action(current_action)
                        else:
                            self.is_data_resource = True
                            self.crawl_next_resource(new_link)
                            self.is_data_resource = False

                            if prediction_code == "CLASS_TRUE":
                                logging.info("Data resource identified at \"" + str(new_link.get_url()) + "\". Crawled at iteration " + str(self.nb_episodes) + ".")
                                reward += 1
                    else:
                        self.not_exploitable_resources.add(new_link.get_url())
                else:
                    self.add_link(new_link)
        
        if link.get_url() != self.starting_url and not self.is_offline_baseline and not self.is_standard_baseline:                  
            self.update_score(reward)


        if reward > 0 and not self.is_standard_baseline and not self.is_offline_baseline:
            self.list_of_rewards.append((link.url, reward))
