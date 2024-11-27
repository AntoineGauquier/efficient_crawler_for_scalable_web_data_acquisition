import sys
import os
#sys.path.append(os.path.join(os.getcwd(),'/home/gauquier/efficent_crawler_for_scalable_web_data_acquisition/auer_crawler/online_auer_crawler/online_auer_crawler/spiders'))
import copy

import math
import numpy as np
import multiprocessing
import sqlite3
import requests

import scrapy
from lxml import html
import logging
from datetime import datetime
import re
from collections import Counter, deque

from scrapy.exceptions import CloseSpider

import ast
from urllib.parse import urljoin, urldefrag, quote, unquote, urlparse, urlunparse
from time import perf_counter

from lxml.etree import ParserError

from itertools import product
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
import heapq

sys.setrecursionlimit(10000)

class Link:
    def __init__(self, url, dom_path, supposed_to_be_target = False, supposed_to_be_html = False):
        self.url = url
        self.dom_path = dom_path
        self.associated_np_idx = -1
        self.depth = -1
        self.priority = -1 # The priority of a link, given by the link classifier. 1 means maximal probability of being a target, 0 means maximal probability of not being. 
        self.supposed_to_be_target = supposed_to_be_target
        self.supposed_to_be_html = supposed_to_be_html

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

class FocusedOnlineCrawler(scrapy.Spider):
    name = "focused_online_crawler"
    handle_httpstatus_list = [i for i in range(300, 600)]
    
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):

        spider = super(FocusedOnlineCrawler, cls).from_crawler(crawler, *args, **kwargs)
 
        crawler.settings.set('LOG_FILE', os.path.join("crawlers", "focused_online_crawler", "logs", "logs_" + kwargs.get('site_name', None) + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"))

        if not os.path.exists(os.path.join("crawlers", "focused_online_crawler", "logs")):
            os.mkdir(os.path.join("crawlers", "focused_online_crawler", "logs"))
        if not os.path.exists(os.path.join("crawlers", "focused_online_crawler", "output")):
            os.mkdir(os.path.join("crawlers", "focused_online_crawler", "output"))

        crawler.settings.set('LOG_ENABLED', True)
        crawler.settings.set('COOKIES_ENABLED', False)
        crawler.settings.set('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        crawler.settings.set('LOG_LEVEL', 'INFO')
        crawler.settings.set('DEPTH_LIMIT', 100000000)
        crawler.settings.set('CONCURRENT_REQUESTS', 16)
        crawler.settings.set('DOWNLOAD_DELAY', 1)
        if float(kwargs.get('budget', 100000000)) == -1:
            budget = 1000000000
        else:
            budget = float(kwargs.get('budget', 100000000))
        crawler.settings.set('CLOSESPIDER_PAGECOUNT', budget)
        #crawler.settings.set('COMPRESSION_ENABLED', False) # Sometimes required, depending on websites

        return spider

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.starting_url = kwargs.get('starting_url', None)
        self.name_of_table = kwargs.get('site_name', None)  
        self.starting_datetime = (datetime.now()).strftime("%d-%m-%Y_%H:%M:%S")
        self.output_path = os.path.join(kwargs.get('output_path', None), self.starting_datetime + "_" + self.name_of_table)
        os.mkdir(self.output_path)
        self.common_db_path = kwargs.get('path_to_common_db', None)

        if float(kwargs.get('budget', 100000000)) == -1:
            self.budget = 1000000000
        else:
            self.budget = float(kwargs.get('budget', 100000000))   

        self.start_urls = [self.starting_url]
        
        self.max_len_from_root = 350 # Maximum depth, to limit robot traps
        self.nb_urls_rejected_by_len = 0
        
        self.prologue_pattern = re.compile(r'<\?xml\s.*?\?>')

        self.nb_episodes = 0
        self.batch_size_save = 100

        self.yielded_to_scrapy = set()
        self.already_visited = set()
        self.data_resources = set()
        self.not_exploitable_resources = set()
        self.is_offline_baseline = False

        self.data_volumes = [] 
        self.elapsed_times = []
        self.nb_data_resources = []

        self.labeled_links = []

        self.current_number_of_requests_in_scrapy_queue = 0

        self.mime_types_data_resources = {'application/octet-stream', 'text/csv', 'application/csv', 'text/x-csv', 'application/x-csv', 'text/x-comma-separated-values', 'text/comma-separated-values', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.oasis.opendocument.spreadsheet', 'application/pdf', 'application/x-pdf', 'application/zip', 'application/x-zip-compressed', 'application/zip-compressed', 'application/x-tar', 'application/x-gtar', 'application/x-gzip', 'application/xml', 'application/json', 'text/json', 'application/yaml', 'text/yaml', 'text/x-yaml', 'application/x-yaml', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.wordprocessingml.template', 'application/vnd.openxmlformats-officedocument.presentationml.presentation', 'text/plain', 'application/vnd.oasis.opendocument.text', 'application/vnd.ms-excel.sheet.macroenabled.12', 'application/x-7z-compressed', 'application/vnd.oasis.opendocument.presentation', 'application/rdf+xml', 'application/rss+xml', 'application/vnd.ms-excel', 'application/vnd.rar', 'application/x-rar-compressed', 'application/x-gtar'}

        self.connection_common_db = sqlite3.connect(os.path.join(self.common_db_path, self.name_of_table + ".db"), timeout=180)
        self.create_table_common_db()

        logging.info("PARAMETERS OF THE CRAWL : ")
        
        logging.info("-\toutput_path:" + str(self.output_path))
        logging.info("-\tstarting_url:" + str(self.starting_url))
        logging.info("-\tbudget:" + str(self.budget))

        self.model = SGDClassifier(loss = 'log_loss', max_iter = 100, random_state = 42)
        ascii_chars = [chr(i) for i in range(32, 127)] 
        self.all_n_grams = [''.join(chars) for chars in product(ascii_chars, repeat = 2)]

        self.vectorizer = CountVectorizer(vocabulary=self.all_n_grams, analyzer='char', ngram_range=(2, 2))

        self.frontier = deque()
        self.frontier_length = 0

        logging.info("Starting crawling from URL \"" + self.starting_url + "\".")
        
        self.data_volumes.append((0, 0))
        self.elapsed_times.append(0)
        self.nb_data_resources.append(0)

        self.max_depth = 0
        self.is_data_resource = False

        self.beginning = perf_counter()
        self.start_link = Link(self.starting_url, "")
        self.start_link.depth = 0
        
    def start_requests(self):
        self.current_number_of_requests_in_scrapy_queue += 1
        self.yielded_to_scrapy.add(self.start_link)
        yield scrapy.Request(self.start_link.get_url(), callback=self.parse, dont_filter=True, meta={'dont_redirect':True, 'link':self.start_link})

    def is_crawling_terminated(self):
        return len(self.frontier) == 0

    def is_link_in_frontier(self, link):
        return link in self.frontier

    def save_information(self):
        np.save(os.path.join(self.output_path, "already_visited.npy"), self.already_visited)
        np.save(os.path.join(self.output_path, "data_resources.npy"), self.data_resources)
        np.save(os.path.join(self.output_path, "elapsed_times.npy"), self.elapsed_times)
        np.save(os.path.join(self.output_path, "nb_data_resources.npy"), self.nb_data_resources)
        np.save(os.path.join(self.output_path, "data_volumes.npy"), self.data_volumes)

        logging.info("Information about the crawling saved succesfully in " + self.output_path)

    def create_table_common_db(self):
        cur = self.connection_common_db.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS " + self.name_of_table + " (url TEXT PRIMARY KEY, http_response TEXT, headers TEXT, body TEXT, content_length INTEGER);")
        self.connection_common_db.commit()
        
    def insert_new_data_common_db(self, url, http_response, headers, body, content_length):
        cur = self.connection_common_db.cursor()
        while True:
            try:
                cur.execute("INSERT INTO " + self.name_of_table + " (url, http_response, headers, body, content_length) VALUES (?, ?, ?, ?, ?);", (url, http_response, headers, body, content_length))
                self.connection_common_db.commit()
                break
            except sqlite3.IntegrityError:
                break # Case where we try to insert a data that is already in the table, just in case the webpage was added by another crawler between the crawl and this point
            except Exception as e:
                pass

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

    def add_link(self, link):
        if self.nb_episodes <= 10:
            self.frontier.append(link)
        else:
            self.get_classification_probabilities(link)
            heapq.heappush(self.frontier, link)

    def get_next_link(self):
        chosen_link = None
        if self.nb_episodes <= 10:
            chosen_link = self.frontier.popleft()
        else:
            chosen_link = heapq.heappop(self.frontier)

        return chosen_link

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
            self.logger.info("Error in parsing HTML document. No URLs extracted.")
            return results, None

    def count_request(self, content_length, is_data_resource):
        if self.is_offline_baseline:
            self.update_data_structure_offline_learning()

        self.nb_episodes += 1
        self.nb_data_resources.append(len(self.data_resources))
         
        if is_data_resource:
            self.data_volumes.append((self.data_volumes[-1][0], self.data_volumes[-1][1] + content_length))
        else:
            self.data_volumes.append((self.data_volumes[-1][0] + content_length, self.data_volumes[-1][1]))
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

    def too_many_repeated_fragments(self, url, k): # Is used to avoid robot traps: it is highly unlikely that a fragment of an URL is repeated several times (especially, more than 2)
        last_url = url
        
        while unquote(last_url) != last_url:
            last_url = unquote(last_url)

        parsed_url = urlparse(last_url)

        elem_path = parsed_url.path.split("/")
        elem_query = parsed_url.query.split("&")
        elem_query_attr_res = []
        for elem in elem_query:
            if "=" in elem:
                elem_query_attr_res.extend(elem.split("="))
            else:
                elem_query_attr_res.extend(elem)
        
        atoms = elem_path + elem_query_attr_res
        atom_counts = Counter(atoms)

        for count in atom_counts.values():
            if count >= k:
                return True
        return False

    def update_scrapy_queue_if_needed(self, is_middleware=False):
        if self.current_number_of_requests_in_scrapy_queue == 0 or is_middleware:
            link = self.get_next_link()
            return scrapy.Request(link.get_url(), callback=self.parse, dont_filter=True, meta={'dont_redirect':True, 'link': link})
        else:
            return None

    def parse(self, response):
        self.current_number_of_requests_in_scrapy_queue -= 1
        assert(self.current_number_of_requests_in_scrapy_queue >= 0)

        if (self.is_crawling_terminated() or self.nb_episodes + 1 >= self.budget - 1) and self.nb_episodes > 1:
            self.elapsed_times.append(self.elapsed_times[-1] + perf_counter() - self.beginning)
            self.nb_data_resources.append(len(self.data_resources))

            self.save_information()
            raise CloseSpider(reason='Crawling terminated: ending condition met.')

        elif self.nb_episodes % self.batch_size_save == 0 and self.nb_episodes > 1:
                self.save_information()
                self.logger.info("Crawling data saved at iteration " + str(self.nb_episodes) + ".")



        if self.nb_episodes > 10 and len(self.labeled_links) == 10:
            self.train_link_classifier()
        elif self.nb_episodes == 10:
            self.update_data_structure()

        link = response.meta['link']
        http_response = str(response.status)

        headers = {key.decode('utf-8'): value for key, value in dict(response.headers).items()}

        try:
            location = headers['Location'][0].decode('utf-8')
        except:
            location = None

        self.already_visited.add(link)
       
        if link.depth > self.max_depth:
            self.max_depth = link.depth
        
        body = "Either not an HTML resource, not a status 2XX or no CT in headers."

        if not response.meta.get('from_common_db'):
            content_type = response.headers.get(b'Content-Type', b'').decode('utf-8').lower()
            if 'html' in content_type and http_response[0] == '2':
                try:
                    body = response.text
                except:
                    body = ''

            content_length = int(response.headers.get(b'Content-Length', -1))        

            if content_length == -1:
                if isinstance(response.body, str):
                    content_length = len((response.body).encode())
                else:
                    content_length = len(response.body)
            self.insert_new_data_common_db(response.url, response.status, str(dict(response.headers)), body, content_length)
            self.logger.info("Webpage at URL \"" + response.url + "\" was added to common_db.")

        else:
            content_length = response.meta['content_length']

        
        self.logger.info("Crawled resource with URL : " + link.get_url() + ". HTTP STATUS : " + str(response.status) + ". Iteration " + str(self.nb_episodes + 1) + ". Content length: " + str(content_length) + ". Link depth:" + str(link.depth))

        if http_response[0] in ['4', '5'] or (http_response[0] == '3' and location is None):
            self.count_request(0, False)
            self.logger.info("HTTP Error " + http_response + " : Resource \"" + link.get_url() + "\" is not available, or status is unknown. Identified at episode " + str(self.nb_episodes) + ". Not added to local DB.")
            self.not_exploitable_resources.add(link.get_url())
            
            request_to_add = self.update_scrapy_queue_if_needed()
            if request_to_add is not None:
                self.current_number_of_requests_in_scrapy_queue += 1
                self.yielded_to_scrapy.add(request_to_add.meta['link'])
                yield request_to_add
            return

        if http_response[0] == "2":
            if 'Content-Type' in headers:
                mime_type = response.headers.get(b'Content-Type', b'').decode('utf-8').lower()
                if ";" in mime_type:
                    mime_type = mime_type.split(";")[0]
                if 'html' in mime_type: 
                    try:
                        body = response.text
                    except:
                        body = ''
                    
                    if len(body) > 0 and not body.isspace():
                        new_elements, base_url = self.extract_urls_from_html(body)
                    else:
                        new_elements = []
                    if self.nb_episodes > 1:
                        self.add_labeled_data(link, 'non_target')
                    self.count_request(content_length, False)
                else:
                    is_data_resource = mime_type in self.mime_types_data_resources
                    if is_data_resource: 
                        self.logger.info("TAG_TARGET: Target found at iteration " + str(self.nb_episodes) + ", at URL \"" + str(link.get_url()) + "\". Content-Type: " + str(mime_type))
                        
                        self.data_resources.add(URL(link.get_url(), 'data_resource', mime_type, 'date_resource'))
                        if self.nb_episodes > 1:
                            self.add_labeled_data(link, 'target')
                    else:
                        if self.nb_episodes > 1:
                            self.add_labeled_data(link, 'non_target') 
                        self.not_exploitable_resources.add(link.get_url())

                    self.count_request(content_length, is_data_resource)
                    request_to_add = self.update_scrapy_queue_if_needed()
                    if request_to_add is not None:
                        self.current_number_of_requests_in_scrapy_queue += 1
                        self.yielded_to_scrapy.add(request_to_add.meta['link'])
                        yield request_to_add
                    return
            else:
                self.not_exploitable_resources.add(link.get_url())
                self.count_request(content_length, False)
                if self.nb_episodes > 1:
                    self.add_labeled_data(link, 'non_target')
                request_to_add = self.update_scrapy_queue_if_needed()
                if request_to_add is not None:
                    self.current_number_of_requests_in_scrapy_queue += 1
                    self.yielded_to_scrapy.add(request_to_add.meta['link'])
                    yield request_to_add
                return

        elif http_response[0] == '3':
            self.count_request(content_length, False)
            try:
                full_new_redirected_url = urldefrag(urljoin(link.get_url(), str(location))).url
            except:
                logging.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" in a redirection. It probably contains unhandled characters under NKFC normalization.")

                request_to_add = self.update_scrapy_queue_if_needed()
                if request_to_add is not None:
                    self.current_number_of_requests_in_scrapy_queue += 1
                    self.yielded_to_scrapy.add(request_to_add.meta['link'])
                    yield request_to_add
                return

            redirection_link = copy.copy(link)
            redirection_link.url = full_new_redirected_url

            self.logger.info("Status " + http_response + " encountered for URL " + link.get_url() + ", which was redirected to " + full_new_redirected_url + ".")
            if redirection_link not in self.already_visited and redirection_link not in self.yielded_to_scrapy  and not self.is_link_in_frontier(redirection_link) and redirection_link not in self.data_resources and redirection_link.get_url() not in self.not_exploitable_resources and self.is_url_on_same_or_sub_domain(self.starting_url, full_new_redirected_url) and len(full_new_redirected_url) <= len(self.starting_url) + self.max_len_from_root and not self.too_many_repeated_fragments(full_new_redirected_url, 3):
                self.current_number_of_requests_in_scrapy_queue += 1
                self.yielded_to_scrapy.add(redirection_link)
                yield scrapy.Request(redirection_link.get_url(), callback=self.parse, dont_filter=True, meta={'dont_redirect':True, 'link': redirection_link})
                return
             
            else:
                if self.nb_episodes > 1:
                    self.add_labeled_data(link, 'non_target')
                request_to_add = self.update_scrapy_queue_if_needed()
                if request_to_add is not None:
                    self.current_number_of_requests_in_scrapy_queue += 1
                    self.yielded_to_scrapy.add(request_to_add.meta['link'])
                    yield request_to_add
                return

        else:
            self.logger.info("Problem of status unhandled.")

        for new_element in new_elements:
            if base_url is None:
                try:
                    full_new_url = urldefrag(urljoin(link.get_url(), new_element['url'])).url
                except:
                    self.logger.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" . It probably contains unhandled characters under NKFC normalization.")
                    continue     
            else:
                try:
                    full_new_url = urldefrag(urljoin(base_url, new_element['url'])).url
                except:
                    self.logger.info("Exception occured while trying to parse the URL \"" + str(new_element['url']) + "\" . It probably contains unhandled characters under NKFC normalization.")
                    continue

            if not self.is_url_on_same_or_sub_domain(self.starting_url, full_new_url) or len(full_new_url) > len(self.starting_url) + self.max_len_from_root or self.too_many_repeated_fragments(full_new_url, 3):
                continue

            if len(full_new_url) > 2:
                if full_new_url[:2] == "//":
                    full_new_url = "https:" + full_new_url

            new_link = Link(full_new_url, new_element['dom_path'])

            if new_link not in self.already_visited and new_link not in self.yielded_to_scrapy and not self.is_link_in_frontier(new_link) and new_link not in self.data_resources and new_link.get_url() not in self.not_exploitable_resources:
                new_link.depth = link.depth + 1
                new_link.anchor = new_element['anchor_text']
                self.set_classifier_features(new_link)
                self.add_link(new_link)
 
        request_to_add = self.update_scrapy_queue_if_needed()

        if request_to_add is not None:
            self.current_number_of_requests_in_scrapy_queue += 1
            self.yielded_to_scrapy.add(request_to_add.meta['link'])
            yield request_to_add

    def closed(self, reason):
        self.logger.info("Crawler finished, reason : " + str(reason))
        self.connection_common_db.close()
