import scrapy
import sqlite3
from urllib.parse import unquote, urljoin, urldefrag, urlparse
from lxml import html
import logging
from datetime import datetime
import re
import sys
from collections import Counter

class OnlineToLocalSpider(scrapy.Spider):
    handle_httpstatus_list = [301, 302, 303, 307, 308]

    def __init__(self, starting_url, site_name):
        self.name = "online_to_local_crawler"
        self.start_urls = [starting_url]
        self.name_of_table = site_name
        self.visited_urls = set()
        self.nb_crawled = 0
        self.max_len_from_root = 350 # Maximum depth, to limit robot traps
        self.nb_urls_rejected_by_len = 0
        self.already_rejected = set()
        self.connection = sqlite3.connect("../local_crawlers/data/" + self.name_of_table + ".db")
        self.create_table()
        self.prologue_pattern = re.compile(r'<\?xml\s.*?\?>')

    def create_table(self):
        cur = self.connection.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS " + self.name_of_table + " (url TEXT PRIMARY KEY, http_response TEXT, headers TEXT, body TEXT, content_length INTEGER);")
        self.connection.commit()

    def insert_new_data(self, url, http_response, headers, body, content_length):
        cur = self.connection.cursor()
        cur.execute("INSERT INTO " + self.name_of_table + " (url, http_response, headers, body, content_length) VALUES (?, ?, ?, ?, ?);", (url, http_response, headers, body, content_length))
        self.connection.commit()

    def extract_urls_from_html(self, html_content):
        urls = []

        cleaned_html_content = re.sub(self.prologue_pattern, '', html_content)
        tree = html.fromstring(cleaned_html_content)

        base_element = tree.find(".//base")
        base_url = base_element.get('href') if base_element is not None else None

        xpath_expressions = [
                '//a/@href',
                '//area/@href',
                '//frame/@src',
                '//iframe/@src',
                ]

        for xpath_expression in xpath_expressions:
            urls.extend(tree.xpath(xpath_expression))

        return urls, base_url

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

    def parse(self, response):
        if self.nb_crawled != 0:
            url = response.meta['original_url']
        else:
            url = self.start_urls[0]

        self.nb_crawled += 1

        self.logger.info("Crawled resource with original URL : " + url + " (response.url : " + str(response.url) + "). HTTP STATUS : " + str(response.status) + ". Iteration " + str(self.nb_crawled) + ". Content length: " + str(len(response.body)))

        self.visited_urls.add(url)
        if isinstance(response.body, str):
            content_length = len((response.body).encode())
        else:
            content_length = len(response.body)
        body = "Either no HTML resource, or no CT in headers."

        content_type = response.headers.get(b'Content-Type', b'').decode('utf-8').lower()

        if 'html' in content_type:
            body = response.text

        self.insert_new_data(url, str(response.status), str(dict(response.headers)), body, content_length)

        if 'html' in content_type and '200' in str(response.status):
            links, base_url = self.extract_urls_from_html(response.text)
            for l in links:
                if base_url is None:
                    new_url = urldefrag(urljoin(url, l)).url
                else:
                    new_url = urldefrag(urljoin(base_url, l)).url
                
                if new_url in self.visited_urls:
                    continue

                if not self.is_url_on_same_or_sub_domain(self.start_urls[0], new_url) or len(new_url) > len(self.start_urls[0]) + self.max_len_from_root or self.too_many_repeated_fragments(new_url, 3):
                    continue

                yield scrapy.Request(new_url, callback=self.parse, meta={'original_url':new_url, 'dont_redirect':True})
        
        elif response.status in self.handle_httpstatus_list:
            location = response.headers.get(b'Location', b'').decode('utf-8').lower()
            new_url = urldefrag(urljoin(url, location)).url

            if self.is_url_on_same_or_sub_domain(self.start_urls[0], new_url) and len(new_url) <= len(self.start_urls[0]) + self.max_len_from_root and not self.too_many_repeated_fragments(new_url, 3):
                self.logger.info("Redirection " + str(response.status) + " : Location is " + location + " from original URL : " + url + ". Once they are joined, they give : " + new_url + ".")
                yield scrapy.Request(new_url, callback=self.parse, meta={'original_url':new_url, 'dont_redirect': True})

    def closed(self, reason):
        self.logger.info("Crawler finished, reason : " + str(reason))
        self.connection.close()

if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess

    starting_url = sys.argv[1]
    site_name = sys.argv[2]
    max_number_of_responses = int(sys.argv[3])

    logging.basicConfig(filename="online_to_local_crawler/logs/" + (datetime.now()).strftime("%d-%m-%Y_%H:%M:%S") + "_"  + site_name + ".out", level=logging.INFO)

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
        'DEPTH_LIMIT': 1000,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 1,
        'CLOSESPIDER_PAGECOUNT':max_number_of_responses,
        #'COMPRESSION_ENABLED':False # Sometimes required, depending on websites
        })

    process.crawl(OnlineToLocalSpider, starting_url=starting_url, site_name=site_name)
    process.start()

    site_name_to_url_connection = sqlite3.connect("../local_crawlers/data/site_names_to_urls.db")
    site_name_to_url_cursor = site_name_to_url_connection.cursor()
    site_name_to_url_cursor.execute("INSERT INTO site_names_to_urls (site_name, start_url) VALUES (?, ?)", (site_name, starting_url))
    site_name_to_url_connection.commit()
    site_name_to_url_connection.close()
