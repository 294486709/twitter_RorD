from bs4 import BeautifulSoup
import requests
import os
from urllib import request
import split_sentence
import gensim
import tensorflow as tf
import numpy as np
import dashboard


os.makedirs('./articles', exist_ok=True)

class Crawl_CNN:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_begin = soup.find_all('p', {'class': 'zn-body__paragraph speakable'})
        for m in content_begin:
            with open('./articles/%s' % article_name, 'a') as f1:
                f1.write(m.get_text())

        content_body = soup.find_all('div', {'class': 'zn-body__paragraph'})
        for i in content_body:
            with open('./articles/%s' % article_name, 'a') as f2:
                f2.write(i.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_BBC:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_body = soup.find_all('div', {'class': 'story-body__inner'})
        for i in content_body:
            j = i.find_all('p')
            for k in j:
                with open('./articles/%s' % article_name, 'a') as f1:
                    f1.write(k.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_NYtimes:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_body = soup.find_all('p', {'class': 'css-1ygdjhk e2kc3sl0'})
        for m in content_body:
            with open('./articles/%s' % article_name, 'a') as f1:
                f1.write(m.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_Time:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')

        # write body
        content_body = soup.find_all('div', {'class': 'padded'})
        for m in content_body:
            n = m.find_all('p')
            for i in n:
                with open('./articles/%s' % article_name, 'a') as f:
                    f.write(i.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

class Crawl_Newsweek:

    def __init__(self, url):
        self.url = url

    def writeToFile(self):
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15;"
        headers = {'user-Agent': user_agent}
        req = request.Request(self.url, headers=headers)
        html = request.urlopen(req).read()
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        article_name = head + '.txt'

        # write head
        with open('./articles/%s' % article_name, 'a') as f:
            f.write(head)
            f.write('\n')


        # write body
        content_body = soup.find_all('div', {'class': 'article-body'})
        for m in content_body:
            n = m.find_all('p')
            for i in n:
                with open('./articles/%s' % article_name, 'a') as f1:
                    f1.write(i.get_text())

        print('%s --has been downloaded\n' % head)

    def get_head(self):
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        head = soup.h1.get_text()
        return head

def main(url):

    url_cnn = 'https://www.cnn.com'
    url_bbc = 'https://www.bbc.com'
    url_nytimes = 'https://www.nytimes.com'
    url_time = 'http://time.com'
    url_newsweek = 'https://www.newsweek.com'

    if url_cnn in url:
        write_to_file = Crawl_CNN(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_bbc in url:
        write_to_file = Crawl_BBC(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_nytimes in url:
        write_to_file = Crawl_NYtimes(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_time in url:
        write_to_file = Crawl_Time(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    elif url_newsweek in url:
        write_to_file = Crawl_Newsweek(url)
        write_to_file.writeToFile()
        return write_to_file.get_head()
    else:
        print('crawl is not available on this website')

def convert(title,model):
	# tokenize and delete excess information
	# makeing it 30 hy 30
	result = np.zeros([1,300])
	if len(title) > 30:
		title = title[:30]

	for i in title:
		try:
			result = np.vstack([result, model[i]])
		except:
			pass
	result = result[1:]
	# padding with 0
	if len(result) < 30:
		temp = np.zeros([1,300])
		for i in range(30-len(result)):
			result = np.vstack([result, temp])
	return  result


if __name__ == '__main__':
    while True:
        url = input('please input url:\n')
        if url == 'break':
            print('finish')
            break
        else:
            Dcount = 0
            Rcount = 0
            Ncount = 0
            a = split_sentence.split_sentence(main(url))
            for i in range(len(a)):
                a[i] = gensim.utils.simple_preprocess(a[i])
            model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            for i in range(len(a)):
                a[i] = convert(a[i], model)
            tf.reset_default_graph()
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph("Rnn/result.ckpt.meta")
                saver.restore(sess, tf.train.latest_checkpoint("Rnn"))
                graph = tf.get_default_graph()
                X = graph.get_tensor_by_name('X:0')
                Y = graph.get_tensor_by_name('outt:0')
                result = []
                cc = 0
                twetnp = np.array(a)
                print(twetnp.shape)
                result = sess.run(tf.nn.softmax(Y), feed_dict={X: twetnp})
                # result = tf.nn.softmax(result)
                # ans = tf.arg_max(result)
                for k in range(len(result)):
                    maxindex = list(result[k]).index(max(list(result[k])))
                    if maxindex == 0:
                        print("Democrat")
                        Dcount += 1
                    elif maxindex == 1:
                        print("Republican")
                        Rcount += 1
                    else:
                        Ncount += 1
                        print("Neutral")
            temp = [Dcount, Rcount, Ncount]
            ttmp = ['Democrat', 'Republican', 'Neutral']
            res = ttmp[temp.index(max(temp))]
            print('URL:{}, Number of Tweets:{}, Democrat:{}, Republican:{}, Neutral:{}, Result:{}'.format(url,len(a),Dcount,Rcount,Ncount,res))
            dashboard.dashboard(res)

            pass


