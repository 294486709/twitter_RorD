import numpy as np
import os
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NUM_FEATRUE = 300

import tweepy

#Get your Twitter API credentials and enter them here
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""


#method to get a user's last 100 tweets
def get_tweets(username):

	#http://tweepy.readthedocs.org/en/v3.1.0/getting_started.html#api
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)

	#set count to however many tweets you want; twitter only allows 200 at once
	number_of_tweets = 200

	#get tweets
	tweets = api.user_timeline(screen_name = username,count = number_of_tweets)

	#create array of tweet information: username, tweet id, date/time, text
	tweets_for_csv = [[username,tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in tweets]
	tweetss = []
	for i in tweets_for_csv:
		tweetss.append(i[3].decode('utf-8'))
	#write to a new csv file from the array of tweets
	print('Twitter Fatch Complete')
	return tweetss
# title is a string
def convert(title,model):
	# tokenize and delete excess information
	title = simple_preprocess(title)
	# makeing it 30 hy 30
	result = np.zeros([1,NUM_FEATRUE])
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
		temp = np.zeros([1,NUM_FEATRUE])
		for i in range(30-len(result)):
			result = np.vstack([result, temp])
	return  result


def main():
	model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	while(1):
		Dcount = 0
		Rcount = 0
		Ncount = 0
		title = input("Please input a twitter account:")
		tweets = get_tweets(title)
		num_tweets = len(tweets)
		# title = "Tensorflow"
		for i in range(len(tweets)):
			tweets[i] = convert(tweets[i],model)
		# testing
		tf.reset_default_graph()
		with tf.Session() as sess:
			saver = tf.train.import_meta_graph("Rnn/result.ckpt.meta")
			saver.restore(sess, tf.train.latest_checkpoint("Rnn"))
			graph = tf.get_default_graph()
			X = graph.get_tensor_by_name('X:0')
			Y = graph.get_tensor_by_name('outt:0')
			result = []
			cc = 0
			twetnp = np.array(tweets)
			print(twetnp.shape)
			result = sess.run(tf.nn.softmax(Y), feed_dict={X : twetnp})
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
		temp = [Dcount,Rcount,Ncount]
		ttmp = ['Democrat','Republican','Neutral']
		res = ttmp[temp.index(max(temp))]
		print('Account:{}, Number of Tweets:{}, Democrat:{}, Republican:{}, Neutral:{}, Result:{}'.format(title,num_tweets,Dcount/num_tweets,Rcount/num_tweets,Ncount/num_tweets,res))





if __name__ == '__main__':
    main()

#convert(title)
