from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import tensorflow as tf
import numpy as np
import os
import csv
NUM_FEATRUE = 300
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MAX_LENGTH = 250

def convert(titles,model):
	Vresult = []
	for title in titles:
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
		Vresult.append(result)
	return Vresult

def test(titles):
	tf.reset_default_graph()
	result = []
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('Rnn/result.ckpt.meta')
		saver.restore(sess, tf.train.latest_checkpoint('Rnn'))
		graph = tf.get_default_graph()
		X = graph.get_tensor_by_name('X:0')
		Y = graph.get_tensor_by_name('outt:0')
		for title in titles:
			res = sess.run(Y, feed_dict={X:title.reshape([1,30,300])})
			res = list((res)[0])
			result.append(res.index(max(res)))
	return result

def readfile(s):
	f = open(s + '_Val.txt')
	VAL = []
	data = f.readlines()
	for i in data:
		VAL.append(i[:-1])
	f.close()
	return VAL

def write(res,VAL,s):
	f = open(s + '_Val_res.txt','w')
	c1=0
	c2=0
	c3=0
	for i in range(len(res)):
		if res[i] == 0:
			current = 'Democrat'
			c1+=1
		elif res[i] == 1:
			current = 'Republican'
			c2+=1
		else:
			current = 'Neutral'
			c3+=1
		f.write(VAL[i].replace(',',';'))
		f.write("--------------")
		f.write(current)
		f.write('\n')
	f.close()
	print(c1/1000,c2/1000,c3/1000,s)


def main():
	NVAL = readfile('N')
	DVAL = readfile('D')
	RVAL = readfile('R')
	model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	VecNVAL = convert(NVAL,model)
	VecDVAL = convert(DVAL,model)
	VecRVAL = convert(RVAL,model)
	resDAVL = test(VecDVAL)
	resNAVL = test(VecNVAL)
	resRAVL = test(VecRVAL)
	write(resRAVL, RVAL, 'R')
	write(resDAVL, DVAL, 'D')
	write(resNAVL, NVAL, 'N')




if __name__ == '__main__':
    main()