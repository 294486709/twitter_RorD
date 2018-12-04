import numpy as np
from random import shuffle
import multiprocessing as mp
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
#Blue VS Red
MAX_LENGTH = 30
NUM_FEATURE = 30
NUM_LABEL = 2
TESTING_RATE = 0.01
VAL_RATE = 0.1

def w2vtrain(data):
	num_worker = mp.cpu_count()
	tokenized = []
	for i in range(len(data)):
		tokenized.append(data[i][0])
	w2vmodel = Word2Vec(tokenized, workers=num_worker, size=NUM_FEATURE, min_count=1)
	w2vmodel.train(tokenized, total_examples=len(tokenized),epochs=100)
	w2vmodel.save("w2vmodel.w2v")


def dataset_gen(data):
	model = Word2Vec.load('w2vmodel.w2v')
	collection1 = []
	collection2 = []
	for i in range(len(data)):
		current = data[i][0]
		print(i,data[i][1])
		result = np.zeros([1,NUM_FEATURE])
		for j in range(len(current)):
			try:
				result = np.vstack((result, model[current[j]]))
			except:
				pass
		result = result[1:]
		if len(result) < MAX_LENGTH:
			for j in range(MAX_LENGTH - len(result)):
				result = np.vstack((result, np.zeros([1,NUM_FEATURE])))
		ylabel = np.zeros([NUM_LABEL])
		if int(data[i][1]) == 1:
			ylabel[0] = 1
			collection1.append([result, ylabel])
		elif int(data[i][1]) == 2:
			ylabel[1] = 1
			collection2.append([result, ylabel])
		print(ylabel)
	xtrain = []
	ytrain = []
	xval = []
	yval = []
	xtest = []
	ytest = []
	shuffle(collection1)
	shuffle(collection2)
	num_test1 = int(len(collection1) * TESTING_RATE)
	num_val1 = int(len(collection1) * VAL_RATE)
	num_test2 = int(len(collection2) * TESTING_RATE)
	num_val2 = int(len(collection2) * VAL_RATE)
	for i in range(len(collection1)):
		if i < num_test1:
			xtest.append(collection1[i][0])
			ytest.append(collection1[i][1])
		elif i < num_test1 + num_val1:
			xval.append(collection1[i][0])
			yval.append(collection1[i][1])
		else:
			xtrain.append(collection1[i][0])
			ytrain.append(collection1[i][1])
	for i in range(len(collection2)):
		if i < num_test2:
			xtest.append(collection2[i][0])
			ytest.append(collection2[i][1])
		elif i < num_test2 + num_val2:
			xval.append(collection2[i][0])
			yval.append(collection2[i][1])
		else:
			xtrain.append(collection2[i][0])
			ytrain.append(collection2[i][1])

	com = list(zip(xtrain, ytrain))
	shuffle(com)
	xtrain[:], ytrain[:] = zip(*com)

	com = list(zip(xtest, ytest))
	shuffle(com)
	xtest[:], ytest[:] = zip(*com)

	com = list(zip(xval, yval))
	shuffle(com)
	xval[:], yval[:] = zip(*com)

	np.save("dataset/33xtrain.npy",np.array(xtrain))
	np.save("dataset/33ytrain.npy", np.array(ytrain))
	np.save("dataset/33xtest.npy", np.array(xtest))
	np.save("dataset/33ytest.npy", np.array(ytest))
	np.save("dataset/33xval.npy",np.array(xval))
	np.save("dataset/33yval.npy",np.array(yval))



	pass


def main():
	unsorted_data = []
	att = ['D','R','N']
	counter = 1
	for i in att:
		f = open(str(i) + ".txt",'r')
		temp = f.readlines()
		for j in temp:
			unsorted_data.append([j.replace('\n',''), counter])
		counter += 1
	poplist = []
	longgest = 0
	for i in range(len(unsorted_data)):
		current = simple_preprocess(unsorted_data[i][0])
		if len(current) < 1:
			poplist.append(i)
		else:
			unsorted_data[i][0] = current
		if len(current) > longgest:
			longgest = len(current)
	if len(poplist) > 1:
		poplist.sort(reverse=True)
	for i in poplist:
		unsorted_data.pop(i)
	print(longgest)
	#w2vtrain(unsorted_data)
	dataset_gen(unsorted_data)
	pass


if __name__ == '__main__':
    main()