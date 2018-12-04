import numpy as np
from random import shuffle
import multiprocessing as mp
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess

MAX_LENGTH = 30
NUM_FEATURE = 30
NUM_LABEL = 3
TESTING_RATE = 0.2
VAL_RATE = 0.2
MAX_LINE = 40000

def w2vtrain(data):
	num_worker = mp.cpu_count()
	tokenized = []
	for i in range(len(data)):
		tokenized.append(data[i][0])
	w2vmodel = Word2Vec(tokenized, workers=num_worker, size=NUM_FEATURE, min_count=1)
	w2vmodel.train(tokenized, total_examples=len(tokenized),epochs=100)
	w2vmodel.save("w2vmodel.w2v")


def dataset_gen(data):
	model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	collection1 = []
	collection2 = []
	collection3 = []
	for i in range(len(data)):
		current = data[i][0]
		print(i,data[i][1])
		result = np.zeros([1,300])
		for j in range(len(current)):
			try:
				result = np.vstack((result, np.reshape(model[current[j]],[1,300])))
			except:
				pass
		result = result[1:]
		if len(result) < MAX_LENGTH:
			for j in range(MAX_LENGTH - len(result)):
				result = np.vstack((result, np.zeros([1,300])))
		# result = result.reshape([50,50,6,1])
		ylabel = np.zeros([NUM_LABEL])
		if int(data[i][1]) == 1:
			ylabel[0] = 1
			collection1.append([result, ylabel])
		elif int(data[i][1]) == 2:
			ylabel[1] = 1
			collection2.append([result, ylabel])
		else:
			ylabel[2] = 1
			collection3.append([result, ylabel])
		print(ylabel)
	xtrain = []
	ytrain = []
	xtest = []
	ytest = []
	xval = []
	yval = []
	shuffle(collection1)
	shuffle(collection2)
	shuffle(collection3)
	num_val1 = int(len(collection1) * VAL_RATE)
	num_val2 = int(len(collection2) * VAL_RATE)
	num_val3 = int(len(collection3) * VAL_RATE)
	num_test1 = int(len(collection1) * TESTING_RATE)
	num_test2 = int(len(collection2) * TESTING_RATE)
	num_test3 = int(len(collection3) * TESTING_RATE)

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
	for i in range(len(collection3)):
		if i < num_test3:
			xtest.append(collection3[i][0])
			ytest.append(collection3[i][1])
		elif i < num_test3 + num_val3:
			xval.append(collection3[i][0])
			yval.append(collection3[i][1])
		else:
			xtrain.append(collection3[i][0])
			ytrain.append(collection3[i][1])
	com = list(zip(xtrain, ytrain))
	shuffle(com)
	xtrain[:], ytrain[:] = zip(*com)

	com = list(zip(xtest, ytest))
	shuffle(com)
	xtest[:], ytest[:] = zip(*com)


	com = list(zip(xval, yval))
	shuffle(com)
	xval[:], yval[:] = zip(*com)
	npxtrain = np.array(xtrain[0])




	np.save("dataset/xxtrain.npy",xtrain)
	np.save("dataset/xytrain.npy", ytrain)
	np.save("dataset/xxtest.npy", xtest)
	np.save("dataset/xytest.npy", ytest)
	np.save("dataset/xxval.npy", xval)
	np.save("dataset/xyval.npy", yval)




	pass


def main():
	unsorted_data = []
	dataraw = ['D','R','N']
	for i in dataraw:
		counter = 0
		f = open(i + ".txt",'r')
		temp = f.readlines()
		for j in temp:
			if counter == MAX_LINE:
				f.close()
				break
			if i == 'D':
				k = 1
			elif i == 'R':
				k = 2
			else:
				k = 3
			unsorted_data.append([j.replace('\n',''), k])
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
	# w2vtrain(unsorted_data)
	dataset_gen(unsorted_data)
	pass


if __name__ == '__main__':
    main()