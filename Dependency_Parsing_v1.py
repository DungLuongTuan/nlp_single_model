import numpy as np 
import gensim
import tensorflow as tf 
from os import listdir

POS_tag = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
X_POS_tag = ['H', 'RV', 'N', 'Eb', 'Mb', 'Np', 'm', 'v', 'Nu', 'Nc', 'V', 'A', 'a', 'Ap', 'P', 'p', 'L', 'M', 'R', 'E', 'C', 'I', 'T', 'B', 'Y', 'S', 'X', 'Ny', 'NY', 'Vy', 'Ay', 'Nb', 'Vb', 'Ab', '.', ',', 'LBKT', 'RBKT', 'CC', '"', '...', '@', '-', '!', ':', 'Z', '?', 'NP', ';', 'VP', '(', ')', '“', '”', '[', ']', '…', '/', '----------', '*', '---------', '--------', '----------------', '------', '', '+', '#', '$', '&', '^', '=', '_', '`', '~', '’', '<', '>', '..', 'oOo']
DP_tag = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'auxpass', 'aux:pass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'discourse', 'dislocated', 'dobj', 'expl', 'fixed', 'flat', 'foreign', 'goeswith', 'iobj', 'obl', 'list', 'mark', 'neg', 'nmod', 'nsubj', 'nsubjpass', 'nummod', 'obj', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']
pos_len = len(POS_tag)
dp_len = len(DP_tag)

def load_data(Doc2vec, batch_size, f_POS, f_head, f_DP):
	max_state = 150
	### load data + pos labels
	POS_data = []
	POS_labels = []
	sequence_length = []
	row_data = []
	row_POS_labels = []
	sen_in_batch = 0
	for row in f_POS:
		list_columns = row[:-1].split('\t')
		if not (list_columns[0].isdigit()):
			if not (len(row_data) == 0):
				sequence_length.append(len(row_data))
				while len(row_data) < max_state:
					row_data.append(np.zeros(100))
					row_POS_labels.append(np.zeros(pos_len))
				POS_data.append(row_data)
				POS_labels.append(row_POS_labels)
				sen_in_batch += 1
				if (sen_in_batch == batch_size):
					break
			row_data = []
			row_POS_labels = []
			continue
		row_data.append(Doc2vec.infer_vector(list_columns[1]))
		label = np.zeros(len(POS_tag))
		label[POS_tag.index(list_columns[3])] = 1
		row_POS_labels.append(label)
	### load head labels
	head_labels = []
	row_head_labels = np.zeros((max_state, max_state))
	ok = False
	cnt = 0
	sen_in_batch = 0
	for row in f_head:
		list_columns = row[:-1].split('\t')
		if not (list_columns[0].isdigit()):
			if (ok):
				head_labels.append(row_head_labels)
				ok = False
				cnt += 1
				sen_in_batch += 1
				if (sen_in_batch == batch_size):
					break
			row_head_labels = np.zeros((max_state, max_state))
			continue
		ok = True
		if (int (list_columns[6]) == 0):
			row_head_labels[int (list_columns[0]) - 1, sequence_length[cnt]] = 1
		else:
			row_head_labels[int (list_columns[0]) - 1, int (list_columns[6]) - 1] = 1
	### load DP labels
	DP_labels = []
	row_DP_labels = np.zeros((max_state, max_state, dp_len))
	ok = False
	cnt = 0
	sen_in_batch = 0
	for row in f_DP:
		list_columns = row[:-1].split('\t')
		if not (list_columns[0].isdigit()):
			if (ok):
				DP_labels.append(row_DP_labels)
				ok = False
				cnt += 1
				sen_in_batch += 1
				if (sen_in_batch == batch_size):
					break
			row_DP_labels = np.zeros((max_state, max_state, dp_len))
			continue
		ok = True
		if (int (list_columns[6]) == 0):
			row_DP_labels[int (list_columns[0]) - 1, sequence_length[cnt], DP_tag.index(list_columns[7])] = 1
		else:
			row_DP_labels[int (list_columns[0]) - 1, int (list_columns[6]) - 1, DP_tag.index(list_columns[7])] = 1

	return POS_data, POS_labels, head_labels, DP_labels, sequence_length

def main():
	print('load Doc2vec model')
	Doc2vec = gensim.models.doc2vec.Doc2Vec.load('best.doc2vec.model')
	### make graph
	print('make graph')
	n_hidden = 100 # also n_cell
	max_state = 150
	epochs = 50
	batch_size = 10
	x = tf.placeholder(tf.float32, [None, max_state, 100])
	y_POS_ = tf.placeholder(tf.float32, [None, max_state, pos_len])
	y_head_ = tf.placeholder(tf.float32, [None, max_state, max_state])
	y_DP_ = tf.placeholder(tf.float32, [None, max_state, max_state, dp_len])
	sequence_length = tf.placeholder(tf.int32, [None])
	w_POS = tf.Variable(tf.random_normal([2*n_hidden, pos_len]), name = 'w_POS')
	b_POS = tf.Variable(tf.random_normal([1, pos_len]), name = 'b_POS')
	w_head = tf.Variable(tf.random_normal([2*n_hidden, 2*n_hidden]), name = 'w_head')
	w_DP = tf.Variable(tf.random_normal([4*n_hidden, dp_len]), name = 'w_DP')
	b_DP = tf.Variable(tf.random_normal([1, dp_len]), name = 'b_DP')
	### POS layer
	fw_lstm_POS = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	bw_lstm_POS = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	with tf.variable_scope('POSlayer'):
		(output_fw_POS, output_bw_POS), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_lstm_POS, cell_bw = bw_lstm_POS, sequence_length = sequence_length, inputs = x, dtype = tf.float32, scope = 'POSlayer')
	# output of B-LSTM also concatenation hidden state
	h_POS = tf.concat([output_fw_POS, output_bw_POS], axis = -1)
	# process output of POS B-LSTM layer
	output_POS_relu = tf.reshape(tf.nn.relu(h_POS), [-1, 2*n_hidden])
	output_POS_slice = tf.nn.softmax(tf.matmul(output_POS_relu, w_POS) + b_POS)
	output_POS = tf.reshape(output_POS_slice, [-1, max_state, pos_len])
	# prepare for next layer
	y_POS_slice = tf.matmul(output_POS_slice, tf.transpose(w_POS))
	y_POS = tf.reshape(y_POS_slice, [-1, max_state, 2*n_hidden])
	### dependency parsing layer
	x_DP = tf.concat([h_POS, x, y_POS], axis = -1)
	fw_lstm_DP = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	bw_lstm_DP = tf.contrib.rnn.LSTMCell(num_units = n_hidden)
	with tf.variable_scope('DPlayer'):
		(output_fw_DP, output_bw_DP), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_lstm_DP, cell_bw = bw_lstm_DP, sequence_length = sequence_length, inputs = x_DP, dtype = tf.float32, scope = 'DPlayer')
	h_DP = tf.concat([output_fw_DP, output_bw_DP], axis = -1)
	h_DP_slice = tf.reshape(h_DP, [-1, 2*n_hidden])
	# process output for head marking
	w_h = tf.reshape(tf.matmul(h_DP_slice, w_head), [-1, max_state, 2*n_hidden])
	h_w_h = tf.matmul(h_DP, tf.transpose(w_h, [0, 2, 1]))
	output_head = tf.nn.softmax(h_w_h)
	# process output for Dependency Parsing
	b1 = tf.reshape(tf.tile(h_DP, [1, 1, max_state]), [-1, max_state*max_state, 2*n_hidden])
	b2 = tf.tile(h_DP, [1, max_state, 1])
	ht_hj = tf.concat([b1, b2], axis = -1)
	output_DP_relu = tf.reshape(tf.nn.relu(ht_hj), [-1, 4*n_hidden])
	output_DP_slice = tf.nn.softmax(tf.matmul(output_DP_relu, w_DP) + b_DP)
	output_DP = tf.reshape(output_DP_slice, [-1, max_state, max_state, dp_len])
	### loss function and optimiers
	loss_POS = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(y_POS_*tf.log(output_POS), reduction_indices = [2]), reduction_indices = [1]))
	loss_head = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(y_head_*tf.log(output_head), reduction_indices = [2]), reduction_indices = [1]))
	loss_DP = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(-tf.reduce_sum(y_DP_*tf.log(output_DP), reduction_indices = [3]), reduction_indices = [2]), reduction_indices = [1]))
	optimizer_POS = tf.train.AdamOptimizer(0.01).minimize(loss_POS)
	optimizer_head = tf.train.AdamOptimizer(0.0001).minimize(loss_head)
	optimizer_DP = tf.train.AdamOptimizer(0.0001).minimize(loss_DP)
	### run models
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	saver = tf.train.Saver()

	for epoch in range(epochs):
		f_POS = open('./data/vi-ud-train.conllu', 'r')
		f_head = open('./data/vi-ud-train.conllu', 'r')
		f_DP = open('./data/vi-ud-train.conllu', 'r')
		start = 0
		sum_POS_loss = 0
		sum_head_loss = 0
		sum_DP_loss = 0
		cnt = 0
		while (start <= 1400 - 1):
			start += batch_size
			cnt += 1
			print('batch: ', cnt, end = '\r')
			batch_data, batch_POS_labels, batch_head_labels, batch_DP_labels, batch_seqlen = load_data(Doc2vec, batch_size, f_POS, f_head, f_DP)
			sess.run((optimizer_POS, optimizer_head, optimizer_DP), feed_dict = {x: batch_data, y_POS_: batch_POS_labels, y_head_: batch_head_labels, y_DP_: batch_DP_labels, sequence_length: batch_seqlen})
			sum_POS_loss += sess.run(loss_POS,  feed_dict = {x: batch_data, y_POS_: batch_POS_labels, sequence_length: batch_seqlen})
			sum_head_loss += sess.run(loss_head, feed_dict = {x: batch_data, y_POS_: batch_POS_labels, y_head_: batch_head_labels, sequence_length: batch_seqlen})
			sum_DP_loss += sess.run(loss_DP, feed_dict = {x: batch_data, y_POS_: batch_POS_labels, y_head_: batch_head_labels, y_DP_: batch_DP_labels, sequence_length: batch_seqlen})
		print('epoch: ', epoch, ' ', 'loss_POS: ', sum_POS_loss, ' loss_head: ', sum_head_loss, ' loss_DP: ', sum_DP_loss)
		if ((epoch + 1) % 5 == 0):
			saver.save(sess, './models/0.01-0.0001-0.0001/'+ str(epoch + 1) + '.ckpt')
		f_POS.close()
		f_head.close()
		f_DP.close()
		
	### evaluate on training data
	f_POS = open('./data/vi-ud-train.conllu', 'r')
	f_head = open('./data/vi-ud-train.conllu', 'r')
	f_DP = open('./data/vi-ud-train.conllu', 'r')
	print('evaluate on training data')
	# evaluate for POS
	start = 0
	cnt_POS = 0
	sum_cnt_POS = 0
	cnt_head = 0
	sum_cnt_head = 0
	cnt_DP = 0
	sum_cnt_DP = 0
	while (start <= 1400 - 1):
		start += 1
		data, POS_labels, head_labels, DP_labels, sequence_length_arr = load_data(Doc2vec, 1, f_POS, f_head, f_DP)
		for i in range(len(sequence_length_arr)):
			result = sess.run(output_POS, feed_dict = {x: np.array([data[i]]), sequence_length: [sequence_length_arr[i]]})
			for j in range(sequence_length_arr[i]):
				sum_cnt_POS += 1
				if (np.argmax(POS_labels[i][j]) == np.argmax(result[0][j])):
					cnt_POS += 1
		# evaluate for head
		for i in range(len(sequence_length_arr)):
			result = sess.run(output_head, feed_dict = {x: np.array([data[i]]), sequence_length: [sequence_length_arr[i]]})
			sum_cnt_head += sequence_length_arr[i]
			for j in range(sequence_length_arr[i]):
				if (np.argmax(head_labels[i][j]) == np.argmax(result[0][j])):
					cnt_head += 1
		# evaluate for dependency parsing
		for i in range(len(sequence_length_arr)):
			head_result = sess.run(output_head, feed_dict = {x: np.array([data[i]]), sequence_length: [sequence_length_arr[i]]})
			DP_result = sess.run(output_DP, feed_dict = {x: np.array([data[i]]), sequence_length: [sequence_length_arr[i]]})
			sum_cnt_DP += sequence_length_arr[i]
			for j in range(sequence_length_arr[i]):
				predicted_head_of_word = np.argmax(result[0][j])
				predicted_DP = np.argmax(DP_result[0][j][predicted_head_of_word])
				true_head_of_word = np.argmax(head_labels[i][j])
				true_DP = np.argmax(DP_labels[i][j][true_head_of_word])
				if (predicted_DP == true_DP):
					cnt_DP += 1
	f_POS.close()
	f_head.close()
	f_DP.close()
	print('evaluate for POS')
	print(cnt_POS, '/', sum_cnt_POS)
	print(cnt_POS/sum_cnt_POS)
	print('evaluate for head')
	print(cnt_head, '/', sum_cnt_head)
	print(cnt_head/sum_cnt_head)
	print('evaluate for DP')
	print(cnt_DP, '/', sum_cnt_DP)
	print(cnt_DP/sum_cnt_DP)
	### evaluate on testing data
	f_POS = open('./data/vi-ud-dev.conllu', 'r')
	f_head = open('./data/vi-ud-dev.conllu', 'r')
	f_DP = open('./data/vi-ud-dev.conllu', 'r')
	print('evaluate on testing data')
	# evaluate for POS
	start = 0
	cnt_POS = 0
	sum_cnt_POS = 0
	cnt_head = 0
	sum_cnt_head = 0
	cnt_DP = 0
	sum_cnt_DP = 0
	while (start <= 1400 - 1):
		start += 1
		test_data, test_POS_labels, test_head_labels, test_DP_labels, test_sequence_length_arr = load_data(Doc2vec, 1, f_POS, f_head, f_DP)
		for i in range(len(test_sequence_length_arr)):
			result = sess.run(output_POS, feed_dict = {x: np.array([test_data[i]]), sequence_length: [test_sequence_length_arr[i]]})
			for j in range(test_sequence_length_arr[i]):
				sum_cnt_POS += 1
				if (np.argmax(test_POS_labels[i][j]) == np.argmax(result[0][j])):
					cnt_POS += 1
		# evaluate for head
		for i in range(len(test_sequence_length_arr)):
			result = sess.run(output_head, feed_dict = {x: np.array([test_data[i]]), sequence_length: [test_sequence_length_arr[i]]})
			sum_cnt_head += test_sequence_length_arr[i]
			for j in range(test_sequence_length_arr[i]):
				if (np.argmax(test_head_labels[i][j]) == np.argmax(result[0][j])):
					cnt_head += 1
		# evaluate for dependency parsing
		for i in range(len(test_sequence_length_arr)):
			head_result = sess.run(output_head, feed_dict = {x: np.array([test_data[i]]), sequence_length: [test_sequence_length_arr[i]]})
			DP_result = sess.run(output_DP, feed_dict = {x: np.array([test_data[i]]), sequence_length: [test_sequence_length_arr[i]]})
			sum_cnt_DP += test_sequence_length_arr[i]
			for j in range(test_sequence_length_arr[i]):
				predicted_head_of_word = np.argmax(result[0][j])
				predicted_DP = np.argmax(DP_result[0][j][predicted_head_of_word])
				true_head_of_word = np.argmax(test_head_labels[i][j])
				true_DP = np.argmax(test_DP_labels[i][j][true_head_of_word])
				if (predicted_DP == true_DP):
					cnt_DP += 1
	f_POS.close()
	f_head.close()
	f_DP.close()
	print('evaluate for POS')
	print(cnt_POS, '/', sum_cnt_POS)
	print(cnt_POS/sum_cnt_POS)
	print('evaluate for head')
	print(cnt_head, '/', sum_cnt_head)
	print(cnt_head/sum_cnt_head)
	print('evaluate for DP')
	print(cnt_DP, '/', sum_cnt_DP)
	print(cnt_DP/sum_cnt_DP)

if __name__ == '__main__':
	main()