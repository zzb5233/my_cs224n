#import tensorflow and other libraries
import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time


#download the shakespeare dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#read the data
text = open(path_to_file).read()

vocab = sorted(set(text))
char2idx = {u:i, for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

#create training examples and targets
seq_length = 100
examples_per_epoch = len(text)//seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#create training batches
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#build the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation'sigmoid')
    
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        
        rnn(rnn_units, return_sequences=True,
            recurrent_initizlizer='glorot_uniform', 
            stateful=True),        
        
        tf.keras.layers.Dense(vocab_size)
    )
    return model

model = build_model(vocab_size = len(vocab), 
                    embedding_dim = embedding_dim, 
                    rnn_units=, batch_size)


#try the model
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

model.summary()
sampled_indices = tf.multinomial(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.sequeeze(sampled_indices, axis=-1).numpy()

#train the model
example_batch_loss = tf.losses.sparse_softmax_cross_entropy(target_example_batch, example_batch_predictions)
model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = tf.losses.sparse_softmax_cross_entropy)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, 
    save_weights_only=True)

EPOCHS=3
history = model.fit(dataset.repat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

#generate text
tf.train.latest_checkpoint(checkpoint_predir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string):
    num_generate = 1000
    start_string = 'ROMEO'
    
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []
    
    temerature = 1.0
    
    model.reset_states()
    for i in range(num_generage):
        predictions = tf.squeeze(predictions, 0)
        
        predictions = predictions / temperture
        predicted_id = tf.multinomial(predictioins, num_samples=1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

#Advanced: customized training
model = build_model(
    vocab_size = len(vocab), 
    embedding_dim = embedding_dim, 
    rnn_units = rnn_units, 
    batch_size = BATCH_SIZE)

optimizer = tf.train.AdaOptimizer()

EPOCHS = 1

for epoch in range(EPOCHS):
    
    hidden = model.reset_states()
    
    for (batchs_n, (inp, target)) in enumrate(dataset):
        with tf.GradientTap() as tape:
            preditions = model(inp)
            loss = tf.losses.spares_softmax_cross_entropy(target, predictions)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        
        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {:.4f}'
            print(template.format(epoch+1, batch_n, loss))
        
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))    

model.save_weights(checkpoint_prefix.format(epoch=epoch))