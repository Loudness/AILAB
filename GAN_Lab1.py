

##################################################
# GAN Lab1: Time series prediction
# Generative Adversarial Networks
# We will generate new images based on another dataset. 
# As the standard MNIST DB has been overused (hand written numbers) we will use the Zalandos fashion MNIST-like image DB instead. (Fashion-MNIST)
# This will also make it much harder to reach the same accuracy as the original MNIST DB.
# How does a GAN work?
# A GAN has 2 networks (other AI techniques are usually happy with only one), the Generator and the Discriminator.
# The Generator starts generating images from random noise and is then compared at the discriminator level how much is resembles the reality. 
# the Discriminator is fed authentic images and gives them a grade near 1.0 and the generator images will get a grade near 0.0 (at start) but will try to increase its value
# to resemble a real image. And this goes one until they reach an equilibrium.
# 
#  

################################################## 

# The Plan:
# 1. Load data 
# 2. Setup input to graph
# 3. Create Generator
# 4. Create Discriminator
# 5. Create the model (aka graph in Tensorflow)
# 6. Train it
# 7. Check model performance, output everything to its own folder for comparison with the other runs.
#    and show some fancy results 
# This is my attempt, there are other ways to do it

# Using the fashion data from Zalando Research as input.
# https://github.com/zalandoresearch/fashion-mnist
                                                                  
#/Aries

from dataload import csvloader
from dataload import fashiondatadownload
from datautils import filehelper
from datautils import graphhelper
from datautils import outputhelper 
from preprocessing import scalers
from preprocessing import transforms

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl     #Use cPickle for speed?
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#Windoze users there is still a bug with seabornin VS own files: If you get NameError: name 'channel' is not defined, see https://github.com/Microsoft/PTVS/pull/3259/files.
# edit C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\Extensions\Microsoft\Python\Core\ptvsd\debugger.py
import seaborn as sns                







def loadData():

####################################################################################################
# Part 1. Import data from file
####################################################################################################
    
    

    #Using Zalando Research MNIST data replacement. 
    #70 000 product in 10 categories, that is 7000 images percategory
    #Split Training 60 000 images and test 10 000 images
    #Overridden path to load Zalando instead of standard NMIST 
    #If it does not exist, download it.
    fashiondatadownload.FashionDownload.downloadIfNotExist()

    ZALANDO_DATA = 'data/fashion'
    if(os.path.isdir(ZALANDO_DATA)):
        data = input_data.read_data_sets(ZALANDO_DATA)
    else:
        print("Fashion Data does not exist!")
        data = None 

        #data = input_data.read_data_sets(ZALANDO_DATA, source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')   #Or better https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

    return data



####################################################################################################
# Part 2. Setup input to graph 
# model_inputs (Graph input), real_dim are real images, z_dim are generated images
####################################################################################################
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z


####################################################################################################
# Part 3. Setup Generator
####################################################################################################    
def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Manual Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        
        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)
        
        return out

####################################################################################################
# Part 4. Setup Discriminator, more or less the same as Genrator, except for the output layer
####################################################################################################   
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Manual Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        
        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)
        
        return out, logits


####################################################################################################
# PrePart 5. STARTUP and Setup Hyperparameters
#################################################################################################### 
def run_GANLab1():


    # Size of input image to discriminator
    INPUT_SIZE = 784   # Standard 28x28 Images flattened
    # Size of latent vector to generator
    Z_SIZE = 100
    # Sizes of hidden layers in generator and discriminator
    G_HIDDEN_SIZE = 128
    D_HIDDEN_SIZE = 128
    # Leak factor for leaky ReLU
    ALPHA = 0.01
    # Smoothing to help the discriminator to generalize better, from 1.0 to 0.9 (as an example) 
    SMOOTH = 0.1 

    
    inData = loadData() #Part 1 call
    if inData is None:
        print("No fashion data exists! Please download before continuing!!")
        return;
    
    resultFolder = filehelper.generateNewResultFolder() #creates new foldername based on timestamp and filename  under results/  
    


####################################################################################################
# Part 5. Setup Model
####################################################################################################   
    
    #Cleanup any existing graph
    tf.reset_default_graph()
    # Create our input placeholders
    input_real, input_z = model_inputs(INPUT_SIZE, Z_SIZE)

    # Build the model
    g_model = generator(input_z, INPUT_SIZE, n_units=G_HIDDEN_SIZE, alpha=ALPHA)
    # g_model is the generator output

    d_model_real, d_logits_real = discriminator(input_real, n_units=D_HIDDEN_SIZE, alpha=ALPHA)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=D_HIDDEN_SIZE, alpha=ALPHA)

    ####################################################################################################
    # Part 5.1 Calculate losses 
    #################################################################################################### 
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real) * (1 - SMOOTH)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_real)))
    d_loss = d_loss_real + d_loss_fake 
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.ones_like(d_logits_fake)))

    ####################################################################################################
    # Part 5.2 Setup Optimizers, AdamOptimizer seems to be the standard one for this kind of problem.
    #################################################################################################### 

    # Optimizers
    LEARNING_RATE = 0.002

    # Get the trainable_variables, split into G and D parts
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    d_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)

####################################################################################################
# Part 6. Train it!
####################################################################################################  

    #Constants to tweak
    BATCH_SIZE = 100     
    EPOCHS = 450         #default 100
    samples = []
    losses = []
    
    # Only save generator variables
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPOCHS):
            for ii in range(inData.train.num_examples//BATCH_SIZE):
                batch = inData.train.next_batch(BATCH_SIZE)
            
                # Get images, reshape and rescale to pass to D
                batch_images = batch[0].reshape((BATCH_SIZE, 784))
                batch_images = batch_images*2 - 1
            
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z_SIZE))
            
                # Run optimizers
                _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
            # At the end of each epoch, get the losses and print them out
            train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
            train_loss_g = g_loss.eval({input_z: batch_z})
            
            print("Epoch {}/{}...".format(e+1, EPOCHS),
                  "Discriminator Loss: {:.4f}...".format(train_loss_d),
                  "Generator Loss: {:.4f}".format(train_loss_g))    
            # Save losses to view after training
            losses.append((train_loss_d, train_loss_g))
        
            # Sample from generator as we're training for viewing afterwards
            sample_z = np.random.uniform(-1, 1, size=(16, Z_SIZE))
            gen_samples = sess.run(
                           generator(input_z, INPUT_SIZE, n_units=G_HIDDEN_SIZE, reuse=True, alpha=ALPHA),
                           feed_dict={input_z: sample_z})
            samples.append(gen_samples)
            #saver.save(sess, './checkpoints/generator.ckpt')
            saver.save(sess, resultFolder + os.sep + 'checkpoints'+ os.sep +'generator.ckpt')

   
    # Save training generator samples
    with open(resultFolder + os.sep + 'train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)


####################################################################################################
# Part 7. Check model performance and show som nice images.
####################################################################################################  

    # Load samples from generator taken while training
    with open(resultFolder + os.sep + 'train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)

    #enable override of seaborn over pyplot/matplotlib
    sns.set()

    # _ = view_samples(-1, samples)
    fig, axes = view_samples(-1, samples)
    fig.savefig(resultFolder + os.sep +'firstsample'+'.png', dpi=224)

    ROWS, COLS = 10, 6
    fig, axes = plt.subplots(figsize=(7,12), nrows=ROWS, ncols=COLS, sharex=True, sharey=True)

    plt.savefig(resultFolder + os.sep + "beeer.png")

    #Discriminator vs  Generator graph
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig(resultFolder + os.sep + "discVsGen.png")




    for sample, ax_row in zip(samples[::int(len(samples)/ROWS)], axes):
        for img, ax in zip(sample[::int(len(sample)/COLS)], ax_row):
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(resultFolder + os.sep +'checkpoints'))
        sample_z = np.random.uniform(-1, 1, size=(16, Z_SIZE))
        gen_samples = sess.run(generator(input_z, INPUT_SIZE, n_units=G_HIDDEN_SIZE, reuse=True, alpha=ALPHA),feed_dict={input_z: sample_z})
    
    #Just testing here
    # _ = view_samples(0, [gen_samples])
    fig, axes = view_samples(0, [gen_samples])
    fig.savefig(resultFolder + os.sep +'generated_Samples'+'.png', dpi=224)
    


#view samples taken while running.
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes  
