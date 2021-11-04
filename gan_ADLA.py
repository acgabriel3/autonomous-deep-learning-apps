# -*- coding: utf-8 -*-
"""
@author: gabri

Use of CGAN's to solve the problem: Autonomous deep learning app

"""


from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import random 


fluxograma = pd.read_csv("data/app_minimo/fluxograma.csv", sep = ";")
#fluxograma = fluxograma[0:7]  #more simple train

#More complex train, with very distinc screens
fluxograma = fluxograma.append(fluxograma[7:15])
fluxograma = fluxograma.append(fluxograma[7:15])
fluxograma = fluxograma.append(fluxograma[7:15])
fluxograma = fluxograma.sample(frac=1).reset_index(drop=True)

shape = (int(970*1/8), int(910*1/8),3)
length = shape[0]*shape[1]*shape[2]
dim = (shape[0],shape[1])
dim2 = (shape[0]*4,shape[1]*4)
click_examples = 20


xclick = []
xim = []
#y = np.zeros((len(fluxograma)*click_examples,length))
y = []
pos = 0
for i,row in fluxograma.iterrows():
    for j in range(0,click_examples):
        coord1 = random.uniform(int(row.min1), int(row.max1))
        coord2 = random.uniform(int(row.min1), int(row.min2))
        im1 = cv2.imread("data/app_minimo/" + str(row.tela))
        im1 = np.histogram(im1.flatten(), 256, [0, 256])
        coord = np.asarray([coord1,coord2])
        xclick.append(coord)
        xim.append(im1[0])
        
        im2 = cv2.imread("data/app_minimo/" + str(row.proxima_tela))
        #y[pos] = cv2.resize(im2, dim, interpolation = cv2.INTER_AREA)
        y.append(cv2.resize(im2, dim2, interpolation = cv2.INTER_AREA))
        pos = pos + 1

xim = np.asarray(xim)#.astype('float32')
xclick = np.asarray(xclick)
y = np.asarray(y)#.astype('float32')

array = range(0,len(xim))
sample = random.sample(array, len(array) )

xim = xim[sample]
xclick = xclick[sample]
y = y[sample]


# Create the discriminator.

#false_image_input_ = layers.Input(shape=(dim2[1],dim2[0],shape[2]), name = 'false')
#false_image_input = layers.MaxPooling2D(2,2)(false_image_input_)
#false_image_input = layers.MaxPooling2D(2,2)(false_image_input)
#false_image_input = layers.MaxPooling2D(2,2)(false_image_input)
#false_image_input = layers.Flatten()(false_image_input)

#true_image_input_ = layers.Input(shape=(dim2[1],dim2[0],shape[2]), name = 'true')
#true_image_input = layers.MaxPooling2D(2,2)(true_image_input_)
#true_image_input = layers.MaxPooling2D(2,2)(true_image_input)
#true_image_input = layers.MaxPooling2D(2,2)(true_image_input)
#true_image_input = layers.Flatten()(true_image_input)


#discriminator = layers.concatenate([false_image_input,true_image_input])
#discriminator = layers.Dense(3675)(discriminator)
#discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)      
#discriminator = layers.Dropout(0.3)(discriminator)

#discriminator = layers.Reshape((35,35,3))(discriminator)
true_image_input_ = layers.Input(shape=(dim2[1],dim2[0],shape[2]), name = 'input')
#true_image_input = layers.MaxPooling2D(2,2)(true_image_input_)
#true_image_input = layers.MaxPooling2D(2,2)(true_image_input_)
discriminator = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same",kernel_regularizer=regularizers.l1(0.01))(true_image_input_)
discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
discriminator = layers.Conv2D(3, (4, 4), strides=(2, 2), padding="same",kernel_regularizer=regularizers.l1(0.01))(discriminator)
discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
#discriminator = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(discriminator)
#discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
#discriminator = layers.Dropout(0.3)(discriminator)
discriminator = layers.GlobalMaxPooling2D()(discriminator)
discriminator = layers.Dense(1)(discriminator)

modelD = Model(inputs=[true_image_input_], outputs=discriminator)

# Create the generator.

click_input = layers.Input(shape=(2,),name="click")
click_network = layers.Dense(1000,kernel_regularizer=regularizers.l1(0.1))(click_input)
click_network = layers.LeakyReLU(alpha=0.2)(click_network)
click_network = layers.BatchNormalization()(click_network)
#click_network = layers.Dense(500,kernel_regularizer=regularizers.l1(0.1))(click_input)
#click_network = layers.LeakyReLU(alpha=0.2)(click_network)
#click_network = layers.BatchNormalization()(click_network)
#click_network = layers.Dense(1000,kernel_regularizer=regularizers.l1(0.01))(click_input)
#click_network = layers.LeakyReLU(alpha=0.2)(click_network)
#click_network = layers.BatchNormalization()(click_network)


actual_im_input_ = layers.Input(shape=(256,),name="im_input")
actual_im_input = layers.Dense(1000,kernel_regularizer=regularizers.l1(0.1))(actual_im_input_)
actual_im_input = layers.LeakyReLU(alpha=0.2)(actual_im_input)
actual_im_input = layers.BatchNormalization()(actual_im_input)
actual_im_input = layers.Dense(1000,kernel_regularizer=regularizers.l1(0.1))(actual_im_input)
actual_im_input = layers.LeakyReLU(alpha=0.2)(actual_im_input)
actual_im_input = layers.BatchNormalization()(actual_im_input)
actual_im_input = layers.Dense(1000,kernel_regularizer=regularizers.l1(0.1))(actual_im_input)
actual_im_input = layers.LeakyReLU(alpha=0.2)(actual_im_input)
actual_im_input = layers.BatchNormalization()(actual_im_input)

generator = layers.concatenate([click_network,actual_im_input])
#generator = (layers.Dense(3000,kernel_regularizer=regularizers.l1(0.01)))(generator)
#generator = layers.LeakyReLU(alpha=0.2)(generator)
#generator = layers.BatchNormalization()(generator)
#generator = (layers.Dense(2000,kernel_regularizer=regularizers.l1(0.01)))(generator)
#generator = layers.LeakyReLU(alpha=0.2)(generator)
#generator = layers.BatchNormalization()(generator)
#generator = (layers.Dense(shape[1]*shape[0]*shape[2],kernel_regularizer=regularizers.l1(0.01)))(generator)
generator = (layers.Dense(500,kernel_regularizer=regularizers.l1(0.01)))(generator)
generator = layers.LeakyReLU(alpha=0.2)(generator)
generator = layers.BatchNormalization()(generator)
generator = (layers.Dense(shape[1]*shape[0]*shape[2],kernel_regularizer=regularizers.l1(0.01)))(generator)
generator = layers.LeakyReLU(alpha=0.2)(generator)
generator = layers.BatchNormalization()(generator)
generator = layers.Reshape((shape[1],shape[0],shape[2]))(generator)
generator = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(generator)
generator = layers.LeakyReLU(alpha=0.2)(generator)
generator = layers.BatchNormalization()(generator)
#generator = layers.Dropout(0.5)(generator)
generator = layers.Conv2DTranspose(3, (10,10), strides=(2,2), padding='same')(generator)

modelG = Model(inputs=[click_input,actual_im_input_], outputs=generator)

class ConditionalGAN(Model):
    def __init__(self, discriminator, generator):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = metrics.Mean(name="discriminator_loss")
        
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn, loss_equality):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.loss_equality = loss_equality
        
    def train_step(self, data):
        
        xclick, xim, y = data

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator({"click": xclick,
                   "im_input": xim})
        
        with tf.GradientTape() as tape:
            #predictions = self.discriminator({"false": generated_images,
             #          "true": y})
            predictions = self.discriminator(generated_images)
            d_loss = self.loss_fn(tf.ones((len(generated_images), 1)), predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        
        print("false images passeds like true :" + str(np.asarray(predictions < 0).sum()/len(predictions)))
        
        self.disc_loss_tracker.update_state(d_loss)
        print(self.disc_loss_tracker.result())
        
        train_set_discriminator = tf.concat(
            [y], axis=0
        )
        #train_double =  tf.concat(
        #    [y,y], axis=0
        #)
        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.zeros((len(generated_images), 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            #predictions = self.discriminator({"false": generated_images,
             #          "true": y})
            predictions = self.discriminator(train_set_discriminator)
            d_loss = self.loss_fn(labels, predictions)
            #print("acuraccy discriminator: " + str(((np.asarray(predictions-labels)==False).sum()/len(predictions))))
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        
        with tf.GradientTape() as tape:
            fake_images =  self.generator({"click": xclick,
                       "im_input": xim})
            #utiliza loss de igualdade para acelerar resultados
            loss_e= self.loss_equality(y,fake_images)
        grads = tape.gradient(loss_e, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((len(generated_images), 1))
        with tf.GradientTape() as tape:
            fake_images =  self.generator({"click": xclick,
                       "im_input": xim})
           
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        

    
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

cond_gan = ConditionalGAN(
    discriminator=modelD, generator=modelG
)
cond_gan.compile(
    d_optimizer=optimizers.Adam(learning_rate=0.0002),
    g_optimizer=optimizers.Adam(learning_rate=0.0002),
    loss_fn=losses.BinaryCrossentropy(from_logits=True),
    loss_equality=losses.MeanSquaredError(reduction='auto')
)



#salvar checkpoints

import os

cond_gan.generator.compile(optimizer='adam', loss='mean_squared_error', 
                           metrics=['accuracy'])
cond_gan.generator.fit({"im_input": xim/xim.max(),
           "click": xclick}, 
          y,
          epochs=10,
          batch_size = 50,
          workers = 3)

#cond_gan.discriminator.compile(optimizer='adam', loss='binary_crossentropy',metrics =['accuracy'])
#cond_gan.discriminator.fit(y,
#                           np.zeros(len(y)),
#                           epochs=50,
#                           batch_size = 10,
#                           workers = 3)

epochs = 3000
for i in range(0,epochs):
    
    print('\n----epoch ' + str(i+1) + ' of ' + str(epochs) + ' epochs-----')
    #for j in range(0,50):
    batch_size = 15
    array = range(0,len(xclick))
    sample = random.sample(array, batch_size)
    print(cond_gan.train_step([xclick[sample],xim[sample]/xim.max(),y[sample]]))
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "testesabado")
    checkpoint = tf.train.Checkpoint(generator=cond_gan.generator,
                                     discriminator=cond_gan.discriminator)
    
    if (i+1) % 100 == 0:
        count = 1
        for im in cond_gan.generator({"click": xclick[sample[1:5]],"im_input": xim[sample[1:5]]/xim.max()}):
            cv2.imwrite("im/" + str(count) + ".jpg", np.asarray(im).astype('uint8'))
            count = count + 1

    if (i+1) % 501 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
    
teste = cond_gan.generator({"click": xclick[range(450,470)],
                            "im_input": xim[range(450,470)]/xim.max()})

for i in range(0,len(teste)):
    cv2.imshow("imagempredicao", np.asarray(teste[i]).astype('uint8'))
    cv2.imwrite("im/" + str(i) + ".jpg", np.asarray(teste[i]).astype('uint8'))
    cv2.waitKey(0)

for i in range(40,80):
    cv2.imshow("imagempredicao", np.asarray(y[i]).astype('uint8'))
    cv2.waitKey(0)

#generated_images = modelG({"click": xclick,
#           "im_input": xim})

#discriminator_result =modelD({"false": generated_images,
#           "true": y})

#loss, acc = cond_gan.discriminator.evaluate(xim/xim.max(), np.ones(800), verbose=2)
