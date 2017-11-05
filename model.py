import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten,Lambda,Dense,Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


def load_input():
    ''' read the input csv file and output a list of the data'''
    lines = []
    with open ('./data/driving_log.csv') as csvfile:
       #skip the header
        next(csvfile) 
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return (lines)

def augment_brightness(img):
    ''' input is the image and output is the image 
     with the brightness augmented '''
    im = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    brightness = 0.25 + np.random.uniform()
    im[:,:,2] = im[:,:,2]*brightness
    im = cv2.cvtColor(im,cv2.COLOR_HSV2RGB)
    return im

def generator(samples,alpha,batch_size=32):
    ''' generates images on the fly'''
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,1,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                c_img = './data/IMG/'+batch_sample[0].split('/')[-1]
                l_img = './data/IMG/'+batch_sample[1].split('/')[-1]
                r_img = './data/IMG/'+batch_sample[2].split('/')[-1]
                center_img = cv2.imread(c_img)
                center_image = cv2.cvtColor(center_img,cv2.COLOR_BGR2RGB)
                left_img = cv2.imread(l_img)
                left_image = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB)
                right_img = cv2.imread(r_img)
                right_image = cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB)            
                measurement = float(batch_sample[3])
                
                camera = np.random.choice(['center','left','right'])
                if camera == "left":
                    measurement = measurement + alpha
                    image = left_image                    
                elif camera == "right":
                    measurement = measurement - alpha
                    image = right_image
                else:
                    image = center_image
                
                #augment brightness
                image = augment_brightness(image)
                
                #randomly flip the image
                flip_prob = np.random.random()
                if flip_prob > 0.5:
                    images.append(cv2.flip(image,1))
                    measurements.append(-1*measurement)
                else:
                    images.append(image)
                    measurements.append(measurement)                
                
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train,y_train)



def scale_img(img):
    '''input is the image (160x320x3) and
    output is resized image(64x64x3)'''
    from keras.backend import tf as ktf
    return ktf.image.resize_images(img,(64,64))


#define the model
ch,row,col = 3,160,320  # image shape
batch_size=32

samples = load_input()

train_samples,validation_samples = train_test_split(samples,test_size=0.2)

train_generator = generator(train_samples,0.20,batch_size)
validation_generator = generator(validation_samples,0.20,batch_size)

# model
model = Sequential()
#normalize
model.add(Lambda (lambda x:x/255.0 - 0.5,input_shape=(row,col,ch)))
#crop the image
model.add(Cropping2D(cropping = ((70,25),(1,1)),input_shape=(160,320,3)))
#scale the image to 64x64 
model.add(Lambda(scale_img))

#layer1
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
#layer2
model.add(Convolution2D(32,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
#layer3
model.add(Dense(120))
#layer4
model.add(Dense(84))
#layer5
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, \
                                    samples_per_epoch= (50000//batch_size)*batch_size ,\
                                    validation_data= \
                                    validation_generator,nb_val_samples= \
                                    (10000//batch_size)*batch_size,nb_epoch=2, \
                                    verbose=1)
model.save('model.h5')
print ('loss')
print (history_object.history['loss'])
print ('validation loss')
print (history_object.history['val_loss'])
#plot training and validation loss
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model MSE loss')
#plt.xlabel('epoch')
#plt.legend(['training set','validation set'],loc='upper right')
#plt.show()


           
          

