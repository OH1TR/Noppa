
import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

PATH = os.getcwd()

#Save file
model_file='myModel.h5'

img_data_list=[]
labels=[]


data_path = os.path.join(PATH ,'..','images','noppa','jpg')
data_dir_list = os.listdir(data_path)

for dataset in data_dir_list:
	img_list=os.listdir(os.path.join(data_path,dataset))
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = os.path.join(data_path,dataset,img)
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
		#print('Input image shape:', x.shape)
		img_data_list.append(x)
		labels.append(int(dataset)-1)

img_data = np.array(img_data_list)
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 6
num_of_samples = img_data.shape[0]

names = ['1','2','3','4','5','6']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#Training the feature extraction also

if os.path.isfile(model_file):
    custom_vgg_model2=load_model(model_file)
else:
    image_input = Input(shape=(224, 224, 3))
    
    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
    
    model.summary()
    
    last_layer = model.get_layer('block5_pool').output
    x= Flatten(name='flatten')(last_layer)
    x = Dropout(0.6)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.6)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    custom_vgg_model2 = Model(image_input, out)
    custom_vgg_model2.summary()
    
    # freeze all the layers except the dense layers
    for layer in custom_vgg_model2.layers[:-5]:
    	layer.trainable = False
    
    custom_vgg_model2.summary()
    

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=1200, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
#hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))

custom_vgg_model2.save(model_file)

(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#%%
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
