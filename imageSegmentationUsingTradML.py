import numpy as np
import os
import cv2
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import keras

img = cv2.imread('sandstone_data_for_ML/sandstone_all_462_images.tif')

#the dataexplorer shows that the image is a rgb image because of the tif file

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img) 

df = pd.DataFrame()

#FEATURE EXTRACTION

#pixel value feature add

#reshaping the original image into 1-D matrix
img2 = img.reshape(-1)
df['Original Image'] = img2

#print(df.head())

#Adding other features

#Add Gabor features of the image
num = 1
kernals = []
 

#we are going to get gabor filter for each of the values in the for loop
for theta in range(2):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):                #sigma value ranges
        for lamda in np.arange(0, np.pi, np.pi/4):             #lamda value ranges between np.pi and 0
            for gamma in (0.05, 0.5):           #gamma values ranges
                
                gabor_label = 'Gabor' + str(num)    #Gabor labels are printed as Gabor1, Gabor2, etc.
                ksize=5
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype = cv2.CV_32F)
                kernals.append(kernel)
                
                #now filter the image and add new features to the new column
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                print(gabor_label, ': theta=', theta, ': sigma= ', sigma, ': lamda= ',lamda, ': gamma= ', gamma)
                num += 1
                
#print(df.head())

#Canny Edge detector is an edge detector algorithm

edges = cv2.Canny(img, 100, 200)
edges1 = edges.reshape(-1)
plt.imshow(edges)
df['Canny Edge'] = edges1

from skimage.filters import roberts, sobel, scharr, prewitt

edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Edge Roberts'] = edge_roberts1
plt.imshow(edge_roberts)

edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Edge Sobel'] = edge_sobel1
plt.imshow(edge_sobel)

edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Edge Charr'] = edge_scharr1
plt.imshow(edge_scharr)

edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Edge Prewitt'] = edge_prewitt1
plt.imshow(edge_prewitt)

#Scharr, Sobel, Prewitt edge detectors are similar pictures

from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1
plt.imshow(gaussian_img)

gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3
plt.imshow(gaussian_img2)

median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1
plt.imshow(median_img)

variance_img = nd.generic_filter(img, np.var, size=3)
variance_img1 = variance_img.reshape(-1)
df['Variance s3'] = variance_img1
plt.imshow(variance_img)


labeled_img = cv2.imread('sandstone_data_for_ML/Sandstone_Versa0180_mask .tif')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGRA2GRAY)
plt.imshow(labeled_img)
labeled_img1 = labeled_img.reshape(-1)
#plt.imshow(labeled_img)
df['Labels'] = labeled_img1

#print(df.columns)
#print(df.head())


#Train the random forest classifier

#the dependent variable is the Labels column we made
y = df['Labels'].values
X = df.drop(labels = 'Labels', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 20)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state = 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


#Check the accuracy of the classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Feature Ranking
features_list = list(X.columns)

features_imp = pd.Series(model.feature_importances_, index = features_list).sort_values(ascending = False)
#print(features_imp)

#To save some computing Time, the best thing is to drop the columns that have 0 importance

#PICKELing the model

import pickle

filename = 'Sandstone_model'
pickle.dump(model, open(filename, 'wb'))

load_model = pickle.load(open(filename, 'rb'))

#Do a prediction
result = load_model.predict(X)

segmented = result.reshape((img.shape))

plt.imshow(segmented, cmap = 'jet')
plt.imsave('segmented_rock.jpg', segmented, cmap = 'jet')




















