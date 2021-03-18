'''
Machine Learning Algorithm

by Calem Bardy

editted: 2021-03-17
'''

from matplotlib import pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from zipfile import ZipFile
import sys
import cv2
import os
import numpy as np

# Key parameters
resize_dims = (64, 64) # Target resizing dimensions of images
train_folders = "./data/train/"
test_folders = "./data/test/"
n_bins = 16 # For HHist
model_type = ['svm', 'nb', 'lreg'] # Choose model type(s). Default: ['svm', 'nb', 'lreg']
'''
svm: Support Vector Machine
nb: Naive Bayes
lreg: Logistical Regression
'''
feature_type = ['HOG', 'HHist', 'raw'] # Choose feature type(s). Default: ['HOG', 'HHist', 'raw']
'''
HOG: Histogram of Oriented Gradients
HHist: Hue Histogram
raw: Flattened Image
'''
show_confusion_matrix = True
show_evaluation = False # Use with all 9 methods

def read_data(address, resize_dim = (64,64)):
    # Grab folders & sub-folder addresses
    folders = [] # Folder addresses
    subfolderNames = [] # Sub-folder names
    n_classes = 0 # Number of classes

    for root, dirs, files in os.walk(address):
      if not root == address:
        folders.append(root)
        subfolderNames.append(root[len(address):])
        n_classes += 1

    data = []
    labels = []
    classes_labels = {}

    for i in range(n_classes): # for each subfolder,
        classes_labels['{}'.format(i)] = subfolderNames[i]
        for root, dirs, files in os.walk(folders[i]): # For each image in a sub-folder
            for image in files:
              labels.append(i)
              path = folders[i] + '/' + image
              img = cv2.imread(path, cv2.IMREAD_COLOR)
              resized_img = cv2.resize(img, resize_dim)
              final_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
              data.append(final_img)

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, classes_labels
def randomize_and_display(data, labels, classes_labels, display_n=5, phase=None, display=True):

    if phase is not None:
      print("\nShuffling {} data".format(phase))
    data_shuff, labels_shuff = shuffle(data, labels)

    if display:
      plt.figure(figsize=(15,5)) #TO-DO # create the figure element
      for i in range(display_n):
        plt.subplot(1, display_n, i+1)
        plt.title(classes_labels[str(labels_shuff[i])])
        plt.axis('off')
        plt.imshow(data_shuff[i])

    return data_shuff, labels_shuff
def my_hog_batch(data, phase = None):

  template = hog(data[0], orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=False)
  hog_size = template.shape[0]

  if phase is not None:
      print("Extracting HOG features from the {} dataset...".format(phase))
  else:
      print("Extracting HOG features...")

  hog_features = np.zeros((data.shape[0],hog_size))

  for i in range (data.shape[0]):
      current = cv2.cvtColor(data[i], cv2.COLOR_RGB2GRAY) # TO-DO
      new_hog = hog(data[i], orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=False)
      hog_features[i] = new_hog

  return hog_features
def my_hhist_batch(input, n_bins, phase = None):

  if phase is not None:
      print("Extracting h histogram features from the {} dataset...".format(phase))
  else:
      print("Extracting h histogram...")

  hsv = np.zeros(input.shape)
  h_hist = np.zeros((input.shape[0],n_bins))

  for i in range(input.shape[0]):
      current = cv2.cvtColor(input[i], cv2.COLOR_RGB2HSV) #TO-DO
      h_hist[i] = np.histogram(current[0], n_bins)[0]/sum(np.histogram(current[0], n_bins)[0])

  return h_hist
def my_raw_batch(data):

    raw = np.zeros((data.shape[0], data[0].flatten().shape[0]))
    for i in range(data.shape[0]):
        raw[i] = data[i].flatten()

    return raw
def my_model_trainer(features, labels, model='nb'):

  if model=='nb':
    print('Creating a Gaussian Naive Bayes Model...')
    model = GaussianNB()
  elif model == 'lreg':
    print('Creating a Logistic Regression Model...')
    model = LogisticRegression()
  elif model == 'svm':
    print('Creating a Support Vector Machine Model...')
    model = SVC(gamma='auto', kernel='linear')
  else:
    print('Please choose a valid model type!')

  print('Finished training the model.')
  model.fit(features, labels)

  return model
def my_evaluation(features, labels, model, model_type):

    results = {
        'accuracy': 0,
        'recall': 0,
        'precision': 0,
        'avg_recall': 0,
        'avg_precision': 0,
        'fscore': 0
    }

    pred = model.predict(features) # TO-DO

    cm_raw = confusion_matrix(labels, pred)

    TP, FP, FN = np.zeros(len(cm_raw)), np.zeros(len(cm_raw)), np.zeros(len(cm_raw))
    for i in range(len(cm_raw)):
      TP[i] = cm_raw[i][i]
      FP[i] = sum(cm_raw[:,i]) - cm_raw[i,i]
      FN[i] = sum(cm_raw[i,:]) - cm_raw[i,i]

    recall = np.zeros(len(cm_raw))
    for i in range(len(TP)):
      recall[i] = TP[i]/(TP[i]+FN[i] + 1e-50)
    avg_recall = sum(recall)/len(recall)
    results['recall'] = recall
    results['avg_recall'] = avg_recall

    precision = np.zeros(len(cm_raw))
    for i in range(len(TP)):
      precision[i] = TP[i]/(TP[i] + FP[i] + 1e-50)
    avg_precision = sum(precision)/len(precision)

    results['precision'] = precision
    results['avg_precision'] = avg_precision
    results['fscore'] = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    results["accuracy"] = np.mean(pred == labels)

    return results

data_downloaded = False
for root, dirs, files in os.walk("."):
  for filename in files:
    if filename == 'data.zip':
      data_downloaded = True
if not data_downloaded:
  print('Please upload data.zip')

unzipped = False
for root, dirs, files in os.walk("."):
  for dir in dirs:
    if dir == 'data':
      unzipped = True
if not unzipped:
  #!unzip './data.zip'
  with ZipFile('data.zip', 'r') as zip_object:
      zip_object.extractall()
      print('Extracted data into \'data\' folder')

# Read data
train_data, train_labels, classes_labels = read_data(train_folders, resize_dims)
test_data, test_labels, classes_labels_test = read_data(test_folders, resize_dims)

# Randomize data
train_data, train_labels = randomize_and_display(train_data, train_labels, classes_labels, display=False)
test_data, test_labels = randomize_and_display(test_data, test_labels, classes_labels_test, display=False)

assert classes_labels==classes_labels_test, 'Your train/test folder names do not match' # Verify training & testing datasets match

# Collect features
features = {}
features['HOG_train'] = my_hog_batch(train_data)
features['HOG_test'] = my_hog_batch(test_data)
features['HHist_train'] = my_hhist_batch(train_data, n_bins)
features['HHist_test'] = my_hhist_batch(test_data, n_bins)
features['raw_train'] = my_raw_batch(train_data)
features['raw_test'] = my_raw_batch(test_data)

# Train Models
models = {}
for curr_model in model_type:
  for curr_feature in feature_type:
    models[curr_model + '_' + curr_feature] = my_model_trainer(features[curr_feature + '_train'], train_labels, curr_model)

if show_confusion_matrix:
    # Plot confusion matrix
    confusion_matrices = {}
    display_labels = classes_labels
    for m in model_type:
        for f in feature_type:
            cm = confusion_matrices[m + '_' + f] = plot_confusion_matrix(models[m + '_' + f], features[f + '_test'], test_labels, display_labels=display_labels)
            cm.ax_.set_title(m + '_' + f)
    plt.show()

if show_evaluation:
    # Show evaluation summary
    data = np.zeros((len(model_type) * len(feature_type),3)) # only three metrics and nine possible models
    results = {}
    idx = 0
    for m in model_type:
      for f in feature_type:
        results = my_evaluation(features[f+'_test'], test_labels, models[m + '_' + f], model_type=m)
        data[idx, 0] = results['avg_recall']
        data[idx, 1] = results['avg_precision']
        data[idx, 2] = results['fscore']
        idx += 1

    row_labels = ['SVM_HOG','SVM_HHist','SVM_Raw','NB_HOG','NB_HHist','NB_Raw','LogR_HOG','LogR_HHist','LogR_Raw']
    col_labels = ["AvgR", "AvgP", "F1"]
    table_data = np.zeros((9,3))
    for i in range(9):
      table_data[i] = data[i,0], data[i,1], data[i,2]
    results_table = plt.table(cellText = table_data, rowLabels = row_labels, colLabels = col_labels, loc='center')
    plt.subplots_adjust(right=1, bottom=0)
    plt.box(on=None)
    plt.axis("off")
    plt.show()
