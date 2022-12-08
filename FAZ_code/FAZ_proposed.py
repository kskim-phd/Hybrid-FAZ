from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import *
from skimage.filters import *
from skimage.feature import *
from skimage.transform import *
from skimage.morphology import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import *
import skimage
from skimage.util import *
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from util import *
import opsfaz as faz
import random
import os
import cv2
from tqdm import tqdm
import natsort
import glob
import pdb
import pickle

# seed setting
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(17)

def roundness(img):
    regions = regionprops(img.astype(int))
    if len(regions) != 1:
        raise('There are one more contours!')
    convex = convex_hull_image(img)
    convex_perimeter = perimeter(convex)
    
    return (4 * math.pi * regions[0].area) / convex_perimeter ** 2

def solidity(img):
    regions = regionprops(img.astype(int))
    if len(regions) != 1:
        raise('There are one more contours!')
    convex = convex_hull_image(img)
    convex_regions = regionprops(convex.astype(int))
    if len(convex_regions) != 1:
        raise('There are one more contours!')
    
    return regions[0].area / convex_regions[0].area

def eccentricity(img):
    regions = regionprops(img.astype(int))
    if len(regions) != 1:
        raise('There are one more contours!')
    
    return regions[0].minor_axis_length / regions[0].major_axis_length

def compactness(img):
    regions = regionprops(img.astype(int))
    if len(regions) != 1:
        raise('There are one more contours!')

    return (4 * math.pi * regions[0].area)/(perimeter(img))**2

# Propsoed
# Data Frame
def make_df(data, label):
    cols = {
    'roundness': [],
    'solidity': [],
    'eccentricity': [],
    'compactness': [],
    }
    area_lst = []
    age_lst = []
    gender_lst = []
    scheme_lst = []
    label_lst = []

    for img_path in tqdm(data):
        data = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        data = data/255
        x, y = data.shape 
        size = data.shape
        
        mm = 3
        deep = 0
        precision = 0 # 
        imOCT = np.zeros((size[0],size[1]),np.float64)
        contours,_ = cv2.findContours(data.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cogidos = []
        cnt,cogidos = higest_contour (contours, cogidos)
        m = cv2.contourArea(cnt)
        fazAreainMM = m*(mm*mm)/(size[0]*size[1])

        im = np.zeros((size[0],size[1]), np.uint8)
        cv2.drawContours(im, [cnt], 0, (255,255,255), -1)
        im = im[:]/255
        
        # make region growing
        
        reg = region_growing(imOCT, im*1.0, fazAreainMM, 0, 4, precision)

        reg = morph ('open', reg, 3)
        reg = morph ('closed', reg, 3)
        image1 = cv2.convertScaleAbs(reg) 
        contours, h = cv2.findContours(image1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cogidos = []
        cnt,cogidos = higest_contour (contours, cogidos)

        m = cv2.contourArea(cnt)
        fazAreainMM = m*(mm*mm)/(size[0]*size[1])
        area = fazAreainMM
        for col in cols:
            cols[col].append(globals()[col](data))
        num = int(str(img_path).split("/")[-1].split(" ")[0].split("_")[-1])
        Later = str(img_path).split("/")[-1].split(".")[0].split(" ")[-1]
        df = pd.read_excel('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Alz_clinical.xlsx")
        df[df["Research number"]==num]
        right = df[df.columns[:15].tolist()]
        left = df[df.columns[15:].tolist()]
        right = right.fillna(right.shift(1))
        df = pd.concat([right,left],axis=1)
        area_lst.append(area)
        scheme_lst.append(int(num))
        age_lst.append(int(df[df["Research number"]==num]["Age"].tolist()[0]))
        gender_lst.append(int(df[df["Research number"]==num]["Sex"].tolist()[0]))
        z = df[df["Research number"]==num]
        label_lst.append(label)
        
    df = pd.DataFrame(cols)
    fold = pd.DataFrame(scheme_lst)
    df['label'] = label_lst
    df['area'] = area_lst
    df["age"] = age_lst
    df["gender"] = gender_lst
    fold["scheme_num"] = scheme_lst

    return df,fold

AD_scp = []
with open('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/AD_pro.pkl","rb") as f :
    AD = pickle.load(f)
for i in AD:
    AD_scp.append('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/AI_base_inference/AD/"+i)
SCD_scp = []
with open('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/SCD_pro.pkl","rb") as l :
    SCD = pickle.load(l)
for i in SCD:
    SCD_scp.append('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/AI_base_inference/SCD/"+i)

AD_scp,fold_num1 = make_df(AD_scp,"AD_scp")
SCD_scp,fold_num2 = make_df(SCD_scp, "SCD_scp")
fold_AD = pd.concat([fold_num1,AD_scp["label"]], axis=1).reset_index(drop=True)
fold_SCD = pd.concat([fold_num2,SCD_scp["label"]], axis=1).reset_index(drop=True)
fold = pd.concat([fold_AD,fold_SCD]).reset_index(drop=True)
fold_dup = fold.drop_duplicates().reset_index(drop=True)
df = pd.concat([AD_scp,SCD_scp]).reset_index(drop=True)
df.dropna(axis=1, inplace=True)
cols = df.columns.tolist()
cols.remove('label')

#Proposed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

df = pd.concat([AD_scp,SCD_scp]).reset_index(drop=True)
le = LabelEncoder()
label = le.fit_transform(fold_dup['label'])
labels = le.fit_transform(fold["label"])
acc_sum = 0
score_sum = 0
acc_lst = []
auc_lst = []
y_lst = []
pred_lst = []
for i in range(5):
    score_sum = 0
    acc_sum = 0
    set_seed(i)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_num, test_num in kf.split(fold_dup["scheme_num"],label):
        # print(test_num)
        train_idxs = fold_dup["scheme_num"].iloc[train_num]
        train_idxs = fold[fold["scheme_num"].isin(train_idxs)]
        train_idx = train_idxs.index
        test_idxs = fold_dup["scheme_num"].iloc[test_num]
        test_idxs = fold[fold["scheme_num"].isin(test_idxs)]
        test_idx = test_idxs.index
        X_train, X_test = df[cols].iloc[train_idx], df[cols].iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        lr_clf = LGBMClassifier(n_estimators=400, learning_rate=0.3, max_depth=3)
        lr_clf.fit(X_train,y_train)       
        pred = lr_clf.predict(X_test)
        preds = lr_clf.predict_proba(X_test)
        score = roc_auc_score(y_test, preds[:, 1])
        y_lst.extend(y_test)
        pred_lst.extend(preds)
        auc_lst.append(float(score))

fold1 = auc_lst[0] + auc_lst[0+5] + auc_lst[0+10] + auc_lst[0+15] + auc_lst[0+20]
fold2 = auc_lst[1] + auc_lst[0+6] + auc_lst[0+11] + auc_lst[0+16] + auc_lst[0+21]
fold3 = auc_lst[2] + auc_lst[0+7] + auc_lst[0+12] + auc_lst[0+17] + auc_lst[0+22]
fold4 = auc_lst[3] + auc_lst[0+8] + auc_lst[0+13] + auc_lst[0+18] + auc_lst[0+23]
fold5 = auc_lst[4] + auc_lst[0+9] + auc_lst[0+14] + auc_lst[0+19] + auc_lst[0+24]

print(f'fold1:{fold1/5}')
print(f'fold2:{fold2/5}')
print(f'fold3:{fold3/5}')
print(f'fold4:{fold4/5}')
print(f'fold5:{fold5/5}')
print(f'mean: {np.array(auc_lst).sum()/25}')
print(f'AUC std: {np.std((fold1/5,fold2/5,fold3/5,fold4/5,fold5/5))}')
