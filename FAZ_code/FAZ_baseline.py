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

#Baseline
def make_dfs(data, label):
    cols = {

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
        precision = 0 
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

ADs_scp = []
with open('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/AD_base.pkl","rb") as f :
    ADs = pickle.load(f)
for i in ADs:
    ADs_scp.append('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/segmen_manual/AD/"+i)
SCDs_scp = []
with open('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/SCD_base.pkl","rb") as l :
    SCDs = pickle.load(l)
for i in SCDs:
    SCDs_scp.append('/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+"/Dataset/segmen_manual/SCD/"+i)

ADs_scp,folds_num1 = make_dfs(ADs_scp,"ADs_scp")
SCDs_scp,folds_num2 = make_dfs(SCDs_scp, "SCDs_scp")
fold_ADs = pd.concat([folds_num1,ADs_scp["label"]], axis=1).reset_index(drop=True)
fold_SCDs = pd.concat([folds_num2,SCDs_scp["label"]], axis=1).reset_index(drop=True)
folds = pd.concat([fold_ADs,fold_SCDs]).reset_index(drop=True)
folds_dup = folds.drop_duplicates().reset_index(drop=True)
dfs = pd.concat([ADs_scp,SCDs_scp]).reset_index(drop=True)
dfs.dropna(axis=1, inplace=True)
colss = dfs.columns.tolist()
colss.remove('label')

#Baseline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')


dfs = pd.concat([ADs_scp,SCDs_scp]).reset_index(drop=True)
le = LabelEncoder()
label = le.fit_transform(folds_dup['label'])
labels = le.fit_transform(folds["label"])
acc_sum = 0
score_sum = 0
acc_lst = []
auc_lst = []
y_lsts = []
pred_lsts = []
for i in range(5):
    score_sum = 0
    acc_sum = 0
    set_seed(i)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_num, test_num in kf.split(folds_dup["scheme_num"],label):
        train_idxs = folds_dup["scheme_num"].iloc[train_num]
        train_idxs = folds[folds["scheme_num"].isin(train_idxs)]
        train_idx = train_idxs.index
        test_idxs = folds_dup["scheme_num"].iloc[test_num]
        test_idxs = folds[folds["scheme_num"].isin(test_idxs)]
        test_idx = test_idxs.index
        X_train, X_test = dfs[colss].iloc[train_idx], dfs[colss].iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        lr_clf = LGBMClassifier(n_estimators=400, learning_rate=0.3, max_depth=3)
        lr_clf.fit(X_train,y_train)      
        pred = lr_clf.predict(X_test)
        preds = lr_clf.predict_proba(X_test)
        score = roc_auc_score(y_test, preds[:, 1])
        y_lsts.extend(y_test)
        pred_lsts.extend(preds)
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
