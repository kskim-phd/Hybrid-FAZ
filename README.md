# Hybrid-FAZ

We release Hybrid-FAZ Alzheimer's diagnosis code.

Contributors: Kyung-su Kim , Chae-yeon Lim

Detailed instructions for testing the image are as follows.

---

# Implementation

Pytorch implementation based on radiomics multi-feature based machine learning using OCTA FAZ segmentation image

Using Paper Dataset

/FAZ/Dataset/AI_base_inference (Proposed AI segmentation)
                      
/FAZ/Dataset/segmen_manual (Baseline manual segmentation)
                      
---

# Environments

Required libraries for training/inference are detailed in requirements.txt

```python
pip install -r requirements.txt
```

---

# Segmentation

Conducting training through nnUNet using a public data set

Public dataset: https://zenodo.org/record/5075563

AI-based segmentation result

![100](https://user-images.githubusercontent.com/86760506/206358072-0e18a8b6-ff58-410d-a988-1cd644c973f1.jpg)
![100](https://user-images.githubusercontent.com/86760506/206358135-4fd705e8-d544-45e4-9b39-cb21c91d0709.png)

![1000](https://user-images.githubusercontent.com/86760506/206358219-c84118c3-f482-4ab3-ac2c-da6f9c73bdab.jpg)
![1000](https://user-images.githubusercontent.com/86760506/206358236-d0eec3c7-2360-4248-bf25-751c3a39b270.png)

![1001](https://user-images.githubusercontent.com/86760506/206358298-4a813f4c-479f-4de9-b8ad-077fe0a0b064.jpg)
![1001](https://user-images.githubusercontent.com/86760506/206358302-23c04514-8c40-45f1-8ca7-b2a500b68e0c.png)

![1002](https://user-images.githubusercontent.com/86760506/206358340-2633dc29-e2a7-4edc-8d3a-5e7af1985ff9.jpg)
![1002](https://user-images.githubusercontent.com/86760506/206358342-f9fdba70-a7ab-4632-a122-01462911bab2.png)

---

# Multi-feature diagnosis

Radiomics multi-feature based machine learning diagnosis
```
Baseline code: python path/to/Hybrid-FAZ/FAZ_code/FAZ_baseline.py

Proposed code: python path/to/Hybrid-FAZ/FAZ_code/FAZ_proposed.py
```
---

# Diagnosis result

```markdown
|Model|Fold1|Fold2|Fold3|Fold4|Fold5|
|------|---|---|---|---|---|
|Baseline|51|64|49|60|68|
|Proposed|76|76|65|70|72|

머리1 | 머리2 | 머리3 | 뚝배기
---- | ---- | ---- | ----
다리 | 다리1 | 다리2 | 뚝배기깹니다
금 | 의 | 환 | 향

```
---

# Acknowledgements

thanks to MIC-DKFZ for sharing the segmentation nnUNet code.

nnUNet github: https://github.com/MIC-DKFZ/nnUNet
