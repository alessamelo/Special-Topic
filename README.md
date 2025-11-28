
#  **Special-Topic — Anonymization Techniques for Medical Data**

This project explores how to safely preprocess and anonymize a medical dataset using Python notebooks.
Two parallel versions of the dataset were created:

1. **Clean Dataset (Classic Preprocessing)**
2. **Anonymized Dataset (Privacy-Aware Preprocessing)**

Both datasets are later compared through a classification model (MLP) to measure how anonymization impacts performance.
In adition to, a complete synthetized dataset was generated using the Anonymized dataset as reference
---

#  **Project Files**

```
.
├── Anonimo.ipynb     # Notebook implementing all anonymization techniques
├── Clean.ipynb       # Classic preprocessing without any privacy methods
├── EDA.ipynb         # Exploratory Data Analysis: distributions, imbalance, correlations
├── Training.ipynb    # MLP model training, SMOTE balancing, metrics evaluation
└── Synthetized       #To generate the synthetized data
```


###  **1. Project Goal**

> **Safely process and analyze a sensitive medical dataset**
> while comparing:
>
> * A **non-anonymized** version
> * A **fully anonymized** version
>
> using the **same model** to observe how privacy transformations affect performance.

---

###  **2. Notebooks Overview**

### **✔ EDA.ipynb**

* Record counts
* Distribution plots
* Rare medical conditions
* Outliers in biometric fields
* Correlation matrix
* Identification of sensitive attributes

---

### **✔ Clean.ipynb** (Baseline Dataset)

Classic preparation:

* Dropping irrelevant features
* Encoding categorical variables
* Scaling numeric attributes
* No anonymization is applied

This dataset represents the **traditional ML workflow**.

---

### **✔ Anonimo.ipynb** (Anonymized Dataset)

Implements all privacy techniques required by your rubric:

#### **1. Suppression**

* Removes `PatientID` (direct identifier)

#### **2. Generalization**

* State → Region
* AgeCategory → AgeGroup
* HeightInMeters → HeightGroup
* BMI → BMIGroup
* Diabetes status → DiabetesGroup

#### **3. Perturbation**

Adds Gaussian noise to weight:

```python
Weight + N(0, 0.8)
```

#### **4. Encoding**

* One-hot: Health, DiabetesGroup, BMIGroup, Region, etc.
* Binary: Illnesses, disabilities
* Sex: Female=0, Male=1

This produces a **privacy-safe dataset** ready for modeling.

---

### **✔ Training.ipynb**

Contains the full ML workflow:

#### **MLP Architecture**

* 256 → 128 → 64 units
* BatchNorm
* Dropout
* ReLU activation
* Final sigmoid via BCEWithLogitsLoss

#### **Training Features**

* Early stopping
* Learning rate scheduler
* Adam optimizer
* Torch DataLoader

#### **SMOTE**

Used to fix extreme imbalance in:

* Heart attacks

#### **Evaluation**

Reports:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Performance is compared for both datasets.
