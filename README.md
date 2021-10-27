# Credit_Risk_Analysis

# Overview
Training  and evaluating  models with unbalanced classes in Credit risk Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily 
outnumber risky loans. RandomOverSampler, and SMOTE algorithms and undersampling 
the data using the Cluster Centroid algorithms combined with use a combinatorial 
approach of over- and undersampling using the SMOTEEN algorithms, we were able 
to predict credit risks in loan disbursement.

# Results:

•	 Naive Random Oversampling 
acc score = 66.7%
![image](https://user-images.githubusercontent.com/70987568/138615021-7682aa37-ecec-4397-851b-fb93dde911b9.png)

•	SMOTE Oversampling acc score =63.6%
![image](https://user-images.githubusercontent.com/70987568/138615058-da00343c-2e78-4a7f-b5db-9a6b7365374d.png)

•	Undersampling 
acc score =  52.2% 
![image](https://user-images.githubusercontent.com/70987568/138615123-01abc96e-f1ad-42f0-8750-14da31ede7d3.png)

•	Combination (Over and Under) Sampling 
acc score = 65.0%
![image](https://user-images.githubusercontent.com/70987568/138617248-cded994d-8c8c-4e1b-bf38-0c8a2472e49e.png)

•	Easy Ensemble AdaBoost Classifier acc score = 89.8%
![image](https://user-images.githubusercontent.com/70987568/138985862-6d2d6849-0826-4c5b-9df7-740e0be63f88.png)

•	Balanced Random Forest Classifier acc score = 89.8%

![image](https://user-images.githubusercontent.com/70987568/138985896-aff23ca1-144a-49f4-9c2b-fb43a5fb396b.png)


# Summary 
combined use of Balanced Random Forest Classifier acc score and Easy Ensemble AdaBoost Classifier acc score showed 89.8 % accuracy scores with predcited high risk/Actual low risk of 1730. 
Therefore it is recommended to use these prediction models for future analysis.
