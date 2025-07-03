import re
import tkinter as tk
import seaborn as sns
import numpy as np
import pandas as pd
from tkinter import messagebox
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# stop_words=set(stopwords.words("english"))
ps=PorterStemmer()
stop_words=["a","is","the","of","all","and","to","can","be","as","once","for","at","am","are","has","have","had","up","his","her","in","on","no","we","do"]


df=pd.read_excel("fake job posting.xlsx")

df_target=df["fraudulent"]
d_cols=["job_id","salary_range","telecommuting","has_company_logo","has_questions","fraudulent"]
for i in d_cols:
    df=df.drop(i,axis=1)

m_cols=["location","department","company_profile","description","requirements","benefits","employment_type","required_experience","required_education","industry","function"]

for i in m_cols:
    most_common=df[i].mode()[0]
    df[i]=df[i].fillna(most_common)

def clean_text(text):
    #Converting to lowercase
    text = str(text).lower()
    #Removing emails
    text = re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', text)
    #Removing URLs
    text = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', text)
    #Removing HTML tags
    # text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub(r'<.*?>', '', text)  # Removes anything between < and >
    #Removing punctuations and numbers
    text = re.sub('[^A-Z a-z ]+', ' ', text)
    #Removing Multiple spaces
    text =  " ".join(text.split())
    #Removing Stop words
    text =  " ".join([t for t in text.split() if t not in stop_words])
    #lemmatizing the text
    # text = lemmatize(text)

    # steeming the text
    text = [ps.stem(word) for word in text]
    # return text    
    return "".join(text)


df_p=df
# print(df_p.info())

text_cols=["title","location","department","company_profile","description","requirements","benefits","employment_type","required_experience","required_education","industry","function"]

for i in text_cols:
    df_p[i]=df_p[i].apply(clean_text)

# print(df_p.head(10))
tfidf=TfidfVectorizer(max_features=100)
df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
# print(df["combined_text"])
text_features = tfidf.fit_transform(df['combined_text'])

final_features=hstack([text_features])


df_train,df_test,dfy_train,dfy_test=train_test_split(final_features,df_target,test_size=0.2,random_state=42)

model_dt=DecisionTreeClassifier()
model_rf=RandomForestClassifier()
model_lr=LogisticRegression()
model_kn=KNeighborsClassifier()
model_nn=MLPClassifier(hidden_layer_sizes=(100,),max_iter=500,random_state=42)
model_sv=SVC()


#                                   Training all Models
model_dt.fit(df_train,dfy_train)
model_rf.fit(df_train,dfy_train)
model_lr.fit(df_train,dfy_train)
model_kn.fit(df_train,dfy_train)
model_nn.fit(df_train,dfy_train)
model_sv.fit(df_train,dfy_train)


#                   Testing of  (Desicion Tree Classifier)

#                                               Accuracy Score
acc_score_d=accuracy_score(dfy_test,model_dt.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_dt.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_dt.classes_,columns=model_dt.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_dt.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File


with open("Performance Result.md","w") as file:
    file.write("# Decision Tree Classifier Results\n\n")
    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_d*100,2)}%**\n\n")
    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")
    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")



#                   Testing of  (Random Forest)

#                                               Accuracy Score
acc_score_r=accuracy_score(dfy_test,model_rf.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_rf.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_rf.classes_,columns=model_rf.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_rf.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File

with open("Performance Result.md","a") as file:
    file.write("# Random Forests Results\n\n")
    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_r*100,2)}%**\n\n")
    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")
    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


#                   Testing of  (Logistic Regression)

#                                               Accuracy Score
acc_score_l=accuracy_score(dfy_test,model_lr.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_lr.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_lr.classes_,columns=model_lr.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_lr.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File


with open("Performance Result.md","a") as file:
    file.write("# Logistic Regression Results\n\n")
    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_l*100,2)}%**\n\n")
    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")
    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")



#                   Testing of  (SVC---Support Vector Machine)

#                                               Accuracy Score
acc_score_s=accuracy_score(dfy_test,model_sv.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_sv.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_sv.classes_,columns=model_sv.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_sv.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File


with open("Performance Result.md","a") as file:
    file.write("# Support Vector Classifier Results\n\n")
    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_s*100,2)}%**\n\n")
    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")
    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


#                   Testing of  (KNN)

#                                               Accuracy Score
acc_score_k=accuracy_score(dfy_test,model_kn.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_kn.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_kn.classes_,columns=model_kn.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_kn.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File

with open("Performance Result.md","a") as file:
    file.write("# K-Nearest Neighbor Results\n\n")
    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_k*100,2)}%**\n\n")
    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")
    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


#                   Testing of  (MLP Classifier ---Neural Networks)

#                                               Accuracy Score
acc_score_n=accuracy_score(dfy_test,model_nn.predict(df_test))

#                                               Confusion Matrix
cm=confusion_matrix(dfy_test,model_nn.predict(df_test))
cm_df=pd.DataFrame(cm,index=model_nn.classes_,columns=model_nn.classes_)

#                                               Classification Report
report=classification_report(dfy_test,model_nn.predict(df_test),output_dict=True)
report_df=pd.DataFrame(report).transpose()

#                                   Reporting Scores to File

with open("Performance Result.md","a") as file:
    file.write("# Neural Networks (MLP Classifier) Results\n\n")
    file.write("## Accuracy Score\n")
    file.write(f"**{round(acc_score_n*100,2)}%**\n\n")
    file.write("## Confusion Matrix\n")
    file.write("```\n")
    file.write(str(cm_df))
    file.write("\n```\n\n")
    file.write("## Classification Report\n")
    file.write("```\n")
    file.write(str(report_df.round(2)))
    file.write("\n```\n")


models=["Decisicon Tree","Random Forest","Logistic Regression","SVC","K-NN","Neural Networks(MLP)"]
test_acc=[acc_score_d,acc_score_r,acc_score_l,acc_score_s,acc_score_k,acc_score_n]


#                                       Accuracy comparasion on diff models

plt.figure(figsize=(8,6))
plt.plot(models,test_acc,marker='o',linestyle='-',color='teal',label='Test Accuracy')
plt.title('Model Accuracy Comparison on Test Set')
plt.ylabel('Accuracy')
plt.ylim(0.90,1.0)
plt.grid(True,linestyle='--',alpha=0.6)
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.show()