import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
# %matplotlib inline

df=pd.read_csv('amazon-reviews.csv')

# feature vector
x=df.drop(['rating','date','name','verified','asin','helpfulVotes'], axis=1)

#target variable
y=pd.cut(df['rating'],bins=[0,1.66,3.33,5],labels=['Negative','Neutral','Positive'])

from textblob import TextBlob

titlePolarity=[]
titleSubjectivity=[]
bodyPolarity=[]
bodySubjectivity=[]

for i in range(len(x)):
    analiseTitle=TextBlob(str(x['title'][i]))
    analiseBody=TextBlob(str(x['body'][i]))

    titlePolarity.append(analiseTitle.sentiment.polarity)
    titleSubjectivity.append(analiseTitle.sentiment.subjectivity)

    bodyPolarity.append(analiseBody.sentiment.polarity)
    bodySubjectivity.append(analiseBody.sentiment.subjectivity)

x['titlePolarity'] = titlePolarity
x['titleSubjectivity'] = titleSubjectivity
x['bodyPolarity'] = bodyPolarity
x['bodySubjectivity'] = bodySubjectivity


# split data into training and testing sets
from sklearn.model_selection import train_test_split # dùng để đánh giá hiệu suất

X=x.drop(['title','body'],axis=1)

x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42) 
# tỉ lệ phần trăm phổ biến nhất với các gtrị được sd là 0.33
# random_state là siêu tham số được sd để thiết lập seed cho trình tạo ngẫu nghiên. Với random_state=0 -> nhận được cùng 1 tập data huấn luyện và thử nghiệm trên các lần thực thi khác nhau. Với random_state=42 -> nhận được cùng 1 tập data huấn luyện và thử nghiệm trên các lần thực thi khác nhau, nhưng lần này tập dữ liệu huấn luyện và thử nghiệm khác với trường hợp trước với random state = 0


import category_encoders as ce

# encode categorical variables with ordinal encoding
encoder=ce.OrdinalEncoder(cols=['titlePolarity','bodyPolarity','titleSubjectivity','bodySubjectivity'])
# sẽ gán các số nguyên cho các nhãn (label) theo thứ tự được quan sát trong dữ liệu.

x_train=encoder.fit_transform(x_train)
x_test=encoder.fit_transform(x_test)

# fit_transform được sd trên training data để có thể chia tỷ lệ training data và cũng tìm hiểu các tham số tỷ lệ của dữ liệu đó.

from sklearn.ensemble import RandomForestClassifier

#instantiate the classifier
rfc=RandomForestClassifier(n_estimators=1000,random_state=0)

# Random Forests là thuật toán học có giám sát (supervised learning), được sử dụng cho cả phân lớp và hồi quy. Nó cũng là thuật toán linh hoạt và dễ sử dụng nhất. Random forests tạo ra cây quyết định trên các mẫu dữ liệu được chọn ngẫu nhiên, được dự đoán từ mỗi cây và chọn giải pháp tốt nhất bằng cách bỏ phiếu. Nó cũng cung cấp một chỉ báo khá tốt về tầm quan trọng của tính năng. Random forests có nhiều ứng dụng, chẳng hạn như công cụ đề xuất, phân loại hình ảnh và lựa chọn tính năng. Nó có thể được sử dụng để phân loại các ứng viên cho vay trung thành, xác định hoạt động gian lận và dự đoán các bệnh. Nó nằm ở cơ sở của thuật toán Boruta, chọn các tính năng quan trọng trong tập dữ liệu.

# fit the model

#y_train contains the values of y(rating) that were turned from continuous to categorical
rfc.fit(x_train,y_train)

# Predict on the test set results
y_pred=rfc.predict(x_test)

from sklearn.metrics import accuracy_score

# Check accuracy score

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test,y_pred)))

feature_scores=pd.Series(rfc.feature_importances_,index=x_train.columns).sort_values(ascending=False)

#Creating a seaborn bar plot
sn.barplot(x=feature_scores,y=feature_scores.index)

#Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')

#Add title to the group
plt.title("Visualizing Important Features")

#Visualize the graph
# plt.show()

from sklearn.metrics import confusion_matrix
#Confusion Matrix ma trận nhầm lẫn hay ma trận lỗi -> cho phép hình dung hiệu suất của một thuật toán.

cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix\n\n',cm)

from sklearn.metrics import classification_report
#Tính toán độ chính xác của mô hình phân loại

print('\nClassification report\n')
print(classification_report(y_test,y_pred))