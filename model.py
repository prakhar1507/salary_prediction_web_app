import pyforest
import pickle
data={"experience":[1,2,5,4,3,1,7,8,9],"test_score":[3,7,5,6,8,2,3,8,8],"interview_score":[9,2,4,5,3,1,7,8,5],"salary":[60000,30000,10000,40000,30000,60000,70000,20000,60000]}
dataset=pd.DataFrame(data)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
regressor=LinearRegression()
regressor.fit(X,y)
#saving model to disk
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
