import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("train_score.csv")

def loss_function(m,b,points):
    total_error=0

    for i in range(len(points)):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        total_error+=(y-(m*x+b))**2
    total_error = total_error / float(len(points))
    return total_error

def gradient_descent(m_now,b_now,points,L):
    m_gradient=0
    b_gradient=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        m_gradient += ((-2/n)*x*(y-(m_now*x+b_now)))
        b_gradient += ((-2/n)*(y-(m_now*x+b_now)))
    m = m_now-m_gradient*L
    b = b_now-b_gradient*L
    return m, b
def predict(x):
    y_predicted=m*x+b
    return y_predicted


m=0
b=0
L=0.00001
epochs=4000
for i in range(epochs):
    if i%50==0:
        print(f"epochs: {i}")
    m,b=gradient_descent(m,b,df,L)

print("Error=",loss_function( m,b,df))
print(predict(2.5))
plt.scatter(df.Hours,df.Scores)
X=df.Hours
Y=m*X+b
plt.plot(X,Y,color="black")
plt.show()
