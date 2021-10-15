from numpy import *
x=array([[2,9,1],[9,3,1],[8,3,1],[8,8,1],[2,1,1],[8,4,1],[4,3,1],[1,8,1],[3,3,1],[5,3,1]])
y=array([290,1054,944,964,246,948,488,167,370,598]).reshape(-1,1)
e=identity(3)
tmp=dot(x.T,x)+2*e
res=dot(dot(linalg.inv(tmp),x.T),y)
print(res)