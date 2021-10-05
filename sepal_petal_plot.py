import sepal_knn,petal_knn,sepal_petal_knn,sepal_petal_nbs,sepal_nbs,petal_nbs
import sepal_knnpredict,petal_knnpredict,sepal_petal_knnpredict
import sepal_nbspredict,petal_nbspredict,sepal_petal_nbspredict
import sepal_dt,petal_dt,sepal_petal_dt
import sepal_dtpredict,petal_dtpredict,sepal_petal_dtpredict
import sepal_lg,petal_lg,sepal_petal_lg

knns=sepal_knn.knnS(0.7)
knnp=petal_knn.knnP(0.7)
knnsp=sepal_petal_knn.knnSP(0.7)
nbsp=sepal_petal_nbs.NBSP(0.7)
nbs=sepal_nbs.NBS(0.7)
nbp=petal_nbs.NBP(0.7)
dts=sepal_dt.sdt(0.7)
dtp=petal_dt.pdt(0.7)
dtsp=sepal_petal_dt.spdt(0.7)
lgs=sepal_lg.slg(0.7)
lgp=petal_lg.plg(0.7)
lgsp=sepal_petal_lg.splg(0.7)


import matplotlib.pyplot as plt
import numpy as np

def sp_bar_graph():
    algorithms=('KNN','Naive Bayes','Decision Tree','Logical Regression')
    x_pos=np.arange(len(algorithms))
    accuracy=[knnsp,nbsp,dtsp,lgsp]
    plt.bar(x_pos,accuracy,align='center')
    plt.xticks(x_pos,algorithms)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Algorithms with Sepal & Petal')
    plt.show()
    
def s_bar_graph():
    algorithms=('KNN','Naive Bayes','Decision Tree','Logical Regression')
    x_pos=np.arange(len(algorithms))
    accuracy=[knns,nbs,dts,lgs]
    plt.bar(x_pos,accuracy,align='center')
    plt.xticks(x_pos,algorithms)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Algorithms with Sepal')
    plt.show()
    
def p_bar_graph():
    algorithms=('KNN','Naive Bayes','Decision Tree','Logical Regression')
    x_pos=np.arange(len(algorithms))
    accuracy=[knnp,nbp,dtp,lgp]
    plt.bar(x_pos,accuracy,align='center')
    plt.xticks(x_pos,algorithms)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Algorithms with Petal')
    plt.show()    

def sp_pie_chart():
    labels=['Knn','Naive Bayes','Decision Tree','Logical Regression']
    sizes=[knnsp,nbsp,dtsp,lgsp]
    colors=['gold','yellowgreen','lightcoral','lightskyblue']
    explode=(0.1,0,0,0)
    plt.pie(sizes,explode=explode,labels=labels,
            colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()

def s_pie_chart():
    labels=['Knn','Naive Bayes','Decision Tree','Logical Regression']
    sizes=[knns,nbs,dts,lgs]
    colors=['gold','yellowgreen','lightcoral','lightskyblue']
    explode=(0.1,0,0,0)
    plt.pie(sizes,explode=explode,labels=labels,
            colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()    

def p_pie_chart():
    labels=['Knn','Naive Bayes','Decision Tree','Logical Regression']
    sizes=[knnp,nbp,dtp,lgp]
    colors=['gold','yellowgreen','lightcoral','lightskyblue']
    explode=(0.1,0,0,0)
    plt.pie(sizes,explode=explode,labels=labels,
            colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()    

if __name__=="__main__":        
    s_bar_graph()
    p_bar_graph()
    sp_bar_graph()
    s_pie_chart()
    p_pie_chart()
    sp_pie_chart()
    s_pie_chart()
