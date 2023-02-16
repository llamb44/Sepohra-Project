# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:52:24 2021

@author: Sweet
"""

#Plot of average stars over loves
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.ensemble
import sklearn.feature_selection
import re
import csv

import pickle
from collections import Counter

cats= re.compile(r":")
def rev():
    plt.figure()
    bigTable.groupby("IsOunce")["Reviews"].mean().sort_index().plot.bar()
    plt.ylabel("Reviews")
    plt.title("Reviews Given Measurement")
    plt.savefig(r"D:\DataScience\Sephora-Project\sephora\Reviews_IsOunce.png", dpi=200)
    
def lBar():
    plt.figure()
    bigTable.groupby("Length of Names")["Loves"].mean().sort_index().plot.bar(figsize=(16,10))
    plt.ylabel("Loves")
    plt.title("Number of Loves Given Length of Names")
    plt.tight_layout()
    #plt.yticks(rotation=45, horizontalalignment="right")
    plt.savefig(r"D:\DataScience\Sephora-Project\sephora\Loves_LenNames2.png", dpi=200)
    
def Dplots():#plots the accuracy scores of the dependent variables over the number of bins
    plt.plot( "Bins","Reviews_Test", data= data, marker= "o", markerfacecolor= "darkslategray", color= "teal", linewidth= 4)
    plt.plot( "Bins","Loves_Test", data= data, marker= "o", markerfacecolor= "darkred", color= "red", linewidth= 4)
    plt.plot( "Bins","Stars_Test", data= data, marker= "o", markerfacecolor= "olive", color= "yellow", linewidth= 4)
    plt.legend()
    plt.xlabel("Number of Bins")
    #x1,x2,y1,y2= plt.axis()
    #plt.axis((x1,x2, .4, 1))
    plt.yscale("log")
    plt.ylabel("Accuracy Scores in Log Scale")
    plt.title("Accuracy Scores for Reviews, Loves, and Stars")
    #plt.show()
    plt.tight_layout()
    plt.savefig(r"D:\DataScience\Sephora-Project\sephora\Accuracy_Scores.png", dpi= 200)
    #data["Reviews_Importances"][1].split("\n") [0].split()[1]
    #THIS-1 is for the importances plot
    
def IplotsP():
    #price importances of the dependent values
    plt.scatter( data["Bins"],data["Stars_Price_I"], c=["yellow"])
    plt.scatter( data["Bins"],data["Love_Price_I"], c="red")
    plt.scatter( data["Bins"],data["Review_Price_I"], c="green")
    #plt.legend()
    plt.ylabel("Importance Values")
    plt.xlabel("Number of Bins")
    plt.title=("Importance of Price")
    plt.tight_layout()
    plt.savefig(r"D:\DataScience\Sephora-Project\sephora\Price_Importances.png", dpi= 200)
    #ax1= data.plot.scatter(x=["Stars_Price_I", "Love_Price_I"], y="Bins", c=["green", "red"])
    
def IplotsL():
    #price importances of the dependent values
    plt.figure()
    plt.scatter(data["Bins"],data["Stars_LengthName_I"],  c=["yellow"])
    plt.scatter(data["Bins"],data["Love_LengthName"],  c="red")
    plt.scatter(data["Bins"],data["Review_LengthName"],  c="green")
    #plt.legend()
    plt.yscale("log")
    plt.ylabel("Importance Values")
    plt.xlabel("Number of Bins")
    plt.title=("Importance of the Length of Names")
    plt.tight_layout()
    plt.savefig(r"D:\DataScience\Sephora-Project\sephora\LengthName_Importances.png", dpi= 200)

def p(rI):
    t=rI.split("\n")
    for i in t:
        if i[0]=="P":
            i=i.split()
            return i[1]
        
def l(rI):
    t=rI.split("\n")
    for i in t:
        if i[0]=="L":
            i= i.split()
            return i[3]

def split(s):
    if type(s)== float and np.isnan(s):
        return ""
    if type(s)== str:
        return cats.split(s)[-1]
    print("exception in split method\t"+ s)
    return 1

def lenChar(s):
    count=0
    for i in s:
        if i.isalnum():
            count+=1
    return count

def isOunce(l):
    for i in l:
        if "oz" in i:
            return True;
    return False

def writeBigCSV():
    pCatch= r"(\d+(?:\.\d+)?)\s*([a-wyzA-WYZ]+|[\"'])"
    spaces= re.compile(r"\s+[&\*\|#!\^%@\$()-_\+=,\.\?<>\~`{]\s*")
    cats= re.compile(r":")
    #lCatch= r"(\d+(?:\.\d+)*)\s*([a-wyz]+|[\"'])"
    unitCatch= re.compile(pCatch)
    Data= pd.read_csv("products.csv")
    sizes= Data[["size"]]
    Brand= Data[["brand"]]
    price= Data[["price"]]
    idNums= Data[["ppid"]]
    names= Data[["name"]]
    reviewNums= Data[["reviews"]]
    loves= Data[["love"]]
    stars= Data[["stars"]]
    category= Data["category"]
    isDict= Data[["discontinued"]]
    index= 0
    uList= list()
    #catList=[]
    #subList= list()
    
    # num chars= lenChar(names.vales[index])
    # num words= len(re.split("\s",s))
    # isOunce = isOunce(subList[WHATEVER VALUE HAS UNITS])
    #
    while index<len(sizes.values):
        if isDict.values[index][0]==False:
            uList.append([lenChar(names.values[index][0]), len(spaces.split(names.values[index][0])), price.values[index][0], isOunce(unitCatch.findall(str(sizes.values[index]))), reviewNums.values[index][0], loves.values[index][0], stars.values[index][0]])
            #catList.append(split(category.values[index]))
        #else:
            
             
        index+=1
        
        
    bDF= pd.DataFrame([i for i in uList], columns= ["Length of Names", "Number of Words in Names", "Price", "IsOunce","Reviews", "Loves", "Stars"])
    #catDF= pd.DataFrame([i for i in catList], columns= ["Categorys"])
    newCats=category.apply(split)
    categoryDummies= pd.get_dummies(newCats)
    categoryDummies= categoryDummies.iloc[:-1]
    #bDF.to_csv("bigTable2.csv", index= False)
    brandDummiesbDF=pd.get_dummies(Brand)
    brandDummiesbDF= brandDummiesbDF.iloc[:-1]
    #catsbDF= pd.get_dummies(category)
    #cats= (pd.get_dummies(catDF)).iloc[:-1]
    big= bDF.join(brandDummiesbDF)
    big= big.join(categoryDummies)
    big.to_csv("bigTable5.csv", index= False)
    #dummiesbDF.to_csv("dummiesTable.csv", index= False)
    
    
        
    
def writeUnitCSV():
    pCatch= r"(\d+(?:\.\d+)?)\s*([a-wyzA-WYZ]+|[\"'])"
    #lCatch= r"(\d+(?:\.\d+)*)\s*([a-wyz]+|[\"'])"
    unitCatch= re.compile(pCatch)
    
    Data= pd.read_csv("products.csv")
    sizes= Data[["size"]]
    idNums= Data[["ppid"]]
    names= Data[["name"]]
    index= 0
    uList= list()
    subList= list()
    while(index< len(sizes)):
#        try:
#            subList= [idNums.values[index], names.values[index], unitCatch.findall(str(sizes.values[index])), unitCatch.findall(str(sizes.values[index]))[1]]
#        except IndexError:
#            subList= [idNums.values[index], names.values[index], unitCatch.findall(str(sizes.values[index]))]
#        try:
#            uList.append([subList[0][0], subList[1][0], subList[2], subList[3]])
#        except IndexError:
#            uList.append([subList[0][0], subList[1][0], subList[2]])
       
        try:
            subList= [idNums.values[index], names.values[index], unitCatch.findall(str(sizes.values[index])), unitCatch.findall(str(sizes.values[index]))[1]]
            uList.append([subList[0][0], subList[1][0], subList[2][0], subList[3]])
        except IndexError:
            subList= [idNums.values[index], names.values[index], unitCatch.findall(str(sizes.values[index]))]
            
            #if str(sizes.values[index][0])=="nan":
            if unitCatch.findall(str(sizes.values[index]))== list():
                val= ""
            else:
                val= subList[2][0]
                
            uList.append([subList[0][0], subList[1][0], val])
            #print("index\t"+str(index))
            #break
        index+=1
    uDF= pd.DataFrame([i for i in uList], columns=["ID", "Name", "Size1", "Size2"])
    uDF.to_csv("unitsL2.csv", index= False)
    #Data.close()
    
def binning(b):
    bDF= pd.read_csv(r"D:\DataScience\Sephora-Project\sephora\bigTable3.csv")
    rev= pd.cut(bDF["Reviews"], b, labels= ["Low", "High"])
    star= pd.cut(bDF["Stars"], b, labels= ("Low", "High"))
    love= pd.cut(bDF["Loves"], b, labels= ("Low", "High"))
    
    bDF["Loves"]= love
    bDF["Reviews"]= rev
    bDF["Stars"]= star
    
    #bDF.to_csv("bigTable4-Bins.csv", index= False)
    seed= 42
    c= [i for i in bDF.columns]
    c.remove("Reviews")
    c.remove("Stars")
    c.remove("Loves")
    Xtrain, Xtest, Ytrain, Ytest= train_test_split(bDF[c], bDF[["Reviews", "Loves", "Stars"]],test_size= .3, random_state= 42, shuffle= True)
    
    
    training=Xtrain.join(Ytrain)
    testing=Xtest.join(Ytest)
    
    training.to_csv("trainingSet.csv", index= False)
    testing.to_csv("testingSet.csv", index= False)


#c= list(newData2.columns)
def bins(b):
    rev= pd.cut(oldData["Reviews"], b, labels= tuple(range(1,b+1)) )
    star= pd.cut(oldData["Stars"], b, labels= tuple(range(1,b+1)) )
    love= pd.cut(oldData["Loves"], b, labels= tuple(range(1,b+1)))
    
    newData2["Loves"]= love
    newData2["Reviews"]= rev
    newData2["Stars"]= star
    
    seed=42
    Xtrain, Xtest, Ytrain, Ytest= train_test_split(newData2[c], newData2[["Reviews", "Loves", "Stars"]],test_size= .3, random_state= seed, shuffle= True)
    training=Xtrain.join(Ytrain)
    testing=Xtest.join(Ytest)
    print("bins is:\t", b)
    validate(training, testing)    
    
    
def validate(training, testing):
    rfc=sklearn.ensemble.RandomForestClassifier()
    rfc.fit(training.loc[:, "Length of Names": "Value & Gift Sets"], training["Reviews"])
    print("Dependent:\tReviews")
    print("training score:\t", rfc.score(training.loc[:, "Length of Names": "Value & Gift Sets"], training["Reviews"]))
    print("testing score:\t", rfc.score(testing.loc[:, "Length of Names": "Value & Gift Sets"], testing["Reviews"]))
    print()
    importances=pd.Series(rfc.feature_importances_, index=c)
    print(importances.sort_values(ascending=False).head())
    print()
    print()
    
    print("Dependent:\tStars")
    rfc.fit(training.loc[:, "Length of Names": "Value & Gift Sets"], training["Stars"])
    print("training score:\t", rfc.score(training.loc[:, "Length of Names": "Value & Gift Sets"], training["Stars"]))
    print("testing score:\t", rfc.score(testing.loc[:, "Length of Names": "Value & Gift Sets"], testing["Stars"]))
    print()
    importances=pd.Series(rfc.feature_importances_, index=c)
    print(importances.sort_values(ascending=False).head())
    print()
    print()
    
    print("Dependent:\tLoves")
    rfc.fit(training.loc[:, "Length of Names": "Value & Gift Sets"], training["Loves"])
    print("training score:\t", rfc.score(training.loc[:, "Length of Names": "Value & Gift Sets"], training["Loves"]))
    print("testing score:\t", rfc.score(testing.loc[:, "Length of Names": "Value & Gift Sets"], testing["Loves"]))
    print()
    importances=pd.Series(rfc.feature_importances_, index=c)
    print(importances.sort_values(ascending=False).head())
    print()
    print()
    
    
def unitSize():
    #keeping here so I don't lose it
    #unitTypeCheck= re.compile(r"\d+(?:\.\d+)*\s*(\w+\b)*(?:\d+\.\d+)*(?:\s\d*(?:\.\d+)*(\w+\b)\W*(\w+\b)\W)*(/*)\s*\d*(?:\.\d+)*\s*(\w+\b)*",flags=0)
                ####pattern2= r"\d+(?:\.\d+)*\s*(\w+\b)*(?:\s\d*(?:\.\d+)*(\w+\b)\W*(\w+\b)\W)*(/*)\s*\d*(?:\.\d+)*\s*(\w+\b)*"
    #unitCatch= re.compile(r"(\d+\.*\d*)\s*([^x]\w+\b)*/*(?:\d*\.*\d*)\s*(?:\w+\b)*")
    pCatch= r"(\d+(?:\.\d+)?)\s*([a-wyzA-WYZ]+|[\"'])"
    lCatch= r"(\d+(?:\.\d+)*)\s*([a-wyz]+|[\"'])"
    unitCatch= re.compile(pCatch)
    #pCatch2= re.compile(pCatch)
    #field= "size"
    Data= pd.read_csv("products.csv")
    sizes= Data[["size"]]
    idNums= Data[["ppid"]]
    names= Data[["name"]]
    #prices= Data[["price"]]
    PPUdict= dict()
    index= 0
    while(index<28405):
        tempID= idNums.values[index]
        #tempPRICE= prices.values[i]
        tempSize= sizes.values[index]
        if str(tempSize[0])== "nan":
            PPUdict[tempID[0]]=""
            index+=1
            continue
        #if 'x' in tempSize[0] or 'X' in tempSize[0]:
        #    unit=1
        else:
            unit= unitCatch.findall(tempSize[0])
            
        PPUdict[tempID[0]]= unit
        index+=1
    return PPUdict
    
#    for i in data.values:
#        uList= unitTypeCheck.findall(str(i))
#        if len(uList)>0:
#            unitTypeCounter+= Counter("".join(uList[0]).split())
    
    #data=
    
def writeCSV(sDict):
    #words
    with open("unitsL.csv", "w") as outFile:
        writer= csv.writer(outFile, delimiter= ",", quotechar='"')
        writer.writerow(("PPID", "Unit"))
        for i in sDict:
            writer.writerow((i, sDict[i]))




def makePlot():
    data= pd.read_csv("products.csv")
    data[["stars", "love"]]
    #data[["stars", "love"]]
    myPlot= data.plot.scatter(x= "stars", y= "love")
    plt.tight_layout()

    plt.savefig(r"starvslove.png")
    
    
    #correlation
    stars= data.stars
    love= data.love
    
    cVal= stars.corr(love)
    data.close()
    
def main():
    print("start i guess")
    writeBigCSV()
    #binning()
    #writeUnitCSV()
    #uDict= unitSize()
    #writeCSV(uDict)
    
    
main()