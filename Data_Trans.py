'''
中文文本情感分析
1.分词
2.朴素贝叶斯
'''

import jieba
import jieba.analyse
import re
# 对训练预料进行切词,返回切词结果和类别
from numpy import zeros, ones, log, array


# 获取评论数据,并切词,返回切词结果和评论分类
def loadDataSet(file_path):
    postingList = []
    classVec = []
    file = open(file_path, "r")
    rowCount = 0
    for line in file.readlines():
        seg_list = jieba.analyse.extract_tags(line.strip().split("##*##")[3], topK=10)
        postingList.append(list(seg_list))
        classify = line.split("##*##")[2]
        if classify == "5":
            classify = 1
        else:
            classify = 0;
        classVec.append(classify)
        # print("分词 %d 结果: " % rowCount, "/".join(seg_list),"    类别:",classify)  # 精确模式
        rowCount = rowCount + 1
    return postingList, classVec


# 构建词典
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 从文件中获取人工干预后的词典
def loadVocabList(file_path):
    vocabSet = set()
    eqVovabFile = open(file_path, "r")
    num = 0
    for line in eqVovabFile.readlines():
        m = re.search("[^(''),]+", line)
        vocabSet.add(m.group())
        num += 1
        if num >= 1000:
            break
    eqVovabFile.close()
    return vocabSet


# 构建文档向量(和词典长度一样)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("单词 %s 不在词典中" % word)
    return returnVec


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):  # trainMatrix:文档矩阵  trainCategory:类别向量
    numTrainDocs = len(trainMatrix)  # 训练文档数
    numWords = len(trainMatrix[0])  # 每个文本的词向量长度 都是1000
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算侮辱性文档的概率(class=1) ,trainCategory是人工标记的
    p0Num = ones(numWords)  # 初始化类别0的数量 向量
    p1Num = ones(numWords)  # 初始化类别1的数量 向量
    p0Denom = 2.0  # 类别0的单词总数
    p1Denom = 2.0
    for i in range(numTrainDocs):  # 遍历所有文档
        if trainCategory[i] == 1:  # 如果训练语料类别为1
            p1Num += trainMatrix[i]  # 类别1数量向量 + 1
            p1Denom += sum(trainMatrix[i])  # 出现单词总数+1
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # 计算类别1的条件概率   词频向量 / 词总数 (P(词代表类别1|文档为类别1))
    p0Vect = log(p0Num / p0Denom)  # 计算类别0的条件概率
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    file_path = "/Users/lixiwei-mac/Documents/DataSet/recommend/EqNumPNUserComment.txt"
    vocab_path = "/Users/lixiwei-mac/Documents/DataSet/recommend/EqNumPNVocab.txt"
    listOfPosts, listClasses = loadDataSet(file_path)
    # myVocabList = createVocabList(listOfPosts)
    myVocabList = list(loadVocabList(vocab_path))
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testingFile = open("/Users/lixiwei-mac/Documents/DataSet/recommend/NoRatingUserComment.txt")
    analyseResultFile = open("/Users/lixiwei-mac/Documents/DataSet/recommend/NBSResult.txt", "a+")
    for line in testingFile.readlines():
        comment = line.split("##*##")[2]
        userno = line.split("##*##")[0]
        bookno = line.split("##*##")[1]
        testEntry = list(jieba.analyse.extract_tags(comment, topK=10))
        thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
        classifiedType = classifyNB(thisDoc, p0V, p1V, pAb)
        rating = -1
        if classifiedType == 1:
            rating = 5
        else:
            rating = 1
        print(testEntry, '被分类为:', classifiedType)
        analyseResultFile.writelines(userno + "," + bookno + "," + str(rating) + "\r")

    testingFile.close()
    analyseResultFile.close()


# 按照词频排序获取词典
def handlePNvocab(file_path, final_path):
    myDict = {}
    file = open(file_path)
    for line in file.readlines():
        tags = jieba.analyse.extract_tags(line.strip().split("##*##")[3], topK=10)
        for tag in tags:
            if myDict.get(tag) is not None:
                myDict[tag] += 1
            else:
                myDict[tag] = 1
                # print('/'.join(tags))
    myDict = sorted(myDict.items(), key=lambda x: x[1], reverse=True)
    print(myDict)
    file.close()
    dest_file = open(final_path, "a+")
    for line in myDict:
        dest_file.writelines(str(line) + "\r\n")
    dest_file.close()


if __name__ == "__main__":
    # print (jieba.analyse.extract_tags('我是你爸爸', topK=10))



    testingNB()
    # handlePNvocab(file_path,final_path)
    # cut_result_path = "/Users/lixiwei-mac/Documents/DataSet/recommend/CutResult.txt"
    # cut_label_path = "/Users/lixiwei-mac/Documents/DataSet/recommend/CutLabel.txt"
    # cut_words(file_path,cut_result_path,cut_label_path)
    # listOfPosts, listCLasses = loadDataSet(file_path)
    # for line in listOfPosts:
    #     print("".join(line))
    # myVocabList = createVocabList(listOfPosts)
    # print("训练语料数:", listOfPosts.__len__())
    # print(listCLasses.__len__())
    # print("词典单词数", myVocabList.__len__())
    # print("词典:",myVocabList)
    # vocabDocVec = setOfWords2Vec(myVocabList, listOfPosts[100])
    # print(vocabDocVec)

    # trainMat = []
    # for postinDoc in listOfPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    # p0V,p1V,pAb = trainNB0(trainMat,listCLasses)
    # print(pAb)
    # print(p0V)
    # print(p1V)
