'''
初始化SparkContext
'''

import os
import sys

# Configure the environment
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/Users/lixiwei-mac/app/spark-1.6.0-bin-hadoop2.6'

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

# Add the PySpark/py4j to the Python Path
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "build"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
from flask import Flask, request
import json


sc = SparkContext('local', 'pyspark')
data = sc.textFile("file:////Users/lixiwei-mac/Documents/DataSet/doubanReading/rating/UserRating.txt").cache()
user_no = data.map(lambda x: x.split(',')[0]).cache()  # (lxw,4129,tom,helen)
user2id = user_no.zipWithUniqueId().cache().collectAsMap()  # {lxw:1,4129:2,tom:3,helen:4}
'''
从本地获取模型
'''


def getLocalModel():
    model = MatrixFactorizationModel.load(sc,
                                          "file:////Users/lixiwei-mac/Documents/DataSet/doubanReading/rating/doubanCFModel")
    return model
'''
测试推荐
'''
def get_recommend_by_userno(userno,model):
    global first
    predict_num = 10
    recommendations = model.recommendProducts(user2id.get(userno),predict_num)
    booknos = []
    for obj in recommendations:
        booknos.append(obj[1])
    return booknos


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return "Hello!!"


@app.route('/getRecommend', methods=['POST'])
def get_recommend():
    userno = request.form['userno']
    model = getLocalModel()
    result = get_recommend_by_userno(userno,model)
    return json.dumps(result)




if __name__ == '__main__':
    app.run()
