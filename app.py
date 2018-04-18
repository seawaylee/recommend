'''
初始化SparkContext
'''

import os
import sys

# Configure the environment

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/Users/lixiwei-mac/app/spark-1.6.0-bin-hadoop2.6'
    os.environ['PYSPARK_PYTHON'] = '/Users/lixiwei-mac/anaconda3/envs/py_35/bin/python3.5'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/lixiwei-mac/anaconda3/envs/py_35/bin/python3.5'

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

# Add the PySpark/py4j to the Python Path
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "build"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))
from pyspark.mllib.recommendation import MatrixFactorizationModel
from pyspark import SparkContext
from flask import Flask, request
import json

MODEL_PATH = '/myCollaborativeFilter'
sc = SparkContext('local', 'pyspark')
rating_data = sc.textFile("file:///Users/lixiwei-mac/Documents/DataSet/recommend/UserRating.txt").cache()
user_no = rating_data.map(lambda x: x.split(',')[0])  # (lxw,4129,tom,helen)
user2id = user_no.zipWithUniqueId().collectAsMap()  # {lxw:1,4129:2,tom:3,helen:4}
model = None


def getLocalModel():
    return MatrixFactorizationModel.load(sc, MODEL_PATH)


def get_recommend_by_userno(userno):
    predict_num = 300
    user_products = model.recommendProducts(user2id.get(userno), predict_num)
    booknos = []
    for obj in user_products:
        print(obj)
        booknos.append({'bookId': obj[1], 'score': obj[2]})
    return booknos


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return "Hello!!"


@app.route('/getRecommend', methods=['GET'])
def get_recommend():
    userno = request.args.get('userno')
    result = get_recommend_by_userno(userno)
    return json.dumps(result)


if __name__ == '__main__':
    model = getLocalModel()
    app.run()
