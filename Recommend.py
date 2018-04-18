import gzip
import os
import sys

# Configure the environment
import marshal

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/Users/lixiwei-mac/app/spark-1.6.0-bin-hadoop2.6'

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']
os.environ["PYSPARK_PYTHON"] = "/Users/lixiwei-mac/anaconda3/envs/py_35/bin/python3.5"

# Add the PySpark/py4j to the Python Path
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "build"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))

'''
分词并训练评分语义模型
'''
from snownlp import SnowNLP
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext

sc = SparkContext('local', 'pyspark')
'''
    Step1:使用情感分析算法，优化训练数据集
    将无评分记录根据positive值转换成评分
'''
MODEL_PATH = '/myCollaborativeFilter'


def trans_comment():
    import re
    file = open("/Users/lixiwei-mac/Documents/DataSet/recommend/UserComment.txt")
    final_file = open("/Users/lixiwei-mac/Documents/DataSet/recommend/UserRating_2.txt", "a+")
    trans_rating = ""
    count = 0;
    for line in file:
        userNo = line.split("##*##")[0]
        bookNo = line.split("##*##")[1]
        comment = line.split("##*##")[2]
        s = float(SnowNLP(comment).sentiments)
        if s <= 0.2:
            rating = 1
        elif s <= 0.4:
            rating = 2
        elif s <= 0.6:
            rating = 3
        elif s <= 0.8:
            rating = 4
        elif s <= 1:
            rating = 5
        new_rating = userNo + "," + bookNo + "," + str(rating) + "\r\n"
        final_file.writelines(new_rating)
        count = count + 1
    final_file.close()
    file.close()
    print("转换完毕，共转换%s条评分" % count)


def init_model():
    ratings = get_ratings()
    rank = 10
    numIterations = 10
    model = ALS.train(ratings, rank, numIterations, 0.1)
    return ratings, model


def make_predict(ratings, model):
    testdata = ratings.map(lambda p: (p[0], p[1])).cache()
    book_file = sc.textFile('file:///Users/lixiwei-mac/Documents/DataSet/doubanReading/rating/BookName.txt')
    book_name_tag = book_file.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1:])).cache().collectAsMap()
    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2])).cache()
    with open('/Users/lixiwei-mac/Documents/DataSet/doubanReading/rating/predictAll.txt', 'w') as f:
        f.writelines(str(predictions.collect()))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    with open('/Users/lixiwei-mac/Documents/DataSet/doubanReading/rating/ratesAndPreds.txt', 'w') as f:
        f.writelines(str(ratesAndPreds.collect()))
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))


def save_model(model):
    # Save and load model
    model.save(sc, MODEL_PATH)


def load_saved_model():
    return MatrixFactorizationModel.load(sc, MODEL_PATH)


def get_dict():
    fname = '/Users/lixiwei-mac/anaconda/lib/python3.5/site-packages/snownlp/sentiment/sentiment.marshal.3';
    try:
        f = gzip.open(fname, 'rb')
        d = marshal.loads(f.read())
        # print(d)
    except IOError:
        f = open(fname, 'rb')
        d = marshal.loads(f.read())
    f.close()


def get_ratings():
    rating_data = sc.textFile("file:///Users/lixiwei-mac/Documents/DataSet/recommend/UserRating.txt")

    user_no = rating_data.map(lambda x: x.split(',')[0])  # (lxw,4129,tom,helen)

    user2id = user_no.zipWithUniqueId().collectAsMap()  # {lxw:1,4129:2,tom:3,helen:4}

    id2user = {v: k for k, v in user2id.items()}  # {1:lxw,2:4129,3:tom,4:helen}

    ratings = rating_data.map(lambda l: l.split(',')).map(lambda rating_array: Rating(int(user2id.get(rating_array[0])), int(rating_array[1]), int(rating_array[2])))

    return ratings


if __name__ == '__main__':
    # ratings = get_ratings()
    # model = load_saved_model()
    # test_data(ratings, model)
    # print('Done')

    # ratings, model = init_model()
    # save_model(model)
    ratings = get_ratings()
    model = load_saved_model()
    make_predict(ratings, model)
    # trans_comment(ratings,model)
    # trans_comment()
    # get_dict()
