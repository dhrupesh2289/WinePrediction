
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce

spark = SparkSession.builder.master("local[*]").getOrCreate()


try:
    if len(sys.argv) == 2:
        filepn = str(sys.argv[1])
        data_test = spark.read.option("delimiter", ";").csv(filepn, header=True, inferSchema=True)
        print("Argument passed is:", str(sys.argv[1]))
    else:
        data_test = spark.read.option("delimiter", ";").csv('ValidationDataset.csv', header=True, inferSchema=True)
except:
    print("Check your directory")
    exit(0)


data_train = spark.read.option("delimiter", ";").csv('TrainingDataset.csv', header=True, inferSchema=True)

# To clean out CSV headers if quotes are present
old_column_name = data_train.schema.names
print(data_train.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"', ''))

data_train = reduce(lambda data_train, idx: data_train.withColumnRenamed(old_column_name[idx], clean_column_name[idx]),
                   range(len(clean_column_name)), data_train)
data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]),
                  range(len(clean_column_name)), data_test)
print(data_train.schema)


data_train_new = data_train.filter(data_train['quality'] != "3")


feature_cols = [x for x in data_train_new.columns if x != "quality"]


vect_assembler = VectorAssembler(inputCols=feature_cols, outputCol="feature")


Scaler = StandardScaler().setInputCol('feature').setOutputCol('Scaled_feature')


logr = LogisticRegression(labelCol="quality", featuresCol="Scaled_feature")

# Creating a pipeline object of stages: [VectorAssembler, StandardScaler, LogisticRegression]
Pipe = Pipeline(stages=[vect_assembler, Scaler, logr])


PipeModel = Pipe.fit(data_train_new)


PipeModel.write().overwrite().save("Modelfile")


try: 
    train_prediction = PipeModel.transform(data_train)
    test_prediction = PipeModel.transform(data_test)
except:
    print("Please check CSV file(Training & Validation). Labels may be improper ")


evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")


train_F1score = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})


train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})


print("[Train] F1 score =", train_F1score)
print("[Train] Accuracy =", train_accuracy)
print("[Validation] F1 score =", test_F1score)
print("[Validation] Accuracy =", test_accuracy)


fp = open("results.txt", "w")
fp.write("[Train] F1 score = %s\n" % train_F1score)
fp.write("[Train] Accuracy = %s\n" % train_accuracy)
fp.write("[Test] F1 score = %s\n" % test_F1score)
fp.write("[Test] Accuracy = %s\n" % test_accuracy)
fp.close()