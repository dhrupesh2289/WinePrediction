
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce

spark = SparkSession.builder.master("local[*]").getOrCreate()


try:
    filepn = "/job/" + str(sys.argv[1])
    data_test = spark.read.option("delimiter", ";").csv(filepn, header=True, inferSchema=True)
    print("Input file :", str(sys.argv[1]))

except:
    exit()


old_column_name = data_test.schema.names
print(data_test.schema)
clean_column_name = []

for name in old_column_name:
    clean_column_name.append(name.replace('"', ''))

data_test = reduce(lambda data_test, idx: data_test.withColumnRenamed(old_column_name[idx], clean_column_name[idx]),
                   range(len(clean_column_name)), data_test)
print(data_test.schema)


try:
    PipeModel = PipelineModel.load("/job/Modelfile")
except:
    exit()

try:
    test_prediction = PipeModel.transform(data_test)
except:
    print("---")


test_prediction.drop("feature", "Scaled_feature", "rawPrediction", "probability").write.mode("overwrite").option(
    "header", "true").csv("/job/resultdata.csv")
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")


test_F1score = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

print("[Test] F1 score =", test_F1score)
print("[Test] Accuracy =", test_accuracy)
fp = open("/job/results.txt", "w")
fp.write("[Test] F1 score =  %s\n" % test_F1score)
fp.write("[Test] Accuracy =  %s\n" % test_accuracy)

# Closing the file
fp.close()