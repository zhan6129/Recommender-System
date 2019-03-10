# Recommender System - Recommend Movies for a New User using pyspark and AWS

## Use Cases

### E-Commerce

- Retention: Ability to continuously calibrate to the preferences of the user
- Improving Cart Value: By using various means of filtering, companies can suggest new products you’re likely to buy
- Improved Engagement and Delight: Many companies use these systems to encourage engagement and activity on their product or platform

### Healthcare

- Better Understand Patient Health Status, Health Professional as End User
Doctors can use recommender systems to enter patient health information to access latest research results and relevant information that can be used to determine and support their decision. 
- Enables Patient Empowerment, Patient as End User
Recommender systems are useful to help patients feels empowered while monitoring the types of information they receive. With access to many websites such as WebMD, patients sometimes misdiagnose themselves without a strong information base. With a recommendation system, the patient can have access to relevant content that decreases their chances of misleading and inaccurate information.

### Media/News

- Personalize and Focus Interest of the User: Relevant articles may have a limited time span 
- Leverage natural language processing as a feature: This is especially helpful when user behavioral data is sparse.

### Music

- Personalized Music Stations, Mixes, and Playlists:  New music is released every day

### Crowd Sourced Websites

- 'Wisdom of the crowd': Uses “wisdom of the crowd” to recommend similar suggestions a customer might be interested in


## Process

![Optional Text](../master/pic.jpg)

The general flow is to import the dataset to AWS S3. Next, you create a cluster on AWS EMR to use PySpark. This helps utilize resources more efficiently. Using MLLib on a cluster we just created, one can successfully build a recommender system. The final output will be a list of recommendations personalized for each users’ input. We will now walk you through a detailed example of our group using this framework to develop a personalized movie recommender system. 

### Load packages and set sqlContext
```py
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
import sys

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import isnan, isnull
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import functions as f

from pyspark.mllib.recommendation import ALS as rdd_als
from pyspark.sql.functions import lit
import math

sqlContext=SQLContext(sc)
```

### Read the movie and rating data on AWS S3 
We used the data from [Movielens](https://grouplens.org/datasets/movielens/). There are 2 data set we are using:
1) rating.csv, including userId, movieId, rating, timestamp
2) movies.csv, including movieId, movieName, genre 

```py
# read rating file
filename='s3://trendsmarketplacemsba2018fall/Ratings/ratings.csv'
colum1=[
    StructField("userId",LongType()),
    StructField("movieId",LongType()),
    StructField("rating",DoubleType()),
    StructField("timestamp",LongType())
]
schm1=StructType(colum1)
ratingDF=sqlContext.read.format('csv').schema(schm1).option("header",'true').load(filename)
ratingDF=ratingDF.select(ratingDF.userId, ratingDF.movieId, ratingDF.rating)
ratingDF.cache()

# read movie file
filename='s3://trendsmarketplacemsba2018fall/Movies/movies.csv'
colum=[
    StructField("movieId",LongType()),
    StructField("movieName",StringType()),
    StructField("genre",StringType())
]
schm=StructType(colum)
movieDF=sqlContext.read.format('csv').schema(schm).option("header",'true').load(filename)
```

### For each movie, calculate how many reviews it has received from users. 
```py
# caculate movie review
movie_reviews_count=ratingDF.groupBy(ratingDF.movieId).agg({"movieId":"count", "rating":"avg"})
movie_reviews_count = movie_reviews_count.select("movieId", 
 f.col("avg```(rating)").alias("Avg_Rating"),
 f.col("count(movieId)").alias("No_Reviews"))
```

### Define the function to get the new user movie recommendations based on their initial ratings on sample movies.
The function takes 4 inputs:
- inputname: the csv file which include the initial rating from a new user
- outputname: the path on AWS S3 to save the output
- ratingDF: the rating spark dataframe
- movieDF: the movie spark dataframe
- movie_reviews_count: the precaculated aggregated movie information spark dataframe
```py
def get_recommendation(inputname,outputname,ratingDF,movieDF,movie_reviews_count):    
    colum2=[
        StructField("userId",LongType()),
        StructField("movieId",LongType()),
        StructField("movieName",StringType()),
        StructField("rating",DoubleType())
    ]
    schm2=StructType(colum2)
    new_user_rating_df=sqlContext.read.format('csv').schema(schm2).option("header",'true').load(inputname)
    new_user_rating_df = new_user_rating_df.select("userId","movieId","rating")
    new_Id = [int(i.userId) for i in new_user_rating_df.select("userId").limit(1).collect()][0]
    
    #New rating form
    ratingDF_new = ratingDF.sample(False,0.2).union(new_user_rating_df)
    rating_rdd = ratingDF_new.rdd
    new_ratings_model = rdd_als.train(rating_rdd,5,10)

    ids = new_user_rating_df.select(new_user_rating_df.movieId)
    new_ids = [int(i.movieId) for i in ids.collect()]
    new_user_df =  movieDF.filter(movieDF.movieId.isin(*new_ids) == False)
    new_user_rdd=new_user_df.rdd.map(lambda x: (new_Id,x[0] ))

    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_rdd)
    new_user_recommendations_df=new_user_recommendations_RDD.toDF()
    final = new_user_recommendations_df.join(movie_reviews_count,
     new_user_recommendations_df.product == movie_reviews_count.movieId, 
     how="left")
    final = final.filter(final.No_Reviews >= 5000)
    final = final.sort("rating", ascending=False)
    final = final.select('movieId','rating','No_Reviews','Avg_Rating').join(movieDF, "movieId", how='left').limit(10)
    final.write.csv(outputname)#s3://trendsmarketplacemsba2018fall/output/output.csv
    return final
```

### Demonstration

```py
inputname="s3://trendsmarketplacemsba2018fall/Input/new_user.csv"
outputname="s3://trendsmarketplacemsba2018fall/output/output"
get_recommendation(inputname,outputname,ratingDF,movieDF,movie_reviews_count)
```

#### input

| userId | movieId | movieName | rating |
| ------ | ------- | --------- | ------ |
| 0 | 260 | Star Wars (1977) | 1 |
| 0 | 1 |Toy Story (1995) | 4 |
| 0 | 16 | Casino (1995) | 3 |
| 0 | 25 | Leaving Las Vegas (1995) | 4 |
| 0 | 32 | Twelve Monkeys (a.k.a. 12 Monkeys) (1995) | 3 |
| 0 | 335 | Flintstones The (1994) | 1 |
| 0 | 379 | Timecop (1994) | 1 |
| 0 | 296 | Pulp Fiction (1994) | 5 | 
| 0 | 858 | Godfather The (1972) | 5 | 
| 0 | 50 | Usual Suspects The (1995) | 4 |
| 0 | 122912 | Avengers: Infinity War - Part I (2018) | 3 |
| 0 | 143347 | Aquaman (2018) | 3 |
| 0 | 164909 | La La Land (2016)| 5 |
| 0 | 7451 | Mean Girls (2004) | 5 |
| 0 | 8368 | Harry Potter and the Prisoner of Azkaban (2004) | 4 |

#### Output

| Average Rating | Movie | Genre |
| -------------- | ----- | ----- |
| 3.68 | Piano, The (1993) | Drama, Romance |
| 3.64 | Crying Game, The (1992) | Drama,Romance,Thriller |
| 3.96 | Postman, The (Postino, Il) (1994) | Comedy,Drama,Romance |
| 3.92 | Like Water for Chocolate (Como agua para chocolate) (1992) | Drama,Fantasy,Romance |
| 3.64 | English Patient, The (1996) | Drama,Romance,War |
| 3.89 | Remains of the Day, The (1993) | Drama,Romance |
| 4.12 | Amelie (Fabuleux destin d'AmÃ©lie Poulain, Le) (2001) | Comedy,Romance |
| 4.12 | Cinema Paradiso (Nuovo cinema Paradiso) (1989) | Drama |
| 3.95 | Sense and Sensibility (1995) | Drama，Romance |
| 3.48 | Muriel's Wedding (1994) | Comedy |


### Reference
1. [Collaborative Filtering - RDD-based API!](https://spark.apache.org/docs/2.2.0/mllib-collaborative-filtering.html)
2. [Building a Movie Recommendation Service with Apache Spark & Flask!](https://www.codementor.io/jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw)
3. [Flask WebPage to showcase!](https://github.com/Armando8766/MovieRecommender)
