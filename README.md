# Anime recommendation system

In this project as part of a recommendation systems course I chose to make a hybrid recommendation system for anime films.
The system combines content-based recommendations, collaborative filtering and default recommendations.

## DataSet:
The dataset i chose for this project is a dataset of anime movies from the Kaggle site, 
which contains ratings of 73,516 users on 12,294 anime movies with an amount of about 1,048567 ratings, and consists of two CSV tables.

## Definition:
Collaborative Filtering - The purpose of collaborative filtering is to make it easier for users to find products and / or knowledge by finding similarities between them and the browsing and consumption history of other users with similar characteristics.
The basic premise of the collaborative filtering approach is that if users have similarities in a particular topic or product, then there is a chance that their preferences will be similar in other topics and in other products as well.

Content Based - A Content-Based Recommender works by the data that we take from the user, either explicitly (rating) or implicitly (clicking on a link). By the data we create a user profile, which is then used to suggest to the user, as the user provides more input or take more actions on the recommendation, the engine becomes more accurate.

## Implementation:
- Collaborative Filtering: A pivot table was used that was input to the knn classifier.

- Content Based: A tfidf vector representation was used that was input to the cosine similarity.

- Default recommendation: Movies are randomly selected with the highest rating. 

## How the system works:
![System](https://user-images.githubusercontent.com/63209732/123109156-aae58d00-d443-11eb-87f8-d34efda04355.png)

## Website of a recommendation system for anime:
![System1](https://user-images.githubusercontent.com/63209732/123112703-99ea4b00-d446-11eb-81ed-3cf57cda6f0b.png)

After entering a user who passed the filtering of step 1 - Recommendations of the CF system, based on similarity between users.
![System2](https://user-images.githubusercontent.com/63209732/123114499-09ad0580-d448-11eb-8fe9-a2c24109da9f.jpg)

Step 2- For identifying an existing user in the system, when the user does not have enough imagination with other users. 
Based on ratings that the user himself gave to the movies.

A recommendation is made according to content based.
![System3](https://user-images.githubusercontent.com/63209732/123116171-59400100-d449-11eb-817c-745e28a1a55d.jpg)

Step 3 - New user, without ratings and similarity with other users -  A default recommendation is made.
![System4](https://user-images.githubusercontent.com/63209732/123116631-bd62c500-d449-11eb-910c-10ecbaba3047.jpg)

## Run:
myWeb.py
