#Collaborative Filtering- user to user
import heapq;
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
pd.set_option('display.max_columns', None)

class anime_recommend():
    #---------------------collaborative filtering movies
    def __make_table(self):
        anime_data = pd.read_csv('anime.csv')
        rating_data = pd.read_csv('rating.csv')
        merged_data = pd.merge(anime_data, rating_data, on='anime_id')
        merged_data = merged_data.rename(columns={'name': 'anime_title', 'rating_y': 'user_rating'})
        merged_data = merged_data[['anime_id', 'user_id', 'user_rating', 'anime_title']]
        merged_data = merged_data.dropna(axis=0)
        # users users
        rateNumbers = 1000
        counts = merged_data['user_id'].value_counts()
        merged_data = merged_data[merged_data['user_id'].isin(counts[counts >= rateNumbers].index)]

        # user rating normalization
        merged_data['user_rating']=merged_data['user_rating'].replace([-1,1,2,3,4,5], 1).replace([6,7], 2).replace(8,3).replace([9,10], 4)
        #merged_data['user_rating'] = merged_data['user_rating'].replace([-1, 1, 2, 3, 4, 5, 6, 7], 1).replace([8, 9, 10], 2)

        # creating user-item matrix
        #merged_data = merged_data[0:30000]
        user_movie_rating = merged_data.pivot_table(index='user_id', columns='anime_title', values='user_rating')
        user_movie_rating.fillna(0, inplace=True)
        # anime_matrix = csr_matrix(user_movie_rating.values)
        user_movie_rating.to_csv("user_movie_rating.csv")
        return user_movie_rating

    def __build_knn(self):
        # Knn-model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(self.user_movie_rating)
        knnPickle = open('knn_model', 'wb')
        pickle.dump(model_knn, knnPickle)
        print("model was saved")

    def __find_similar_users(self,user_id):
        distances, indices = self.knn_model.kneighbors(self.user_movie_rating.iloc[user_id, :].values.reshape(1, -1),n_neighbors=6)
        return distances, indices

    def __top_recommendations(self, distances, indices, random_user, recom_num):
        dict = {}
        movie_names = list(self.user_movie_rating.columns.values.tolist())
        recommend = {}
        for i in range(1, len(distances.flatten())):  # all simlar users found before
            for index, j in enumerate(self.user_movie_rating.iloc[indices.flatten()[i]]):  # all user movies rate
                if j == 4 and (self.user_movie_rating.iloc[random_user, :])[index] == 0:
                    if movie_names[index] not in recommend or (recommend[movie_names[index]] < distances.flatten()[i]):
                        recommend[movie_names[index]] = distances.flatten()[i]
        if (len(recommend) == 0):
            print("ERROR! user watched all movies that the recommenders watched")
        else:
            final_recommend = heapq.nlargest(recom_num, recommend)
        final_dict = {}
        for x in final_recommend:
            final_dict[x] = recommend[x]
        return final_dict

    def __check_if_user_exist(self,user_id):
        if int(user_id) in list(self.user_movie_rating.index):
            return True
        return False

    def __make_suggest_collaborative(self,user_id,k):
        if int(user_id) not in list(self.user_movie_rating.index): #check if user exist in user_movie_rating
            return False
        user_id=list(self.user_movie_rating.index).index(user_id)
        distances, indices = self.__find_similar_users(user_id)
        recommend_movies = self.__top_recommendations( distances, indices, user_id,k)
        return recommend_movies

    # ---------------------content base movies
    def __vector_tf_idf(self,anime_data):
        # getting tfidf
        tfv = TfidfVectorizer(min_df=3, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3),
                              stop_words='english')

        # Filling NaNs with empty string
        anime_data['genre'] = anime_data['genre'].fillna('')
        genres_str = anime_data['genre'].str.split(',').astype(str)
        tfv_matrix = tfv.fit_transform(genres_str)
        return tfv_matrix

    def __cosine_sim_linear_kernel(self,anime_data):
        anime_data = pd.read_csv('anime.csv')
        cosine_sim = linear_kernel(self.tfv_matrix, self.tfv_matrix)
        tf_sim = pd.DataFrame(data=cosine_sim)
        tf_sim.index = anime_data['name']
        tf_sim.columns = anime_data['name']
        return tf_sim

    def __similar_animes_content_based(self,anime,anime_data,k):
        if anime not in self.tf_sim.columns:
            return ('No anime called {}'.format(anime))
        sim_values = self.tf_sim.sort_values(by=anime, ascending=False).loc[:, anime].tolist()[1:k+1]
        sim_animes = self.tf_sim.sort_values(by=anime, ascending=False).index[1:k+1]
        rating_anime = dict(pd.concat([anime_data['name'], anime_data['rating']], axis=1).values.tolist())

        frames = pd.DataFrame([[anime, sim, rating_anime[anime]] for anime, sim in zip(sim_animes, sim_values)]).rename(
            columns={0: "anime name", 1: "similarity", 2: "rating"})
        return frames

    def __make_suggest_content_base(self, user_id,k):
        anime_data = pd.read_csv('anime.csv')
        rating_data = pd.read_csv('rating.csv')
        if user_id in list(rating_data['user_id']):
            df_anime_data = pd.merge(anime_data, rating_data, on='anime_id', suffixes=['', '_user'])
            data_name_anime_user = df_anime_data.loc[df_anime_data['user_id'].isin([user_id])]
            data_name = data_name_anime_user.loc[
                df_anime_data['rating_user'].isin([data_name_anime_user['rating_user'].max()])]
            name_anime = list(data_name['name'])[0]
            self.tfv_matrix = self.__vector_tf_idf(anime_data)
            self.tf_sim = self.__cosine_sim_linear_kernel(anime_data)
            return self.__similar_animes_content_based(name_anime, anime_data,k)
        else:
            return False

    # ---------------------User Group Clustering
    def __one_hot_genre(self,anime_data):
        if anime_data['genre'] is np.nan:
            return anime_data
        else:
            genres = list(map(lambda row: row.strip(), anime_data['genre'].split(',')))
            for genre in genres:
                anime_data[genre] = 1
            return anime_data

    def __one_hot_type(self,rating_data):
        one_hot = pd.get_dummies(self.genre_one_hot['type'])
        one_hot[one_hot == 0] = np.nan
        anime_one_hot = (self.genre_one_hot
                         .drop(columns=['type', 'episodes', 'genre','rating','name', 'members'])
                         .join(one_hot, rsuffix='-type'))
        return anime_one_hot

    def __rating_to_weight_anime(self,rating_data):
        # the rating becomes a weight in the anime properties anime_one_hot is the dataframe joined before.
        rating_anime = rating_data.join(self.anime_one_hot.set_index('anime_id'), on='anime_id')
        attr = self.anime_one_hot.columns.tolist()
        attr.remove('anime_id')
        rating_anime[attr] = rating_anime[attr].mul(rating_anime['rating'], axis=0)

        return rating_anime

    def __users(self):
        # calculate user preference as the mean values for its gradings in each category
        users = (self.rating_anime
                 .drop(columns=['anime_id', 'rating'])
                 .groupby(by='user_id')
                 .mean())
        users= users.fillna(value=0)
        return users

    def __pca_modle(self):
        pca = PCA()
        pca.fit(self.users)
        number_of_components = 20
        pca.set_params(n_components=number_of_components)
        # Fit on training set only
        pca.fit(self.users)
        # Apply transform to both the training set and the test set
        users_pca = pca.transform(self.users)
        users_pos_pca = pd.DataFrame(users_pca)
        users_pos_pca['user_id'] = self.users.index
        users_pos_pca = users_pos_pca.set_index('user_id')
        return users_pos_pca

    def __K_means(self):
        users_with_label = pd.DataFrame(PCA(n_components=3).fit_transform(self.users))
        users_with_label['user_id'] = self.users.index
        users_with_label = users_with_label.set_index('user_id')
        # find each user's cluster
        kmeans = KMeans(n_clusters=6, n_init=30)
        pickle.dump(kmeans, open('Kmeans_modle', 'wb'))

    def main_kmeans(self):
        anime_data = pd.read_csv('anime.csv')
        rating_data = pd.read_csv('rating.csv')
        self.genre_one_hot = anime_data.apply(self.__one_hot_genre, axis=1)
        self.anime_one_hot = self.__one_hot_type(rating_data)
        self.rating_anime=self.__rating_to_weight_anime(rating_data)
        self.users=self.__users()
        self.users_pos_pca = self.__pca_modle()
        self.__K_means()

    #----------------------default suggestion
    def __most_popular_anime(self,k):
        anime_data = pd.read_csv('anime.csv')
        rating_data = pd.read_csv('rating.csv')
        df_anime_fulldata = pd.merge(anime_data, rating_data, on='anime_id', suffixes=['', '_user'])
        df_anime_fulldata = df_anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating', 'rating': 'anime_rating'})
        get_anime_count = df_anime_fulldata['anime_title'].value_counts()
        dict_anime = dict(zip(list(get_anime_count.index), list(get_anime_count)))
        sort_dict_anime = list(dict(sorted(dict_anime.items(), key=lambda x: x[1], reverse=True)).keys())[:k]
        return sort_dict_anime

    #---------------------site use
    def make_suggest(self,user_id,k):
        recom=self.__make_suggest_collaborative(user_id,k)
        if(recom==False):
            print("cf suggest failed, making content based suggest")
            recom=self.__make_suggest_content_base(user_id,k)
            if(type(recom)==bool):
                print("cb suggest failed, making default suggestions")
                recom=self.popular[:k]
            else:
                recom = list((recom)['anime name'])
        else:
            recom=list(recom.keys())
        return recom

    #---------------------------
    def __init__(self):

        self.user_movie_rating = self.__make_table()
        #load models
        try:
            self.knn_model=pickle.load(open('knn_model', 'rb'))
            print("collaborative filtering-KNN model wad loaded")

        except():
            print("KNN model was not found, creating a new model...")
            self.__build_knn()
            self.knn_model = pickle.load(open('knn_model', 'rb'))

        self.popular=self.__most_popular_anime(20)