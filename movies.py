# MovieLens data of movie ratings by users
# Collaborative filtering 
# Recommend movies for a user
# Pearson's correlation coefficient used


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import random

# load the 3 files: user info, ratings, movie info

users = pd.read_table('users.dat',sep='::',header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip'])
ratings = pd.read_table('ratings.dat',sep='::',header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_table('movies.dat',sep='::',header=None,names=['movie_id', 'title', 'genres'])

print 'users ', users.shape
print 'ratings ', ratings.shape
print 'movies ', movies.shape


data = pd.merge(pd.merge(ratings, users), movies)
data = data.drop(['gender', 'occupation', 'timestamp', 'zip', 'age'],1)
print 'data ', data.shape
print data[:10]

num_users = 6040
num_movies = 3952
num_data = data.shape[0]

# reviews is 1s and 0s: if user has rated movie = 1; else = 0
# rating_mu is rating given for movie m by user u
reviews = np.zeros((num_movies+1,num_users+1), dtype=np.float32)
rating_mu = np.zeros((num_movies+1,num_users+1), dtype=np.float32)
for i in range(num_data):
  m = data.movie_id[i]
  u = data.user_id[i]  
  reviews[m,u] = 1.0
  rating_mu[m,u] = data.rating[i]
  
print 'reviews ', reviews.shape 
print 'rating_mu ', rating_mu.shape



# FIND CLOSEST USERS FOR A GIVEN USER

# Pearson correlation coefficient between users
pc = np.zeros((num_users+1), dtype=np.float32)
i = np.random.randint(1,num_users+1) 
print 'randomly chosen user: ', i
for j in range(1,num_users+1):  
 pc[j] = scipy.stats.pearsonr(reviews[:,i],reviews[:,j])[0]
pc[i] = 0.0  

# N closest users to user i
N = 10
ipc = np.argsort(pc)[-N:][::-1]
print N, 'closest users to user ', i, 'are: ', ipc
print 'with Pearson correlation coefficient: ', pc[ipc] 


N = num_users
ipc = np.argsort(pc)[-N:][::-1]


# Movies not rated by user i
movies_i = np.array([m  for m in range(1,num_movies+1) if (reviews[m,i]==0)])
not_rated = movies_i.shape[0]
print '# of movies not rated by user ', i, ': ', not_rated
print movies_i[:20]

# We will predict ratings for the movies he/she has not rated
# Find nearest user to user i 
ratings_i = np.zeros((not_rated), dtype=np.float32) 
for j in range(not_rated):
  m = movies_i[j]
  num = 0.0
  den = 0.0  
  for k in range(num_users) :
    u = ipc[k]   
    nn = 1
    if(u!=i and rating_mu[m,u]>0 and nn == 1):
     ratings_i[j] = rating_mu[m,u]
     nn = 2
     break 
      
   

print 'our prediction of rating for the movies user ', i, 'has not rated: '
ratings_i = ratings_i.astype(int)
print ratings_i

for j in range(5,0,-1):
 print 'Number of movies we predicted for user ', i, 'with ', j, 'stars: ', ratings_i[ratings_i == j].shape

