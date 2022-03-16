import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


columns_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=columns_names)
# print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')
# print(movie_titles.head())

df = pd.merge(df, movie_titles, on='item_id')
# print(df.head())
sns.set_style('white')
# print(df.groupby('title')['rating'].mean().sort_values(ascending=False))
# print(df.groupby('title')['rating'].count().sort_values(ascending=False))

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = df.groupby('title')['rating'].count();

print(ratings.head())

# ratings['num of ratings'].hist(bins=70)
# sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=.5)

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
# print(moviemat.head())

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
# print(corr_starwars.head())

# Here we have built a correlation between movies but the values will be garbage as one person might rate starwars 5 and another movie that was not watched by anyone 5 which will show perfect correlation with starwars
# For this we filter out the movies that have less than 100 reviews (100 obtained from the histogram)

corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False))
#this will return really good correlation with other starwars movies

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
# print(corr_starwars.head())

# Here we have built a correlation between movies but the values will be garbage as one person might rate starwars 5 and another movie that was not watched by anyone 5 which will show perfect correlation with starwars
# For this we filter out the movies that have less than 100 reviews (100 obtained from the histogram)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending=False))
plt.show()