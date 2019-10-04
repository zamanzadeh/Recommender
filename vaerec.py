import collections
import itertools
import math
import scipy

import networkx as nx
import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt


dataPath = '/users/hadi/documents/projects/gbrs/data/100k/'
assert (dataPath != ""), "enter a valid data path"

df = pd.read_csv(dataPath+'u1.base', sep='\\t', engine='python', names=['UID', 'MID', 'rate', 'time'])

# print(df)

pairs = []
grouped = df.groupby(['MID', 'rate'])
for key, group in grouped:
    pairs.extend(list(itertools.combinations(group['UID'], 2)))

# print(pairs)

counter = collections.Counter(pairs)
alpha = 0.01 * 1682  # param*i_no
edge_list = map(list, collections.Counter(el for el in counter.elements() if counter[el] >= alpha).keys())

G = nx.Graph()
for el in edge_list:
    G.add_edge(el[0], el[1], weight=1)
# print(G.edges())
# nx.draw(G)
# plt.show()

#           User Features
df_user = pd.read_csv(dataPath+'u.user', sep='\\|', engine='python',
                      names=['UID', 'age', 'gender', 'job', 'zip'])


def convert_categorical(df_X, _X):
    values = np.array(df_X[_X])
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #   binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    # invert first example
    # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    # print(inverted)
    df_X = df_X.drop(_X, 1)
    for j in range(integer_encoded.max() + 1):
        df_X.insert(loc=j + 1, column=str(_X) + str(j + 1), value=onehot_encoded[:, j])
    return df_X


df_user = convert_categorical(df_user, 'job')
df_user = convert_categorical(df_user, 'gender')

# define age
df_user['bin'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 100], labels=['1', '2', '3', '4', '5', '6'])
df_user['age'] = df_user['bin']
df_user = df_user.drop('bin', 1)
df_user = convert_categorical(df_user, 'age')

#           Graph Features
pr = nx.pagerank(G.to_directed())
df_user['PR'] = df_user['UID'].map(pr)
df_user['PR'] /= float(df_user['PR'].max())

dc = nx.degree_centrality(G)
df_user['CD'] = df_user['UID'].map(dc)
df_user['CD'] /= float(df_user['CD'].max())

cc = nx.closeness_centrality(G)
df_user['CC'] = df_user['UID'].map(cc)
df_user['CC'] /= float(df_user['CC'].max())

bc = nx.betweenness_centrality(G)
df_user['CB'] = df_user['UID'].map(bc)
df_user['CB'] /= float(df_user['CB'].max())

lc = nx.load_centrality(G)
df_user['LC'] = df_user['UID'].map(lc)
df_user['LC'] /= float(df_user['LC'].max())

nd = nx.average_neighbor_degree(G, weight='weight')
df_user['AND'] = df_user['UID'].map(nd)
df_user['AND'] /= float(df_user['AND'].max())

#       Auto Encoder
df_user = df_user.drop('zip', 1)
X_train = df_user[df_user.columns[1:]]
X_train.fillna(0, inplace=True)

ncol = X_train.shape[1]
input_dim = Input(shape=(ncol,))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 2
inter_dim = 16
encoding_dim = 8
# DEFINE THE ENCODER LAYER
encoded = Dense(inter_dim, activation='relu')(input_dim)
encoded = Dense(encoding_dim, activation='relu', name='bottleneck')(encoded)
# DEFINE THE DECODER LAYER
decoded = Dense(inter_dim, activation='relu')(encoded)
decoded = Dense(ncol, activation='sigmoid')(decoded)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(inputs=input_dim, outputs=decoded)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=False, validation_data=(X_train, X_train))

encoder = Model(inputs=input_dim, outputs=encoded)
Zenc = encoder.predict(X_train)  # bottleneck representation

# Compute input of clustering
X = StandardScaler().fit_transform(Zenc)

# elbow method to find optimal number of clusters in K-means
wcss = []
K = range(1, 51)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    wcss.append(km.inertia_)

km10 = KMeans(n_clusters=10)
km10 = km10.fit(X)
y_kmeans = km10.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# centers = km10.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


distances = []
for i in range(1, 50):
    # distances.append(p.distance_to_line(p1, p2))
    ch = abs((wcss[49] - wcss[0]) * i - (50 - 1) * wcss[i - 1] + (50 * wcss[0] - 1 * wcss[49]))
    dis = math.sqrt(math.pow(50 - 1, 2) + math.pow(wcss[49] - wcss[0], 2))
    distances.append(ch / dis)

n_clusters_ = np.argmax(distances) + 1
print("NOC = ", n_clusters_)
kmeans = KMeans(n_clusters_, random_state=0).fit(X)
# print kmeans.labels_
labels = kmeans.labels_
usr_cluster = []
cluster_rate = []
for j in range(0, n_clusters_):
    usr_cluster.append([X[0] for X, value in np.ndenumerate(labels) if value == j])
    # print indexes
    ls = df[df['UID'].isin(usr_cluster[j])]
    cluster_rate.append(ls.groupby('MID')['rate'].mean())


def find_similar_movie(crate, mid):
    rec_rate = 1
    df_item = pd.read_csv(dataPath+'u.item', sep='\\|', engine='python',
                          names=['MID', 'title', 'rdate', 'vdate', 'URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                                 'Western'])
    np_item = np.array(df_item[df_item.columns[5:]])
    genres, = np.where(np_item[mid - 1] == 1)
    # print "genres", genres
    for kk, vv in crate.iteritems():
        comp_genres, = np.where(np_item[kk - 1] == 1)
        if np.array_equal(genres, comp_genres):
            rec_rate = vv
            # print("rec_rate", rec_rate)
            return rec_rate

    for kk, vv in crate.iteritems():
        # print kk

        return rec_rate


#   test - calculate mse
s_err = 0
df_test = pd.read_csv(dataPath+'u1.test', sep='\\t', engine='python', names=['UID', 'MID', 'rate', 'time'])
for i in range(0, len(df_test)):
    cc = labels[df_test['UID'][i]]
    mm = df_test['MID'][i]
    rr = df_test['rate'][i]
    try:
        predict_rate = cluster_rate[cc][mm]
        # print predict_rate
        s_err += pow((predict_rate - rr), 2)
    except KeyError:
        try:
            mm = cluster_rate[cc][mm]
        except KeyError:
            # print "C: ", cc, ", M: ", mm, ", R: ", rr
            predict_rate = find_similar_movie(cluster_rate[cc], mm)
            s_err += pow((predict_rate - rr), 2)
        pass

rmse = math.sqrt(float(s_err / len(df_test)))
print("RMSE: ", rmse)
