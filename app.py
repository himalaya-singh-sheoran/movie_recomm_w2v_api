from flask import Flask,request,json
from flask_cors import CORS

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle



app = Flask(__name__)
CORS(app)

    

def get_recomm(_id:int):
    df = pd.read_csv('final.csv')
    with open("embeddings.txt", "rb") as fp:
        embeddings = pickle.load(fp)
    cosine_similarities = cosine_similarity(embeddings,embeddings)
    ids = df[['id']]
    indices = pd.Series(df.index,index = df['id'])
    ix = indices[_id]
    cosine_sim = list(enumerate(cosine_similarities[ix]))
    cosine_sim = sorted(cosine_sim,key = lambda x: x[1],reverse = True)
    cosine_sim = cosine_sim[1:6]
    indx_ = [i[0] for i in cosine_sim]
    watch_next = ids.iloc[indx_]
    ls = str("")
    for index,row in watch_next.iterrows():
        ls += str(row['id']) + " "
    ls = ls[:-1]
    return ls

@app.route('/movie_id', methods = ['GET'])
def recommend_movies():
    res = int(request.args.get('id'))
    res = get_recomm(res)
    dict_ = {"id":res}
    return json.dumps(dict_)

if __name__ == '__main__':
    app.run(port = 5000)
