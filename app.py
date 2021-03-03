from flask import Flask,request,json
from flask_cors import CORS

import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)
CORS(app)


def get_vectors(df):
    global embeddings
    embeddings = []
    model =  Word2Vec.load('model_w2v.model')
    for line in df['content']:
        w2v = count = 0
        for word in line.split():
            if word in model.wv.vocab:
                count +=1
                if w2v is None:
                    w2v = model[word]
                else :
                    w2v +=  model[word]
        if w2v is not None:
            w2v = w2v/count
            embeddings.append(w2v)
    

def get_recomm(_id:int):
    df = pd.read_csv('final.csv')
    get_vectors(df)
    cosine_similarities = cosine_similarity(embeddings,embeddings)
    ids = df[['id']]
    indices = pd.Series(df.index,index = df['id']).drop_duplicates()
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
    app.run(port = 5000, debug = True)