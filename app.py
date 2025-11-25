
from flask import Flask, render_template, request, redirect, url_for
import pickle, numpy as np
from scipy.stats import spearmanr

app = Flask(__name__)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

train_matrix = model["train_matrix"]
user_means = model["user_means"]
rating_scale = model.get("rating_scale",[1,2,3,4,5])

def spearman_similarity(u_ratings, v_ratings, min_overlap=3):
    common = u_ratings.index.intersection(v_ratings.index)
    u_vals = u_ratings.loc[common].dropna()
    v_vals = v_ratings.loc[common].dropna()
    common = u_vals.index.intersection(v_vals.index)
    if len(common) < min_overlap:
        return 0.0, len(common)
    u_vals = u_vals.loc[common]
    v_vals = v_vals.loc[common]
    rho, p = spearmanr(u_vals, v_vals)
    if np.isnan(rho):
        return 0.0, len(common)
    return float(rho), len(common)

def predict_rating(user_id, item_id, k=10, min_overlap=3):
    if item_id not in train_matrix.columns:
        return None
    if user_id not in train_matrix.index:
        return None
    neighbors = train_matrix[train_matrix[item_id].notna()].index.tolist()
    if len(neighbors) == 0:
        return user_means.get(user_id, train_matrix.stack().mean())
    sims = []
    for nb in neighbors:
        if nb == user_id: continue
        rho, overlap = spearman_similarity(train_matrix.loc[user_id], train_matrix.loc[nb], min_overlap=min_overlap)
        if rho != 0.0:
            sims.append((nb, rho))
    if len(sims) == 0:
        return user_means.get(user_id, train_matrix.stack().mean())
    sims_sorted = sorted(sims, key=lambda x: abs(x[1]), reverse=True)[:k]
    num = 0.0
    den = 0.0
    target_mean = user_means.get(user_id, train_matrix.stack().mean())
    for nb, rho in sims_sorted:
        nb_mean = user_means.get(nb, train_matrix.stack().mean())
        nb_rating = train_matrix.at[nb, item_id]
        if np.isnan(nb_rating):
            continue
        num += rho * (nb_rating - nb_mean)
        den += abs(rho)
    if den == 0.0:
        return target_mean
    pred = target_mean + num / den
    pred = float(np.clip(pred, min(rating_scale), max(rating_scale)))
    return pred

def recommend_top_n(user_id, n=5, k=10, min_overlap=3):
    if user_id not in train_matrix.index:
        return []
    unrated = train_matrix.columns[train_matrix.loc[user_id].isna()]
    preds = []
    for item in unrated:
        pr = predict_rating(user_id, item, k=k, min_overlap=min_overlap)
        if pr is not None:
            preds.append((item, pr))
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    return preds_sorted[:n]

@app.route("/", methods=["GET","POST"])
def index():
    users = list(train_matrix.index)
    if request.method == "POST":
        user = request.form.get("user_id")
        topn = int(request.form.get("top_n",5))
        recs = recommend_top_n(user, n=topn)
        return render_template("index.html", users=users, recs=recs, selected=user, topn=topn)
    return render_template("index.html", users=users, recs=None, selected=None, topn=5)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
