from flask import Flask
import pandas as pd
import pickle

from flask import request, jsonify

application = app = Flask(__name__)


def predict_fire_cause(a, c, d, e, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x):
    pfile = open("model.pkl", "rb")
    model = pickle.load(pfile)

    y_predict = model.predict(pd.DataFrame({'A': [a],
                                            'C': [c],
                                            'D': [d],
                                            'E': [e],
                                            'H': [h],
                                            'I': [i],
                                            'K': [k],
                                            'L': [l],
                                            'M': [m],
                                            'N': [n],
                                            'O': [o],
                                            'P': [p],
                                            'Q': [q],
                                            'R': [r],
                                            'S': [s],
                                            'T': [t],
                                            'U': [u],
                                            'V': [v],
                                            'W': [w],
                                            'X': [x]}))[0]

    return y_predict


@app.route("/")
def hello():
    return "A simple web service for accessing a ML model to classify probability of the sample's being a part of some secret class."


@app.route("/predict", methods=["GET"])
def api_all():
    a = request.args["a"]
    c = request.args["c"]
    d = request.args["d"]
    e = request.args["e"]
    h = request.args["h"]
    i = request.args["i"]
    k = request.args["k"]
    l = request.args["l"]
    m = request.args["m"]
    n = request.args["n"]
    o = request.args["o"]
    p = request.args["p"]
    q = request.args["q"]
    r = request.args["r"]
    s = request.args["s"]
    t = request.args["t"]
    u = request.args["u"]
    v = request.args["v"]
    w = request.args["w"]
    x = request.args["x"]
    probability = predict_fire_cause(a, c, d, e, h, i, k, l, m, n, o, p, q, r, s, t, u, v, w, x)

    return jsonify(probability=probability)


if __name__ == '__main__':
    app.run()
