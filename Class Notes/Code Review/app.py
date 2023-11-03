from flask import Flask, request
import pickle
import numpy as np

app = Flask("yourapp")
#app = Flask(__name__)

# home route that returns below text when root url is accessed
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/hello")
def viewfunc():
    return "<p>Second Route</p>"

@app.route("/hello/bye")
def viewfunc1():
    return "<h1>Third Route</h1>"

@app.route("/add",methods=["POST"])
def add():
    a=request.json['a']
    b=request.json['b']
    return {"sum":int(a)+int(b)}


@app.route("/iris",methods=["POST"])
def classify():
    a=request.json['a']
    b=request.json['b']
    c=request.json['c']
    d=request.json['d']
    loaded_model = pickle.load(open("model.pkl", 'rb'))
    # row,features as we have 1 row here shape is 1,features (4)
    test=np.array([a,b,c,d]).reshape(1,-1)
    print(test.shape)
    print(loaded_model.predict(test))
    return {"prediction":str(loaded_model.predict(test)[0])}


if __name__ == '__main__':  
   app.run()
