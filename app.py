from flask import Flask, render_template, request, Response, jsonify, make_response
import pickle

app = Flask(__name__)
# app = Flask(__name__, template_fol)

clsfy = pickle.load(open('lf.pkl', "rb"))
loaded_vec = pickle.load(open('count_vect.pkl', 'rb'))


@app.route('/')
def doctype():
    return render_template('documenttype.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form['Data']
        print(result);
        result_pred = clsfy.predict(loaded_vec.transform([result]))
        return render_template("documenttyperesult.html", result=result_pred)


@app.route('/result2', methods=['POST'])
def result2():
    request_data = request.get_json();
    data = request_data['ocr'];
    print(data);
    result_pred = clsfy.predict(loaded_vec.transform([data]));
    lt = result_pred.tolist()
    # return render_template("documenttyperesult.html", result=result_pred)
    print(lt)
    res = {"docType": lt}
    # r = Response(response=jsonify(res), status=200, mimetype="application/json")
    # r.headers["Content-Type"] = "application/json"
    # return r
    return make_response(jsonify(res), 200)

@app.route('/addClassificationEntry', methods=['POST'])
def addClassificationEntry():
    request_data  = request.get_json();
    data = request_data['ocr'];
    cat = request_data['category']
    print(cat);
    #load data model file (pkl)
    # - apply stop words
    #apply lemma
    # add data and cat to model
    #save data model

    res = {"status": "success"}
   # r = Response(response=jsonify(res), status=200, mimetype="application/json")
   # r.headers["Content-Type"] = "application/json"
   # return r
    return make_response(jsonify(res), 200)



if __name__ == '__main__':
    app.debug = True
    app.run()
