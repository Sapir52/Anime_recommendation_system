from flask import Flask, request, render_template,url_for,redirect
from anime_recommendations import anime_recommend
app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def login():
    if request.method=="POST":
        user= request.form["un"]
        return redirect(url_for('UserSuggest',user=user))
    else:
        return render_template('index.html')



@app.route("/<user>" ,methods=["POST","GET"]) #continue from here
def UserSuggest(user):
    if request.method=="POST":
        user= request.form["un"]
        return redirect(url_for('UserSuggest',user=user))
    print("user:"+user)
    user = int(user)
    suggest = recommend_model.make_suggest(user,5)
    print(suggest)
    return render_template('index.html', suggest=suggest)


if __name__ == '__main__':
    recommend_model =anime_recommend()
    app.run(debug=True)
