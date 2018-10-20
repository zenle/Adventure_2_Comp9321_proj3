import json
import requests
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import re

 
app = Flask(__name__)
 
token = ''

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return "Hello Boss!  <a href='/logout'>Logout</a>"
 
@app.route('/login', methods=['POST'])
def do_admin_login():
 
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
   
    r = requests.get("http://127.0.0.1:5000/token?username="+ POST_USERNAME + "&password=" + POST_PASSWORD)

    if "authorization has been refused for those credentials." not in r.text: 
        token = json.loads(re.sub("\n", "", r.text))['token']
        print(token)

        if POST_USERNAME == "user":
            return "Logged in as User!  <a href='/logout'>Logout</a>"    
        else:
            return "Logged in as Landlord!  <a href='/logout'>Logout</a>"    
    else:
        return home() 

 
@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()
 
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=4000)

