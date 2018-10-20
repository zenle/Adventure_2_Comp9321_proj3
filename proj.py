import requests
#import data_processing
#import price_predicting
import json
from functools import wraps
import pandas as pd
import json
import string
import random
import time
from flask import Flask, jsonify
from flask import request
from itsdangerous import SignatureExpired, JSONWebSignatureSerializer, BadSignature
from flask_restplus import Resource, Api, abort
from flask_restplus import fields
from flask_restplus import inputs
from flask_restplus import reqparse



class AuthenticationToken:
    def __init__(self, secret_key, expires_in):
        self.secret_key = secret_key
        self.expires_in = expires_in
        self.serializer = JSONWebSignatureSerializer(secret_key)

    def generate_token(self, username):
        info = {
            'username': username,
            'creation_time': time.time()
        }

        token = self.serializer.dumps(info)
        return token.decode()

    def validate_token(self, token):
        info = self.serializer.loads(token.encode())

        if time.time() - info['creation_time'] > self.expires_in:
            raise SignatureExpired("The Token has been expired; get a new token")

        return info['username']

secret_key = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase, k=30))
expires_in = 600
auth = AuthenticationToken(secret_key, expires_in)

app = Flask(__name__)
api = Api(app, authorizations={
                'API-KEY': {
                    'type': 'apiKey',
                    'in': 'header',
                    'name': 'AUTH-TOKEN'
                }
            },
          security='API-KEY',
          default="Books",  # Default namespace
          title="Book Dataset",  # Documentation Title
          description="This is just a simple example to show how publish data as a service.")  # Documentation Description


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        token = request.headers.get('AUTH-TOKEN')
        print(token)
        if not token:
            abort(401, 'Authentication token is missing')

        try:
            user = auth.validate_token(token)
        except SignatureExpired as e:
            abort(401, e.message)
        except BadSignature as e:
            abort(401, e.message)

        return f(*args, **kwargs)

    return decorated


credential_model = api.model('credential', {
    'username': fields.String,
    'password': fields.String
})

credential_parser = reqparse.RequestParser()
credential_parser.add_argument('username', type=str)
credential_parser.add_argument('password', type=str)

@api.route('/token')
class Token(Resource):
    @api.response(200, 'Successful')
    @api.response(401, 'Denied')
    @api.doc(description="Generates a authentication token")
    @api.expect(credential_parser, validate=True)
    def get(self):
        args = credential_parser.parse_args()

        username = args.get('username')
        password = args.get('password')
        token = auth.generate_token(username)
        print(token)
        if (username == 'user' and password == 'user') or (username == 'landlord' and password == 'landlord'):
            return {"token": token}

        return {"message": "authorization has been refused for those credentials."}, 401


def read_csv(csv_file):

    return pd.read_csv(csv_file)


def clean_csv(dfm):

    dfm = dfm[~dfm['rent'].isnull()]
    dfm['rent'] = dfm['rent'].astype(str)
    dfm['rent'] = dfm['rent'].str.replace(',', '')#clean data
    dfm['rent'] = dfm['rent'].str.replace(' ', '')#clean data
    dfm['rent'] = dfm['rent'].astype(float)
    return dfm

def get_location_names(dfm,location):
    print(list(dfm.location.unique()))
    if location not in list(dfm.location.unique()):
        return False
    else:
        True

def topBottom(dfm, type, number, location):

    dfm = dfm.loc[dfm['location'] == location]
    dfm = dfm.sort_values(by=['rent'],ascending=False)

    if type == 'top':
        rent = list(dfm[:number]['rent'])
        address = list(dfm[:number]['address'])
    
    else:
        rent = list(dfm[-number:]['rent'])
        address = list(dfm[-number:]['address'])
    
    my_data = dict(zip(address, rent))

    my_data = {"location": location, "list_house" :my_data}
    return my_data


def inrange(dfm, maxx, minn, location):
    dfm = dfm.loc[dfm['location'] == location]
    dfm = dfm[(dfm['rent'] >= minn) & (dfm['rent'] <= maxx)]
    print(dfm)
    rent = list(dfm['rent'])
    address = list(dfm['address'])

    my_data = dict(zip(address, rent))

    my_data = {"location": location, "range_house" :my_data}

    return my_data

def getAvg(dfm, location):
    dfm = dfm.loc[dfm['location'] == location]
    total_rent = list(dfm['rent'])
    avg =  sum(total_rent)/len(total_rent)
    return {"location":location, "avg":avg}


@api.route('/rent/predict')# This function predicts the price of property using certain parameters
class predict(Resource):

    @api.response(201, 'Price Successfully Predicted')
    @api.response(404, 'Fail to Predict Price')
    @requires_auth
    def get(self):# Finding property price for tenant        
        location = request.args.get('location')
        #timePeriod = request.args.get('time')
        #my_data = {"location":location, "time" : timePeriod}
        return location

    @api.response(201, 'Price Successfully Predicted')
    @api.response(400, 'Validation Error')
    @requires_auth
    def post(self):# Get house specificatoin from landlord and return predicted price  
        resp = request.get_json()#Get request object
        #price = price_predicting.price_predicting(resp)
        #sendObj = {"sellPrice": str(price)}
        #return jsonify(sendObj)


@api.route('/avg/rent/<string:location>')
class average(Resource):

    @api.response(201, 'Price Successfully Predicted')
    @api.response(404, 'Fail to Predict Price')
    @requires_auth
    def get(self, location):      
        file = read_csv("rent_houseab.csv")
        dfm = clean_csv(file)
        location_exist = get_location_names(dfm, location)
        if(location_exist is False):
            return jsonify({"message":f'{location} does not exist'})
        my_data = getAvg(dfm, location)
        return jsonify(my_data)


@api.route('/rent/range/<string:location>') # this api returns the list of house in a location in the range(min, max)
class rangef(Resource):

    @api.response(201, 'Price Successfully Predicted')
    @api.response(404, 'Fail to Predict Price')
    @requires_auth
    def get(self, location):        
        maximum = int(request.args.get('max'))
        minimum = int(request.args.get('min'))
        file = read_csv("rent_house.csv")
        dfm = clean_csv(file)
        location_exist = get_location_names(dfm, location)
        if(location_exist is False):
            return jsonify({"message":f'{location} does not exist'})
        my_data = inrange(dfm, maximum, minimum, location)     
        return jsonify(my_data)

@api.route('/rent/<string:location>') # this api returns the number of most or least expensive house in a location
class topbottom(Resource):

    @api.response(201, 'Houses Identified')
    @api.response(404, 'Unable to Identify Houses')
    @requires_auth
    def get(self, location):        
        typeof = request.args.get('type')
        number = int(request.args.get('number'))
        file = read_csv("rent_house.csv")
        dfm = clean_csv(file)
        location_exist = get_location_names(dfm, location)
        if(location_exist is False):
            return jsonify({"message":f'{location} does not exist'})
        
        my_data = topBottom(dfm, typeof, number, location)    
        return jsonify(my_data)


if __name__ == '__main__':
    #run the application
    #dp = data_processing.Data_processor()
    #dp.data_processing()
    #dp.data_to_numeric()
    #dp.make_train_test()
    app.run(port = 5000, debug=True)
