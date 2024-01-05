from flask import Blueprint, json, Response

status_api = Blueprint('status_api', __name__)

@status_api.route("/", methods=['GET'])
def get_index():
   return get_status()

@status_api.route("/status/", methods=['GET'])
def get_status():
    jd = {'status': 'OK'}
    data = json.dumps(jd)
    return Response(data, status=200, mimetype='application/json')
