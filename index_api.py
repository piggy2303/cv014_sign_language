import requests
from flask import (
    Flask,
    make_response,
    jsonify,
    request,
    send_file,
    send_from_directory,
    abort,
)
from flask_cors import CORS
from io import BytesIO
import base64
import sys
import os
import time
import argparse

from predict import cv014

app = Flask(__name__)
CORS(app)


def response_body(status, data):
    status = int(status)
    if status == 0:
        status = "error"
    if status == 1:
        status = "success"

    response_body_json = {
        "status": status,
        "data": data
    }
    res = make_response(jsonify(response_body_json), 200)
    return res


@app.route("/cv014", methods=['POST'])
def cv014_api():
    # get data from user
    try:
        body_req = request.get_json(force=True)
        input_source = body_req["input_source"]
    except Exception as error:
        error = str(error)
        print(error)
        return response_body(status=0, data=error)

    try:
        result = cv014(input_source)
        return response_body(status=1, data=result)
    except Exception as error:
        error = str(error)
        print(error)
        return response_body(status=0, data=error)


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5018)
