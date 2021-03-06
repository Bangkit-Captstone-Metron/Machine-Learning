import os
import logging
from logging import Formatter, FileHandler
from flask import Flask, request, jsonify
import json

from fake_checker import process_image_url, process_image_file
from fake_checker_stride8 import process_image_url_stride8, process_image_file_stride8
from fake_checker_ela import process_image_url_ela, process_image_file_ela

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

app = Flask(__name__)
_VERSION = 1


@app.route('/v{}/health'.format(_VERSION), methods=['GET'])
def health():
    return 'OK'


@app.route('/v{}/fake_checker'.format(_VERSION), methods=['POST'])
def fake_checker():
    try:
        url = request.get_json()['image_url']
    except TypeError:
        try:
            data = json.loads(request.data.decode('utf-8'), encoding='utf-8')
            url = data['img_url']
        except:
            return jsonify({
                "error": "Could not get 'image_url' from the request object",
                "data": request.data
            })
    except:
        return jsonify({
            "error": "Non-TypeError. Please send {'image_url': 'http://.....'}",
            "data": request.data
        })

    # Process the image
    print("URL extracted:", url)
    try:
        output = process_image_url(url)
    except OSError:
        return jsonify({
            "error": "URL not recognized as image",
            "url": url
        })
    except:
        return jsonify({
            "error": "Unknown processing image",
            "request": request.data
        })
    app.logger.info(output)
    return jsonify({"fake": output})


@app.route('/v{}/file_fake_checker'.format(_VERSION), methods=['POST'])
def file_fake_checker():
    try:
        image = request.files['image']
    except:
        return jsonify({
            "error": "Please send multipart/form-data with image file",
            "data": request.data
        })

    try:
        output = process_image_file(image)
    except OSError:
        return jsonify({
            "error": "Not recognized as image"
        })
    except:
        return jsonify({
            "error": "Unknown processing image",
            "request": request.data
        })
    app.logger.info(output)
    return jsonify({"fake": output})


@app.route('/v{}/fake_checker'.format(_VERSION+1), methods=['POST'])
def fake_checker_stride8():
    try:
        url = request.get_json()['image_url']
    except TypeError:
        try:
            data = json.loads(request.data.decode('utf-8'), encoding='utf-8')
            url = data['img_url']
        except:
            return jsonify({
                "error": "Could not get 'image_url' from the request object",
                "data": request.data
            })
    except:
        return jsonify({
            "error": "Non-TypeError. Please send {'image_url': 'http://.....'}",
            "data": request.data
        })

    # Process the image
    print("URL extracted:", url)
    try:
        output = process_image_url_stride8(url)
    except OSError:
        return jsonify({
            "error": "URL not recognized as image",
            "url": url
        })
    except:
        return jsonify({
            "error": "Unknown processing image",
            "request": request.data
        })
    app.logger.info(output)
    return jsonify({"fake": output})


@app.route('/v{}/file_fake_checker'.format(_VERSION+1), methods=['POST'])
def file_fake_checker_stride8():
    try:
        image = request.files['image']
    except:
        return jsonify({
            "error": "Please send multipart/form-data with image file",
            "data": request.data
        })

    try:
        output = process_image_file_stride8(image)
    except OSError:
        return jsonify({
            "error": "Not recognized as image"
        })
    except:
        return jsonify({
            "error": "Unknown processing image",
            "request": request.data
        })
    app.logger.info(output)
    return jsonify({"fake": output})

@app.route('/v{}/fake_checker'.format(_VERSION+2), methods=['POST'])
def fake_checker_ela():
    try:
        url = request.get_json()['image_url']
    except TypeError:
        try:
            data = json.loads(request.data.decode('utf-8'), encoding='utf-8')
            url = data['img_url']
        except:
            return jsonify({
                "error": "Could not get 'image_url' from the request object",
                "data": request.data
            })
    except:
        return jsonify({
            "error": "Non-TypeError. Please send {'image_url': 'http://.....'}",
            "data": request.data
        })

    # Process the image
    print("URL extracted:", url)
    try:
        output = process_image_url_ela(url)
    except OSError:
        return jsonify({
            "error": "URL not recognized as image",
            "url": url
        })
    except:
        return jsonify({
            "error": "Unknown processing image",
            "request": request.data
        })
    app.logger.info(output)
    return jsonify(output)


@app.route('/v{}/file_fake_checker'.format(_VERSION+2), methods=['POST'])
def file_fake_checker_ela():
    try:
        image = request.files['image']
    except:
        return jsonify({
            "error": "Please send multipart/form-data with image file",
            "data": request.data
        })

    try:
        output = process_image_file_ela(image)
    except OSError:
        return jsonify({
            "error": "Not recognized as image"
        })
    except:
        return jsonify({
            "error": "Unknown processing image",
            "request": request.data
        })
    app.logger.info(output)
    return jsonify(output)


@app.errorhandler(500)
def internal_error(error):
    print("*** 500 ***\n{}".format(str(error)))  # ghetto logging


@app.errorhandler(404)
def not_found_error(error):
    print("*** 404 ***\n{}".format(str(error)))


if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: \
            %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Started app.py on port: {port}")
    app.run(host='0.0.0.0', port=port)
