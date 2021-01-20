import argparse
from fastai.vision.all import *


import base64
from PIL import Image
from io import BytesIO


def base64_to_cv(data_source):
    data_source = data_source.split(",")[-1]
    data_source = base64.b64decode(data_source)
    data_source = np.fromstring(data_source, dtype=np.uint8)
    print(data_source)
    # img = cv2.imdecode(data_source, 1)
    # return img


def base64_to_file(data_input):
    data_input = data_input.split(",")[-1]
    img = Image.open(BytesIO(base64.b64decode(data_input)))
    time_now = str(time.time())
    source = "test/"+time_now+".jpg"

    img = img.save(source)
    return source


def np_to_base64(data):
    im = Image.fromarray(data)
    buff = BytesIO()
    im.save(buff, format="JPEG")
    mask_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return mask_base64


def cv014(data_input):
    source = base64_to_file(data_input)

    learn_inf = load_learner("sign_alphabet_export_201128.pkl")
    pred, pred_idx, probs = learn_inf.predict(source)
    detail = (f'{probs[pred_idx]:.04f}')

    print(detail)

    result = {
        "prediction": pred,
        "probability": detail
    }

    os.remove(source)
    return result
