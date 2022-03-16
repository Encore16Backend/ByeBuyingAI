from flask import Flask, request
import numpy as np
import glob
from PIL import Image
import FeatureExtractor

app = Flask(__name__)
extractor = FeatureExtractor.FeatureExtractor() # VGG16 특성 추출

# 미리 상품들에 대해 특성 추출한 파일들 로딩
features = []
names = []
for path in glob.glob('./npyfiles/*.npy'):
    features.append(np.load(path))
    res = path.split('\\')[-1]
    names.append(res[:-4])
features = np.array(features)
print(len(features))

@app.route('/')
def start():  # put application's code here
    return 'Hello World!'


@app.route('/image', methods=['POST'])
def imageReceive():
    file_ = request.files['file']
    if file_ is None:
        return "Fail"

    # VGG16을 사용하여 이미지 특성 추출 후 유사도 측정
    # 빠르게 실행되고 정확도도 꽤 높은 듯하다
    img = Image.open(file_.stream)
    print(img)
    feature = extractor.extract(img)
    distance = np.linalg.norm(features - feature, axis=1)
    ids = np.argsort(distance)[:30]
    result = ','.join([names[i] for i in ids])

    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
