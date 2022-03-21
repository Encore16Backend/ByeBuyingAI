import os.path

from flask import Flask, request
import numpy as np
import glob
from PIL import Image
import FeatureExtractor
import pandas as pd

app = Flask(__name__)

# 이미지 검색
extractor = FeatureExtractor.FeatureExtractor() # VGG16 특성 추출
# 미리 상품들에 대해 특성 추출한 파일들 로딩
features = []
names = []
for path in glob.glob('./npyfiles_fc2/*.npy'): # npyfiles_fc1/*.npy
    features.append(np.load(path))
    res = path.split('\\')[-1]
    names.append(res[:-4])
features = np.array(features)
print(len(features))

# 추천 시스템에서 사용할 사용자 * 상품 행렬 - 구매 유무
purchaseHistory = pd.read_csv('PurchaseHistory.csv', index_col='itemid')

@app.route('/') # Test
def start():  # put application's code here
    return 'Hello World!'


@app.route('/image', methods=['POST'])
def imageReceive():
    if request.method == 'POST':
        print("### Item Retrieval by Item Image ###")
        file_ = request.files['file']
        if file_ is None:
            return "Fail"

        # VGG16을 사용하여 이미지 특성 추출 후 유사도 측정
        # 빠르게 실행되고 정확도도 꽤 높은 듯하다
        img = Image.open(file_.stream).resize((224, 224)) # resize를 하면 extract 계산이 더 빠르게 됨
        print(img)
        feature = extractor.extract(img)
        distance = np.linalg.norm(features - feature, axis=1)
        ids = np.argsort(distance)[:30]
        result = ','.join([names[i] for i in ids])
        print(result)
        return result
    return "FAIL"

@app.route('/user', methods=['POST'])
def userAdd():
    global purchaseHistory
    if request.method == 'POST':
        username = request.get_json()['username']
        print(f"### Add New User Column to PurchaseHistory.csv -> {username} ###")
        purchaseHistory[username] = np.zeros(1080, dtype=np.uint8)
        try:
            purchaseHistory.to_csv('PurchaseHistory.csv')
            purchaseHistory = pd.read_csv('PurchaseHistory.csv', index_col='itemid')
        except:
            return "FAIL"
        return "SUCCESS"
    return "FAIL"

@app.route("/order", methods=['POST'])
def orderAdd():
    global purchaseHistory
    if request.method == 'POST':
        params = request.get_json()
        username = params['username']
        itemids = params['itemids']
        print(f"### User PurChase Item Check -> {username} ###")
        print(f'Item ID: {itemids}')
        try:
            purchaseHistory[username][itemids] = 1
            purchaseHistory.to_csv('PurchaseHistory.csv')
            purchaseHistory = pd.read_csv('PurchaseHistory.csv', index_col='itemid')
        except:
            return "FAIL"
        return "SUCCESS"
    return "FAIL"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # debug=True
