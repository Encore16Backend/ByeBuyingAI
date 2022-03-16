import numpy as np
import pymysql
import pandas as pd
from PIL import Image
import FeatureExtractor

conn = pymysql.connect(host="127.0.0.1", user="root", password='qwer1234', db='byebuying', charset='utf8')
cur = conn.cursor()
sql = 'select A.itemid, A.itemname, B.imgpath' \
       + ' from item A, image B, item_images C' \
       + ' where A.itemid=C.item_itemid and B.imgid=C.images_imgid' \
       + ' GROUP BY a.itemid'
cur.execute(sql)
result = cur.fetchall()
cur.close()

itemId = []
itemName = []
imagePath = []
for Id, name, path in result:
    itemId.append(Id)
    itemName.append(name)
    imagePath.append(path)
df = pd.DataFrame({'번호':itemId, '상품명':itemName, '이미지경로':imagePath})

descriptor = FeatureExtractor.FeatureExtractor()

for i in range(len(df)):
    itemid, name, path = df.iloc[i]
    print(name)
    features = descriptor.extract(img=Image.open(path))
    feature_path = f'{itemid}.npy'
    np.save(feature_path, features)