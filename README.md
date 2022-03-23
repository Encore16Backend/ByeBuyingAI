# ByeBuyingAI - Server

### Image Retrieval
* CBIR
  * VggNet을 사용하여 상품들의 특성 추출
  * 검색하고자 하는 이미지의 특성 추출
  * 상품들과 검색 이미지와의 유사도 계산
  * 상위 20개의 상품을 출력

### Recommendation System
* User-based Collaborative-Filtering
  * 사용자는 상품이 마음에 든다면 여러 번을 구매할 것이라는 가정을 세우고 개발
  * 사용자간 피어슨 상관계수를 사용하여 유사한 구매 경향을 가진 사용자를 추출
  * 상위 10명의 사용자들의 구매 상품에 대해서 해당 사용자가 자주 구매할 것 같은 상위 10개의 상품을 추천
