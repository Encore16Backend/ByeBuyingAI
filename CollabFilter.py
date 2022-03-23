import pandas as pd
import numpy as np

# User-Based Collaborative-Filtering
class CollabFilter:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_T = data.replace(0, np.NaN).T

    def recommendation_system(self, username, n=11):
        user_similarity = self.data.corr()
        similar_users = user_similarity[:][username].sort_values(ascending=False)[1:n]
        purchased_item = self.data[self.data[username] != 0].index
        similar_user_items = self.data_T[self.data_T.index.isin(similar_users.index)].dropna(axis=1, how='all')
        similar_user_items.drop(purchased_item, inplace=True, axis=1, errors='ignore')
        item_score = {}
        for i in similar_user_items.columns:
            item_rating = similar_user_items[i]
            total = 0
            cnt = 0
            for j in similar_users.index:
                if not pd.isna(item_rating[j]):
                    score = similar_users[j] * item_rating[j]
                    total += score
                    cnt += 1
            item_score[i] = total / cnt
        item_score = pd.DataFrame(item_score.items(), columns=['item', 'score'])
        ranked_item_score = item_score.sort_values(by='score', ascending=False)
        return ','.join(list(map(str, ranked_item_score[:10]['item'].values)))