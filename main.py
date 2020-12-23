import pandas as pd
import requests
import uvicorn
import json
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable

app = FastAPI()
client = TestClient(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# sim_scores daki indisler ile dönen kişi miktarını değiştirebilirsin.
def get_recommendations(users_frame, indices, title, cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:30]

    people_indices = [i[0] for i in sim_scores]

    return users_frame[0].iloc[people_indices]


def get_recommendations_based_on_cos_sim(username, user_infos):
    users_frame = pd.DataFrame(user_infos)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(users_frame[1])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    users_frame = users_frame.reset_index()
    indices = pd.Series(users_frame.index, index=users_frame[0])
    final_list = get_recommendations(users_frame, indices, username, cosine_sim2)

    # Json'a yazmak için
    result = {"matches": [i for i in final_list]}
    return result


# Burada vektörize edebilmek için kullanacağımız string verilerini concat ediyoruz
def extract_user_info(encoded_data):
    user_infos = []
    for i in encoded_data.get("matches", []):
        place_holder = [i["username"]]
        features_string = ' '.join([str(elem).replace(" ", "") for elem in i["features"]])
        hobbies_string = ' '.join([str(elem).replace(" ", "") for elem in i["hobbies"]])
        job_string = i["job"].replace(" ", "")
        school_string = i["school"].replace(" ", "")
        location_string = i["location"]["country"].replace(" ", "") + ' ' + i["location"]["city"].replace(" ", "")
        last_str = (features_string + ' ' + hobbies_string + ' ' + job_string + ' ' +
                    school_string + ' ' + location_string)
        place_holder.append(last_str)
        user_infos.append(place_holder)
    return user_infos


def get_sample_data(username):
    '''url = f'http://user-info-service.herokuapp.com/user/samples/{username}'
    sample_data = requests.get(url).json()
    return sample_data'''
    with open('mock/user_list.json') as json_file:
        data = json.load(json_file)
    return data


def get_user_recommendations(username):
    non_clear_data = get_sample_data(username)
    user_infos = extract_user_info(non_clear_data)
    result = get_recommendations_based_on_cos_sim(username, user_infos)
    return result


@app.get("/user/{username}/recommendations")
def match_users(match_result: dict = Depends(get_user_recommendations)):
    return match_result


if __name__ == "__main__":
    uvicorn.run(app)
