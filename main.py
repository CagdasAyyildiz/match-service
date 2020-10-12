import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# sim_scores daki indisler ile dönen kişi miktarını değiştirebilirsin.
def get_recommendations(users_frame, indices, title, cosine_sim):
    # Get the index of the people that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all people with that people
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 30 most similar people
    sim_scores = sim_scores[1:30]

    # Get the movie indices
    people_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return users_frame[0].iloc[people_indices]


user_infos = []


# Burada vektörize edebilmek için kullanacağımız string verilerini concat ediyoruz
def extract_user_info(encoded_data):
    for i in encoded_data["matches"]:
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


@app.get("/user/{username}/recommendations")
def match_users(username):
    non_clear_data = requests.get(f'http://user-info-service.herokuapp.com/user/samples/{username}').json()

    extract_user_info(non_clear_data)
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


if __name__ == "__main__":
    uvicorn.run(app, port=5000)
