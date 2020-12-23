import unittest
import json
from fastapi.testclient import TestClient
from main import extract_user_info, app, get_user_recommendations, get_recommendations_based_on_cos_sim

client = TestClient(app)


class TestExtractUserInfo(unittest.TestCase):

    def test_success(self):
        data = retrieve_data()
        expected = retrieve_vector_data()
        actual = extract_user_info(data)
        self.assertEqual(actual, expected)

    def test_empty_dictionary(self):
        data = {}
        expected = []
        actual = extract_user_info(data)
        self.assertEqual(actual, expected)

    def test_invalid_datatype(self):
        data = "Invalid data type"
        self.assertRaises(AttributeError, extract_user_info, data)


def override_user_recommendations(username):
    data = retrieve_data()
    user_infos = extract_user_info(data)
    result = get_recommendations_based_on_cos_sim(username, user_infos)
    return result


app.dependency_overrides[get_user_recommendations] = override_user_recommendations


def test_for_user1():
    username = "testuser1"
    response = client.get(f"/user/{username}/recommendations")
    assert response.status_code == 200
    assert response.json() == {"matches": ["testuser3", "testuser4", "testuser2"]}
    app.dependency_overrides = {}


def test_for_user2():
    username = "testuser2"
    response = client.get(f"/user/{username}/recommendations")
    assert response.status_code == 200
    assert response.json() == {"matches": ["testuser4", "testuser3", "testuser1"]}
    app.dependency_overrides = {}


def test_for_user3():
    username = "testuser3"
    response = client.get(f"/user/{username}/recommendations")
    assert response.status_code == 200
    assert response.json() == {"matches": ["testuser1", "testuser2", "testuser4"]}
    app.dependency_overrides = {}


def test_for_user4():
    username = "testuser4"
    response = client.get(f"/user/{username}/recommendations")
    assert response.status_code == 200
    assert response.json() == {"matches": ["testuser2", "testuser1", "testuser3"]}
    app.dependency_overrides = {}


def retrieve_data():
    with open('mock/user_list.json') as json_file:
        data = json.load(json_file)
    return data


def retrieve_vector_data():
    with open('mock/user_list_vector.json') as json_file:
        expected = json.load(json_file)
    return expected


if __name__ == '__main__':
    unittest.main()
