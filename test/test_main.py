import unittest
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from main import extract_user_info, get_recommendations, get_recommendations_based_on_cos_sim


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
