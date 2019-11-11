import unittest
import index


class TestHandlerCase(unittest.TestCase):

    def test_response(self):
        print("testing response.")
        result = index.handler({"testpost": {"body": "HelloWorld123"}}, None)
        print(result)
        self.assertEqual(result['statusCode'], 200)
        self.assertEqual(result['headers']['Content-Type'], 'application/json')
        self.assertIn('HelloWorld123', result['body'])


if __name__ == '__main__':
    unittest.main()
