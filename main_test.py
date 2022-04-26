import unittest
import main


class MainTest(unittest.TestCase):
    def test_helloworld(self):
        ret = main.helloworld("Hong")
        self.assertEqual(ret, "Hello World! Hong!")


if __name__ == "__main__":
    unittest.main()