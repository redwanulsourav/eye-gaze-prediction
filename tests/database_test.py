import os
import unittest

from ..src import database_obj

class TestDatabaseMethods(unittest.TestCase):
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.db = database_obj.Database(f'{self.path}/res/runs')

    def test_find(self):
        run_id = self.db.find('s=1,l=1,t=0')
        assert run_id == ['0'], f'test_find failed, expected = 0, found {run_id}'

    def test_findmin(self):
        min_error = self.db.findmin('train','s=1,l=1,t=0')
        assert min_error == 45.33, f'test_findmin failed, expected = 45.33, found {min_error}'



obj = TestDatabaseMethods()
obj.test_find()
obj.test_findmin()
