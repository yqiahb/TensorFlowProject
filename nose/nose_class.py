'''

@author: huzq
'''


class TestClass():

    def setUp(self):
        print ("MyTestClass setup")

    def tearDown(self):
        print ("MyTestClass teardown")

    def Testfunc1(self):
        print ("this is Testfunc1")
        pass

    def test_func1(self):
        print ("this is test_func1")
        pass

    def Testfunc2(self):
        print ("this is Testfunc2")
        pass

    def test_func2(self):
        print ("this is test_func2")
        pass