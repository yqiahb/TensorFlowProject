import logging

log = logging.getLogger(__name__)


def test_learn_3():
    print("test_lean_3")
    pass


def test_lean_4():
    print("test_learn_2")


def test_lean_5():
    print("test_learn_5")


def setUp():
    print("0002 test setUp")


def tearDown():
    print("0002 test teardown")


test_learn_3.teardown = test_lean_5.teardown = tearDown