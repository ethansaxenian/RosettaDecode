try:
    try:
        pass
    except (MyException1, MyOtherException):
        pass
    except SomeOtherException:
        pass
finally:
    do_some_cleanup() # run in any case, whether any exceptions were thrown or not
