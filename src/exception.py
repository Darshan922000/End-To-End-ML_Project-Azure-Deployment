import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()  
      
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}], error message [{2}]".format(
    file_name, exc_tb.tb_lineno,str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):         
        return self.error_message
    

#tb = traceback -> It contains detailed information about where the exception occurred in the code.
#the format() method in Python is used to insert values into a string at specific placeholders {}.\
#It allows for creating dynamic and formatted strings in a clean and readable way.
# def __str__(self): is a special method in Python used to define how an object is represented as a string. It is called when you use str(object) or print(object).



