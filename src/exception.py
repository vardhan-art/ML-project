import traceback   # Used to extract detailed stack trace information
import sys         # Gives access to system-specific parameters and functions (like current exception info)

# Custom exception class that extends Python's built-in Exception class
class CustomException(Exception):

    # Constructor method for initializing the custom exception
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Call the base Exception constructor
        # Create a detailed error message using the static method below
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    # Static method to build a detailed error message with file name & line number
    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        
        # Get the traceback object from the current exception
        _, _, exc_tb = sys.exc_info()

        # Extract the file name where the exception occurred
        file_name = exc_tb.tb_frame.f_code.co_filename

        # Extract the line number where the exception occurred
        line_number = exc_tb.tb_lineno

        # Return a formatted error string with file name, line number, and the message
        return f"Error in {file_name} , line {line_number} : {error_message}"
    
    # When the exception object is converted to string, return the detailed error message
    def __str__(self):
        return self.error_message


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message