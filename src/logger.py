#import Python's built-in logging library
import logging

# Import Os Utilities for filesystem operation(like making folders)
import os

#Import datatime to timestamp log files with today's  date
from datetime import datetime

#Name of the folder where logs will be stored
LOGS_DIR="Logs"

#Create the Logs folder if it doesn't alreday exits
os.makedirs(LOGS_DIR,exist_ok=True)

#Build a log file path like:Logs/Log_2025-09-19 LOg(change daily)
LOGS_FILE=os.path.join(
    LOGS_DIR,
    f"Log_{datetime.now().strftime('%Y-%m-%d')}.Log"
)

#Configure the ROOT logger once for the whole programe
logging.basicConfig(
    #write all logs to this file
    filename=LOGS_FILE,
    #Log messagge format:
    #-%(asctime)s :timestamp
    #-%(levelname)s:log level (INFO,INFO ,ERRRO etc)
    #-%(message):the log message text
    format='%(asctime)s-%(levelname)s-%(message)s',
    #Maximun level to record (INFO and above)
    level=logging.INFO
)

def get_logger(name):
    """
    Returns a named logger that inherits the root configuration above.
    Use different names per module (e.g., _name_) to identify sources.
    """
    # Get (or create) a logger with the given name
    logger = logging.getLogger(name)

    # Ensure this logger emits INFO and above (can be customized per logger)
    logger.setLevel(logging.INFO)

    # Return the configured named logger
    return logger
