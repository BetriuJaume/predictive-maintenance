import logging

class ScrapyLogger(object):
    """
    Represents a Logger object.
    """
    def __init__(self, name="logger", level="debug", logging_file = False):
        """
        Args:
            -name (str): Custom name of Logger. 
                    Default: "logger"

            -level (str): Level or severity of the events they are used to track
                    Options: "debug", "info", "warning", "error". 
                    Default: "debug"

            -logging_file (bool): If True, a file with logging output will be created. 
                                    Default: False.

        Ex:
            >>>log = MatchingLogger(name= "logger_ml", level="debug", logging_file=True)
        """

        self.logger = logging.getLogger(name) #Logger object

        self.levels = {"debug":logging.DEBUG, "info":logging.INFO, "warning":logging.WARNING, "error":logging.ERROR} #Levels types
        try:
            self.logger.setLevel(self.levels[level]) 
        except:
            self.logger.error("Invalid level option")

        sh = logging.StreamHandler() #sends logging output to streams (Terminal)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s') #Create a format
        sh.setFormatter(formatter) #Add formatter
        self.logger.addHandler(sh) #Add sh to logger

        if logging_file:
            fh = logging.FileHandler('%s.log' % name, 'w') #sends logging output to a disk file (.log file)
            fh.setFormatter(formatter) #Add formatter
            self.logger.addHandler(fh) #Add fh to logger

    #Logging Events
    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg,exc_info=1)