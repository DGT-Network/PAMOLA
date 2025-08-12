import os
from loguru import logger
from datetime import datetime
import sys
import time


class Logger():
    def __init__():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name = f"app_{timestamp}.log"
        logger.add(
            sys.stderr, format="{time:MMMM D, YYYY > HH:mm:ss!UTC} | {level} | {message} | {extra}", serialize=True)
        logger.add(f"{os.path.abspath(os.curdir)}/logs/{log_name}",
                   serialize=True)

    def logtrace(message, data=None):
        logger.trace(message, extra=data)

    def logdebug(message, data=None):
        logger.debug(message, extra=data)

    def logsuccess(message, data=None):
        logger.success(message, extra=data)

    def logInformation(message, data=None):
        logger.info(message, extra=data)

    def logWarning(message, data=None):
        logger.warning(message, extra=data)

    def logError(message, data=None):
        logger.error(message, extra=data)

    def logCritical(message, data=None):
        logger.critical(message, extra=data)

    '''
   log exception with a custom message
   example:
    youfunction():
    try:
    //do some thing
    except Exception as ex:
     Logger.logException('your custom message' ex)
    '''
    def logException(message='', data=None):
        logger.exception(message, extra=data)

        '''
    a decorator to count time from start and finish
    to use is try 
    @timing
    yourfunction():
    do some thing
    the out put will be  
    start function yourfunction at 2025-02-24 10:05:53
    finish function yourfunction at 2025-02-24 10:05:56 took xxx.yyy ms
    '''
    def timing(f):
        def wrap(*args, **kwargs):
            time1 = time.time()
            print('start function {:s} at {:s}'.format(
                f.__name__, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time1))))
            ret = f(*args, **kwargs)
            time2 = time.time()
            print('finish function {:s} at {:s} took {:.3f} ms'.format(f.__name__, time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(time2)), (time2-time1)*1000.0))
            return ret
        return wrap
