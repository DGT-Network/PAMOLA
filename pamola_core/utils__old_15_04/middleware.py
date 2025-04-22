from django.http import Http404, HttpResponse
from django.core.exceptions import PermissionDenied, SuspiciousFileOperation, RequestDataTooBig
from loguru import logger
import sys
import os
from datetime import datetime


class ExceptionHandler(object):
    def __init__(self, get_response):
        self.get_response = get_response
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name = f"app_Test{timestamp}.log"
        logger.add(
        sys.stderr, format="{time:MMMM D, YYYY > HH:mm:ss!UTC} | {level} | {message} | {extra}", serialize=True)
        logger.add(f"{os.path.abspath(os.curdir)}/logs/{log_name}",
                serialize=True)
        
    def __call__(self, request):        
        response = self.get_response(request)
        return response     

    def process_exception(self, request, e):
        
        if isinstance(e, SuspiciousFileOperation):
            logger.exception(extra=e)
            return HttpResponse('invalid path')
        elif isinstance(e, PermissionDenied):
            return HttpResponse('Forbidden', status=403)
        logger.exception(msg="Exception on request processing", extra=e)
        if isinstance(e, RequestDataTooBig):
            logger.exception(
                msg="==== RequestDataTooBig todo: raise error with the limit ======== ", extra=e)
            raise e
        if isinstance(e, Http404):
            raise e
        if isinstance(e, Exception):
            logger.exception(msg="Exception on request processing", extra=e)
            raise e
        return HttpResponse(str(e), status=500)