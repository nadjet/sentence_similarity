import logging

logger = logging.getLogger('converting transformer model to tensorflow')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)