import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)
logger = logging.getLogger("polars_bio_bench")
logger.setLevel(logging.INFO)
