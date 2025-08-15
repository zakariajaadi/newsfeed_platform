import logging

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.getLogger('faiss.loader').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
