import logging
from .jhmdb_eval import save_jhmdb_results


def jhmdb_evaluation(dataset, predictions, output_folder, **_):
    logger = logging.getLogger("hit.inference")
    logger.info("performing jhmdb evaluation.")
    return save_jhmdb_results(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
