"""db_cleanup.py — teardown test data by tag."""

import logging

logger = logging.getLogger(__name__)


def cleanup_mongo(mongo_client, db_name: str, case_id: str) -> None:
    db = mongo_client[db_name]
    for col in ["documents", "summaries", "audit_log", "Document Storage"]:
        try:
            result = db[col].delete_many({"case_id": {"$regex": "^test-case-"}})
            if result.deleted_count:
                logger.debug("Cleaned %d docs from %s", result.deleted_count, col)
        except Exception as exc:
            logger.warning("Cleanup failed for collection %s: %s", col, exc)


def cleanup_qdrant(qdrant_client, case_id: str, collection: str = "case_docs") -> None:
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            ),
        )
    except Exception as exc:
        logger.warning("Qdrant cleanup failed for case_id=%s: %s", case_id, exc)


def cleanup_minio(minio_client, bucket: str, prefix: str) -> None:
    try:
        objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
        for obj in objects:
            minio_client.remove_object(bucket, obj.object_name)
    except Exception as exc:
        logger.warning("MinIO cleanup failed for prefix=%s: %s", prefix, exc)
