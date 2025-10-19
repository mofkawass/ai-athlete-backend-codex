# app/signed_urls.py
from datetime import timedelta
from google.cloud import storage

def get_v4_signed_put_url(storage_client: storage.Client, bucket_name: str, object_name: str,
                          content_type: str = "video/mp4", minutes: int = 10) -> str:
    """
    Create a V4 signed URL for uploading (HTTP PUT) to GCS.
    The client must send the SAME Content-Type when uploading.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="PUT",
        content_type=content_type,
        headers={"Content-Type": content_type},
    )
    return url

def get_v4_signed_get_url(storage_client: storage.Client, bucket_name: str, object_name: str,
                          minutes: int = 60) -> str:
    """
    Create a V4 signed URL for downloading/streaming (HTTP GET) from GCS.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="GET",
    )
    return url
