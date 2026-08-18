import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

from chainlit.data.storage_clients.base import BaseStorageClient, storage_expiry_time

logger = logging.getLogger(__name__)


class AzuriteBlobStorageClient(BaseStorageClient):
    """Azure blob client that respects APP_AZURE_STORAGE_CONNECTION_STRING for Azurite."""

    def __init__(self, container_name: str, connection_string: str):
        self.container_name = container_name
        self.connection_string = connection_string
        self.service_client = AsyncBlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.service_client.get_container_client(container_name)
        self.blob_endpoint = self._extract_blob_endpoint(connection_string)
        logger.info("AzuriteBlobStorageClient initialized")

    @staticmethod
    def _extract_blob_endpoint(connection_string: str) -> str:
        for part in connection_string.split(";"):
            if part.startswith("BlobEndpoint="):
                return part.split("=", 1)[1].rstrip("/")
        return "http://127.0.0.1:10000/devstoreaccount1"

    async def get_read_url(self, object_key: str) -> str:
        account_name = self.service_client.account_name
        account_key = self.service_client.credential.account_key
        start_time = datetime.now(tz=timezone.utc)
        expiry_time = start_time + timedelta(seconds=storage_expiry_time)
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self.container_name,
            blob_name=object_key,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            start=start_time,
            expiry=expiry_time,
        )
        return f"{self.blob_endpoint}/{self.container_name}/{object_key}?{sas_token}"

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        blob_client = self.container_client.get_blob_client(object_key)

        if isinstance(data, str):
            data = data.encode("utf-8")

        content_settings = ContentSettings(
            content_type=mime,
            content_disposition=content_disposition,
        )
        await blob_client.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=content_settings,
        )
        properties = await blob_client.get_blob_properties()

        return {
            "path": object_key,
            "object_key": object_key,
            "url": await self.get_read_url(object_key),
            "size": properties.size,
            "last_modified": properties.last_modified,
            "etag": properties.etag,
            "content_type": properties.content_settings.content_type,
        }

    async def delete_file(self, object_key: str) -> bool:
        try:
            blob_client = self.container_client.get_blob_client(blob=object_key)
            await blob_client.delete_blob()
            return True
        except Exception:
            logger.warning("Failed to delete blob %s", object_key, exc_info=True)
            return False

    async def close(self) -> None:
        await self.container_client.close()
        await self.service_client.close()


def build_storage_client() -> AzuriteBlobStorageClient | None:
    connection_string = os.getenv("APP_AZURE_STORAGE_CONNECTION_STRING")
    bucket_name = os.getenv("BUCKET_NAME")
    if not connection_string or not bucket_name:
        return None
    return AzuriteBlobStorageClient(
        container_name=bucket_name,
        connection_string=connection_string,
    )
