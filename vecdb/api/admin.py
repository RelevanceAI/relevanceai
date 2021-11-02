from ..base import Base

class Admin(Base):
    def request_read_api_key(self, read_username: str):
        """Creates a read only key for your project. Make sure to save the api key somewhere safe. When doing a search the admin username should still be used.
        """
        return self.make_http_request(
            "admin/request_read_api_key",
            method="POST",
            parameters={
                "read_username": read_username
            }
        )
