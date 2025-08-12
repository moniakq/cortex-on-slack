"""
This module provides a thread-safe class to manage and refresh Snowflake connections.
"""

import os
import threading
from datetime import datetime, timedelta
import snowflake.connector


class SnowflakeConnectionManager:
    """
    A thread-safe class to manage and refresh Snowflake connections.

    This class handles the lifecycle of a Snowflake connection, ensuring that it is
    always valid and refreshed periodically to prevent timeouts. It is designed
    to be used in a multi-threaded environment, such as a web server or a
    long-running application.
    """

    def __init__(self, logger, max_age_hours=4, initial_connection=None):
        """
        Initializes the connection manager.

        Args:
            logger: A logger instance for logging messages.
            max_age_hours: The maximum age of a connection in hours before it is
                           considered stale and needs to be refreshed.
            initial_connection: An optional initial Snowflake connection to use.
                                If not provided, a new connection will be created
                                when `get_connection` is first called.
        """
        self.log = logger
        self.log.info(
            f"Connection manager initialized with a max age of {max_age_hours} hours."
        )

        # Connection details from environment variables
        self.spcs_token_file = "/snowflake/session/token"
        self.snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.snowflake_host = os.getenv("SNOWFLAKE_HOST")
        self.warehouse = os.getenv("WAREHOUSE")
        self.role = os.getenv("ROLE")
        self.database = os.getenv("DATABASE")
        self.schema = os.getenv("SCHEMA")
        self._connection = None
        self._connection_timestamp = None

        if initial_connection:
            self._connection = initial_connection
            self._connection_timestamp = datetime.now()
        else:
            self._connection = None
            self._connection_timestamp = None

        self._lock = threading.Lock()
        self.max_age = timedelta(hours=max_age_hours)

    def _create_connection(self):
        """
        Creates and returns a new Snowflake connection using the SPCS token.

        This method reads the Snowflake session token from the file specified by
        `self.spcs_token_file` and uses it to create a new database connection.
        """
        self.log.info("Creating a new Snowflake database connection...")

        if not os.path.exists(self.spcs_token_file):
            raise FileNotFoundError(
                f"SPCS token file not found at {self.spcs_token_file}."
            )

        # Read the SPCS token from the file system.
        with open(self.spcs_token_file, "r") as f:
            token = f.read()

        try:
            conn = snowflake.connector.connect(
                authenticator="oauth",
                token=token,
                account=self.snowflake_account,
                host=self.snowflake_host,
                warehouse=self.warehouse,
                role=self.role,
                database=self.database,
                schema=self.schema,
            )
            self.log.info("New Snowflake connection established successfully.")
            return conn
        except Exception as e:
            self.log.critical(f"Fatal error creating new Snowflake connection: {e}")
            raise

    def get_connection(self):
        """
        Returns a valid connection, creating a new one if the old one is stale.

        This method checks if the current connection is still valid based on its
        age. If the connection is stale or does not exist, it creates a new one.
        This method is thread-safe.
        """
        with self._lock:
            # Check if the connection is stale.
            is_stale = (
                self._connection is None
                or self._connection_timestamp is None
                or datetime.now() - self._connection_timestamp > self.max_age
            )

            if is_stale:
                self.log.info(
                    "Current connection is stale or non-existent. Refreshing..."
                )
                if self._connection and not self._connection.is_closed():
                    self._connection.close()
                    self.log.info("Closed stale connection.")

                self._connection = self._create_connection()
                self._connection_timestamp = datetime.now()
            else:
                self.log.info("Returning existing, valid connection.")

            return self._connection
