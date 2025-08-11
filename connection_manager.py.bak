import os
import threading
from datetime import datetime, timedelta
import snowflake.connector

class SnowflakeConnectionManager:
    """A thread-safe class to manage and refresh Snowflake connections."""

    def __init__(self, logger, max_age_hours=4, initial_connection=None):
        self.log = logger
        self.log.info(f"Connection manager initialized with a max age of {max_age_hours} hours.")
        
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
        """Creates and returns a new Snowflake connection using the SPCS token."""
        self.log.info("Creating a new Snowflake database connection...")
        
        if not os.path.exists(self.spcs_token_file):
            raise FileNotFoundError(f"SPCS token file not found at {self.spcs_token_file}.")
        with open(self.spcs_token_file, 'r') as f:
            token = f.read()

        try:
            conn = snowflake.connector.connect(
                authenticator='oauth',
                token=token,
                account=self.snowflake_account,
                host=self.snowflake_host,
                warehouse=self.warehouse,
                role=self.role,
                database=self.database,
                schema=self.schema
            )
            self.log.info("New Snowflake connection established successfully.")
            return conn
        except Exception as e:
            self.log.critical(f"Fatal error creating new Snowflake connection: {e}")
            raise

    def get_connection(self):
        """Returns a valid connection, creating a new one if the old one is stale."""
        with self._lock:
            is_stale = (
                self._connection is None or 
                self._connection_timestamp is None or
                datetime.now() - self._connection_timestamp > self.max_age
            )
            
            if is_stale:
                self.log.info("Current connection is stale or non-existent. Refreshing...")
                if self._connection and not self._connection.is_closed():
                    self._connection.close()
                    self.log.info("Closed stale connection.")
                
                self._connection = self._create_connection()
                self._connection_timestamp = datetime.now()
            else:
                self.log.info("Returning existing, valid connection.")

            return self._connection