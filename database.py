from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. The Connection URL
# Format: postgresql://user:password@address:port/dbname
SQLALCHEMY_DATABASE_URL = "postgresql://admin:password123@127.0.0.1:5433/predictive_maintenance"

# 2. Create the "Engine" (The actual connection)
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 3. Create a "SessionLocal" class
# Each time a user requests the API, we create a new 'Session' (conversation)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. The Base Class
# All our database tables will inherit from this class
Base = declarative_base()

# 5. Helper function to get the database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()