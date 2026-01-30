# filepath: app/utils/database.py

import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Your NeonDB Connection String
DATABASE_URL = "postgresql://neondb_owner:npg_wJUDWKt43Yva@ep-winter-band-ahm5u9ip-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SessionRecord(Base):
    """Tracks the lifecycle of an Agent session."""
    __tablename__ = "sessions"
    
    plan_id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True) # Stores the generated chat title
    e2b_sandbox_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    messages = relationship("MessageRecord", back_populates="session", cascade="all, delete-orphan")
    files = relationship("FileRegistryRecord", back_populates="session", cascade="all, delete-orphan")

class MessageRecord(Base):
    """Stores persistent chat history."""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(String, ForeignKey("sessions.plan_id"))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    session = relationship("SessionRecord", back_populates="messages")

class FileRegistryRecord(Base):
    """
    Stores metadata about files indexed in the intelligence pools.
    Content is stored on server disk to save DB space.
    """
    __tablename__ = "file_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(String, ForeignKey("sessions.plan_id"))
    filename = Column(String)
    namespace = Column(String) # 'vault', 'blueprint', 'lab', 'gallery'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    session = relationship("SessionRecord", back_populates="files")

# Create tables and handle migrations
def init_db():
    # 1. Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # 2. Robust Migration: Check for 'title' column using Inspector
    inspector = inspect(engine)
    columns = [c['name'] for c in inspector.get_columns('sessions')]
    
    if 'title' not in columns:
        with engine.begin() as conn: # begin() handles the transaction correctly
            print("Migration: Adding 'title' column to 'sessions' table...")
            conn.execute(text("ALTER TABLE sessions ADD COLUMN title VARCHAR"))
            print("Migration: 'title' column added successfully.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()