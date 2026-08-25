from sqlalchemy.orm import Session
from src.db.models.user import User
from sqlalchemy import select
from uuid import UUID

def create_user(db : Session, email : str, password_hash : str) -> User :
    user = User(emial=email, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_email(db: Session, email : str) -> User | None:
    statement = select(User).where(User.email==email)
    return db.scalar(statement)

def get_user_by_id(db: Session, user_id :UUID) -> User | None:
    statement = select(User).where(User.id == user_id)
    return db.scalar(statement)
