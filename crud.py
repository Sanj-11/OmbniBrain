from sqlalchemy.orm import Session
import models
import security # Changed from import auth

# User CRUD
def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user_data: dict):
    hashed_password = security.get_password_hash(user_data["password"]) # Use security.get_password_hash
    db_user = models.User(username=user_data["username"], hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Dataset CRUD
def create_dataset(db: Session, dataset_data: dict, user_id: int):
    db_dataset = models.Dataset(**dataset_data, owner_id=user_id)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def get_datasets_by_user(db: Session, user_id: int):
    return db.query(models.Dataset).filter(models.Dataset.owner_id == user_id).all()

def get_dataset_by_id(db: Session, dataset_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()

def delete_dataset(db: Session, dataset_id: int):
    db_dataset = get_dataset_by_id(db, dataset_id)
    if db_dataset:
        db.delete(db_dataset)
        db.commit()
        return True
    return False