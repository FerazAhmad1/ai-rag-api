from database import Base
from sqlalchemy import Column,Integer,String,Boolean

class Users(Base):
    __tablename__="users"
    id=Column(Integer,primary_key=True,index=True)
    name=Column(String(100),unique=True,index=True)
    email=Column(String(255),unique=True,index=True,nullable=False)
    phone_number = Column(String(15),unique=True,index=True,nullable=False)
    country_code = Column(Integer,nullable=False,index=True)
    password = Column(String)