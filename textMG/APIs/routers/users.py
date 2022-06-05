#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/6/5 15:39
"""
from fastapi import APIRouter
from textMG.APIs.api_loggers.api_logger import logger
from textMG.APIs.base_models.inquiries_model import UserRequest, UserUpdateRequest
from textMG.APIs.base_models.responses import Response
from textMG.APIs.db.db_models import User
from textMG.APIs.db.database import Database
from textMG.APIs.routers.users_token import get_password_hash
from sqlalchemy import and_, desc

router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

database = Database()
engine = database.get_db_connection()


@router.post("/", response_description="user data added into the database")
async def add_user(req: UserRequest):
    """add new a new user"""
    new_user = User()
    new_user.first_name = req.first_name
    new_user.last_name = req.last_name
    new_user.email = req.email
    new_user.hashed_password = get_password_hash(req.password)
    new_user.is_active = req.is_active
    session = database.get_db_session(engine)
    session.add(new_user)
    session.flush()
    # get id of the inserted user
    session.refresh(new_user, attribute_names=['id'])
    data = {"user_id": new_user.id}
    session.commit()
    session.close()
    return Response(data, 200, "user added successfully.", False)


@router.put("/")
async def update_user(req: UserUpdateRequest):
    """update an existing user"""
    user_id = req.id
    session = database.get_db_session(engine)
    try:
        is_user_updated = session.query(User).filter(User.id == user_id).update({
            User.first_name: req.first_name, User.last_name: req.last_name,
            User.email: req.email,
            User.is_active: req.is_active,
        }, synchronize_session=False)
        session.flush()
        session.commit()
        response_msg = "user updated successfully"
        response_code = 200
        error = False
        if is_user_updated == 1:
            # After successful update, retrieve updated data from db
            data = session.query(User).filter(
                User.id == user_id).one()

        elif is_user_updated == 0:
            response_msg = "user id not updated. No user found with this id :" + \
                str(user_id)
            error = True
            data = None
        return Response(data, response_code, response_msg, error)
    except Exception as e:
        logger.critical('this is exception', exc_info=1)


@router.delete("/")
async def delete_user(user_id: str):
    """delete an existing user"""
    session = database.get_db_session(engine)
    try:
        is_user_updated = session.query(User).filter(and_(User.id == user_id, User.is_active == True)).update({
            User.is_active: False}, synchronize_session=False)
        session.flush()
        session.commit()
        response_msg = "user is deleted successfully"
        response_code = 200
        error = False
        data = {"user_id": user_id}
        if is_user_updated == 0:
            response_msg = "user not deleted. No user found with this id :" + \
                str(user_id)
            error = True
            data = None
        return Response(data, response_code, response_msg, error)
    except Exception as ex:
        logger.critical('this is exception', exc_info=1)


@router.get("/")
async def get_user(user_id: str):
    """retrieve an user from db"""
    session = database.get_db_session(engine)
    response_message = "user retrieved successfully"
    data = None
    try:
        data = session.query(User).filter(
            and_(User.id == user_id, User.is_active == True)).one()
    except Exception as e:
        logger.critical('user Not found exception', exc_info=1)
        response_message = "user Not found"
    error = False
    return Response(data, 200, response_message, error)


@router.get("/all")
async def get_all_users(page_size: int, page: int):
    """get all users"""
    session = database.get_db_session(engine)
    data = session.query(User).filter(User.is_active == True).order_by(
        desc(User.created_at)).limit(page_size).offset((page-1)*page_size).all()
    return Response(data, 200, "users retrieved successfully.", False)