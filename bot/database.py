from typing import Optional, Any

import pymongo
import uuid
from datetime import datetime

from bot import config


class Database:
    def __init__(self):
        self.client = pymongo.MongoClient(config.mongodb_uri)
        self.db = self.client["chatgpt_telegram_bot"]

        self.user_collection = self.db["user"]
        self.dialog_collection = self.db["dialog"]
        self.payment_collection = self.db["payment"]
        self.newsletter_collection = self.db["newsletter"]

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        if self.user_collection.count_documents({"_id": user_id}) > 0:
            return True
        else:
            if raise_exception:
                raise ValueError(f"User {user_id} does not exist")
            else:
                return False

    def check_if_payment_exists(self, payment_id: int, raise_exception: bool = False):
        if self.payment_collection.count_documents({"_id": payment_id}) > 0:
            return True
        else:
            if raise_exception:
                raise ValueError(f"Payment {payment_id} does not exist")
            else:
                return False

    def check_if_newsletter_exists(self, newsletter_id: int, raise_exception: bool = False):
        if self.newsletter_collection.count_documents({"_id": newsletter_id}) > 0:
            return True
        else:
            if raise_exception:
                raise ValueError(f"Newsletter {newsletter_id} does not exist")
            else:
                return False
        
    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        initial_token_balance: int = 5000,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        user_dict = {
            "_id": user_id,
            "chat_id": chat_id,

            "username": username,
            "first_name": first_name,
            "last_name": last_name,

            "last_interaction": datetime.now(),
            "first_seen": datetime.now(),
            
            "current_dialog_id": None,
            "current_chat_mode": "assistant",

            "n_used_tokens": 0,
            "token_balance": initial_token_balance
        }

        if not self.check_if_user_exists(user_id):
            self.user_collection.insert_one(user_dict)
            
    def start_new_dialog(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        dialog_id = str(uuid.uuid4())
        dialog_dict = {
            "_id": dialog_id,
            "user_id": user_id,
            "chat_mode": self.get_user_attribute(user_id, "current_chat_mode"),
            "start_time": datetime.now(),
            "messages": []
        }

        # add new dialog
        self.dialog_collection.insert_one(dialog_dict)

        # update user's current dialog
        self.user_collection.update_one(
            {"_id": user_id},
            {"$set": {"current_dialog_id": dialog_id}}
        )

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        user_dict = self.user_collection.find_one({"_id": user_id})

        if key not in user_dict:
            raise ValueError(f"User {user_id} does not have a value for {key}")

        return user_dict[key]

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.user_collection.update_one({"_id": user_id}, {"$set": {key: value}})

    def check_if_user_attribute_exists(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        user_dict = self.user_collection.find_one({"_id": user_id})
        return key in user_dict

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")
            if dialog_id is None:
                return []

        dialog_dict = self.dialog_collection.find_one({"_id": dialog_id, "user_id": user_id})               
        return dialog_dict["messages"]

    def set_dialog_messages(self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")
        
        self.dialog_collection.update_one(
            {"_id": dialog_id, "user_id": user_id},
            {"$set": {"messages": dialog_messages}}
        )

    def count_documents_in_collection(self, collection_name: str):
        return self.db[collection_name].count_documents({})

    def get_new_unique_payment_id(self):
        if self.payment_collection.count_documents({}) == 0:
            payment_id = 0
        else:
            payment_id = 1 + self.payment_collection.find_one(sort=[("_id", pymongo.DESCENDING)])["_id"]

        return payment_id

    def add_new_payment(
        self,
        payment_id: int,
        payment_method: str,  # like: cryptomus, cards
        payment_method_type: str,  # like: cryptomus, telegram_payments
        product_key: str,
        user_id: int,
        amount: float,
        currency: str,
        status: str,
        invoice_url: str,
        n_tokens_to_add: int,
        expired_at: datetime
    ):
        payment_dict = {
            "_id": payment_id,
            "payment_method": payment_method,
            "payment_method_type": payment_method_type,
            "product_key": product_key,
            "user_id": user_id,
            "amount": amount,
            "currency": currency,
            "status": status,
            "invoice_url": invoice_url,
            "n_tokens_to_add": n_tokens_to_add,
            "expired_at": expired_at,
            "created_at": datetime.now(),
            "are_tokens_added": False
        }

        self.payment_collection.insert_one(payment_dict)

    def get_payment_attribute(self, payment_id: int, key: str):
        self.check_if_payment_exists(payment_id, raise_exception=True)
        payment_dict = self.payment_collection.find_one({"_id": payment_id})

        if key not in payment_dict:
            raise ValueError(f"Payment {payment_id} does not have a value for {key}")

        return payment_dict[key]

    def set_payment_attribute(self, payment_id: int, key: str, value: Any):
        self.check_if_payment_exists(payment_id, raise_exception=True)
        self.payment_collection.update_one({"_id": payment_id}, {"$set": {key: value}})

    def get_all_not_expried_payment_ids(self):
        payments_find = self.payment_collection.find({
            "$and": [{"expired_at": {"$gt": datetime.now()}}, {"status": {"$ne": "paid"}}]
        })
        return [x["_id"] for x in payments_find]

    def create_newsletter(self, newsletter_id: str):
        if not self.check_if_newsletter_exists(newsletter_id):
            newsletter_dict = {
                "_id": newsletter_id,
                "already_sent_to_user_ids": [],
                "created_at": datetime.now()
            }

            self.newsletter_collection.insert_one(newsletter_dict)

    def add_user_to_newsletter(self, newsletter_id: str, user_id: int):
        self.check_if_newsletter_exists(newsletter_id, raise_exception=True)
        self.check_if_user_exists(user_id, raise_exception=True)

        newsletter_dict = self.newsletter_collection.find_one({"_id": newsletter_id})
        if user_id not in newsletter_dict["already_sent_to_user_ids"]:
            self.newsletter_collection.update_one(
                {"_id": newsletter_id},
                {"$push": {"already_sent_to_user_ids": user_id}}
            )

    def get_newsletter_attribute(self, newsletter_id: int, key: str):
        self.check_if_newsletter_exists(newsletter_id, raise_exception=True)
        newsletter_dict = self.newsletter_collection.find_one({"_id": newsletter_id})

        if key not in newsletter_dict:
            raise ValueError(f"Newsletter {newsletter_id} does not have a value for {key}")

        return newsletter_dict[key]