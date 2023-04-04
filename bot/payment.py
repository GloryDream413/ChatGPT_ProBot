from typing import Dict
import datetime

import cryptomus


class CryptomusPayment:
    def __init__(self, api_key, merchant_id):
        self.api_key = api_key
        self.merchant_id = merchant_id

    def create_invoice(self, payment_id: int, amount: float, currency: str = "USD"):
        payment = cryptomus.Client.payment(self.api_key, self.merchant_id)

        r = payment.create({
            "amount": str(amount),
            "currency": currency,
            "order_id": str(payment_id),
            "substract": "0"
        })

        return r["url"], r["status"], datetime.datetime.fromtimestamp(r["expired_at"])

    def check_invoice_status(self, payment_id: int):
        payment = cryptomus.Client.payment(self.api_key, self.merchant_id)
        r = payment.info({"order_id": str(payment_id)})
        return r["status"] in {"paid", "paid_over"}
