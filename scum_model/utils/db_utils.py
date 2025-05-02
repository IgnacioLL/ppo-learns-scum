import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C
from db.db import MongoDBManager

def extract_all_models_in_db(mongodb_manager: MongoDBManager):
    params = mongodb_manager.find_many(C.NAME_COLLECTION_CHECKPOINTS)
    return list({param['model_id'] for param in params})
