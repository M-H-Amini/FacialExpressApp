from json import load, dump
import numpy as np
from dataclasses import dataclass
import os
from datetime import datetime

@dataclass
class MHRecord:
    date: datetime
    feeling: str
    feeling_index: int
    confidence: float

class MHHandler:
    def __init__(self, filename='logs.json'):
        self.filename = filename
        self.records = []
        os.path.exists(self.filename) and self.load()
        self.dict = {'records': self.records}

    def addRecord(self, record):
        self.records.append({
            'date': record.date.strftime('%d/%m/%Y %H:%M:%S'),
            'feeling': record.feeling,
            'feeling_index': record.feeling_index,
            'confidence': record.confidence
            })

    def save(self):
        with open(self.filename, 'w') as f:
            dump(self.dict, f)
    
    def load(self):
        with open(self.filename, 'r') as f:
            self.dict = load(f)
            self.records = self.dict['records']
