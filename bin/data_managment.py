from icecream import ic
import os.path
import csv
import pandas as pd
import yaml


ic.enable()

nome_f = os.path.basename(__file__)

class DataManager:

    def __init__(self, csv_file, database, char):
        self.config_file = "./config.yml"
        self.items = self.read_yaml(csv_file, database)
        self.hyper_model_main = self.items[0]
        self.hyper_model = self.items[1]
        self.path = self.items[2]
        self.files = self.items[3]

        if isinstance(self.files, dict):
            self.input_csv_raw = f'{self.path}{self.files.get("csv")[0]}'
            self.input_csv = f'{self.path}{self.files.get("csv")[1]}'
            self.output_csv = f'{self.path}{self.files.get("csv")[2]}'
            self.chat_logs = f'{self.path}{self.files.get("csv")[3]}'
            return self.read_csv()
        elif ".db" in self.files:
            self.database_file = f"{self.path}{self.files}"
            return self.database_chat()
        else:
            self.parquet_dir = f"{self.path}{self.files}"
            if char is None:
                char = "a"
            return self.read_parquet(csv_file, char)


    def read_yaml(self, csv_file=False, database=False):
        items = []
        with open(self.config_file, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file)
            hyper_model_main = data["hyperparameters"]["model_main"]
            items.append(hyper_model_main)  # 0
            hyper_model = data["hyperparameters"]["model"]
            items.append(hyper_model)  # 1

            datasets = data["datasets"]
            path = datasets["path"]
            items.append(path)  # 2

            if csv_file is True and database is False:
                files = datasets["file_names"]["csv"]
                print("Using csv file")
            
            elif csv_file is False and database is False:
                files = datasets["file_names"]["parquets"]
                print("Using parquet file")

            elif csv_file is False and database is True:
                files = datasets["file_names"]["db"]
                print("Using database")
            items.append(files)  # 3
            return items
        
    def return_hyper(self):
        return self.hyper_model_main, self.hyper_model

    def read_csv():
        data = []
        with open("../data/input.csv", 'r', encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                msg = row["Message"]
                data.append(msg)
        
        return '\n'.join(data)


    def csv_chat(data=None):
        if data is None:
            with open("../data/chat_logs.csv", 'r', encoding="utf-8") as file:
                reader = csv.DictReader(file)
        else:
            with open("../data/chat_logs.csv", 'a', encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["Time", "Input", "Output"])
                writer.writerow(data)

        return reader